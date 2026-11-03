# %% [markdown]
# # Sokoban World Model from a Language Model
#
# **Question:** If we fine-tune a small pretrained LM on `(board, action) -> next board`
# transitions, does it learn Sokoban's *consequences* well enough to plan with? And if we
# also teach it *value* ("how many moves until solved, or is this already lost?"), does that
# turn into *strategy* that avoids irreversible mistakes?
#
# One model (`Qwen/Qwen3-0.6B-Base`, full fine-tune) is trained on three tagged tasks:
#
# | Task | Prompt ends with | Target | What it measures |
# |---|---|---|---|
# | Dynamics | `ACTION RIGHT\nNEXT\n` | full next board | consequences |
# | Value | `MOVES_TO_SOLVE ` | `17` or `DEAD` | strategy / foresight |
# | Policy | `BEST_ACTION ` | `LEFT` | imitation baseline |
#
# Then a single fixed beam-search planner is run with every combination of
# {exact simulator, learned dynamics} x {hand heuristic, learned value}, plus the direct
# policy. Differences between rows isolate where competence comes from.
#
# **Kaggle:** Accelerator = *GPU T4 x2* (one GPU is used), Internet = *On*.
# Expected wall-clock on a T4: ~5 min data, ~60 min training, ~40 min evaluation (estimates).
# Set `SOKOBAN_SMOKE=1` to run a tiny CPU end-to-end check with a random-init model.

# %%
import json
import math
import multiprocessing as mp
import os
import random
import re
import subprocess
import sys
import time
from collections import Counter, defaultdict, deque
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

import transformers

if tuple(int(x) for x in transformers.__version__.split(".")[:2]) < (4, 51):
    # Qwen3 support landed in transformers 4.51.
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", "transformers>=4.51"], check=True)
    raise SystemExit("Upgraded transformers; restart the kernel and run again.")

from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

SMOKE = os.environ.get("SOKOBAN_SMOKE") == "1"
OUT = "/kaggle/working" if os.path.isdir("/kaggle/working") else os.path.abspath("sokoban_cwm_out")
os.makedirs(OUT, exist_ok=True)

CFG = dict(
    model_id="Qwen/Qwen3-0.6B-Base",
    resume_from=None,           # path to a saved checkpoint dir to skip training
    seed=20261018,
    size=8,                     # 8x8 including the outer wall -> 6x6 interior
    train_boxes=2,
    ood_boxes=3,                # out-of-distribution: one more box, same board size
    wall_p=0.12,
    id_len=(8, 30),             # accepted optimal solution length (moves)
    ood_len=(10, 36),
    max_states=150_000,
    n_train_levels=1200,
    n_dev_levels=30,
    n_test_id_levels=60,
    n_test_ood_levels=40,
    train_per_level=dict(dyn=12, value=20, policy=10),  # dyn states are branched x4 actions
    eval_per_level=dict(dyn=4, value=8, policy=6),
    dead_frac=0.35,
    lr=5e-5,
    batch_size=16,
    epochs=1,
    warmup=100,
    log_every=50,
    eval_every=500,
    gen_batch=96,
    fewshot_n=120,
    rollout_seqs=64,
    rollout_len=16,
    plan_levels=40,
    beam_width=6,
    workers=max(1, (os.cpu_count() or 2)),
)
if SMOKE:
    CFG.update(
        n_train_levels=24, n_dev_levels=4, n_test_id_levels=4, n_test_ood_levels=3,
        train_per_level=dict(dyn=4, value=4, policy=4),
        eval_per_level=dict(dyn=2, value=4, policy=3),
        batch_size=8, warmup=5, log_every=5, eval_every=20, gen_batch=16,
        fewshot_n=8, rollout_seqs=6, rollout_len=4, plan_levels=3, beam_width=3,
        ood_len=(6, 20), max_states=60_000,
    )

random.seed(CFG["seed"])
np.random.seed(CFG["seed"])
torch.manual_seed(CFG["seed"])
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# T4 (sm75) has no bf16 tensor cores: use fp16 autocast + loss scaling there.
if DEVICE == "cuda":
    AMP_DTYPE = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
else:
    AMP_DTYPE = None
print("device:", DEVICE, torch.cuda.get_device_name(0) if DEVICE == "cuda" else "", "amp:", AMP_DTYPE)
print("smoke:", SMOKE, "out:", OUT)

# %% [markdown]
# ## 1. Deterministic Sokoban
#
# A level holds the static walls and goals; a state is `(player, sorted boxes)`.
# `explore` enumerates every state reachable from the start, then runs a reverse BFS from
# solved states. That gives **exact** labels for every reachable state: optimal moves to
# solve, or `DEAD` when no sequence of actions can ever solve it (an irreversible mistake).

# %%
ACTIONS = ("UP", "DOWN", "LEFT", "RIGHT")
DELTA = {"UP": (-1, 0), "DOWN": (1, 0), "LEFT": (0, -1), "RIGHT": (0, 1)}
INF = 10**9


@dataclass(frozen=True)
class Level:
    size: int
    walls: frozenset
    goals: frozenset


def step(level, state, action):
    (pr, pc), boxes = state
    dr, dc = DELTA[action]
    ahead = (pr + dr, pc + dc)
    if ahead in level.walls:
        return state, "blocked_wall"
    if ahead in boxes:
        beyond = (ahead[0] + dr, ahead[1] + dc)
        if beyond in level.walls or beyond in boxes:
            return state, "blocked_box"
        moved = tuple(sorted(beyond if b == ahead else b for b in boxes))
        return (ahead, moved), ("push_on_goal" if beyond in level.goals else "push")
    return (ahead, boxes), "move"


def is_solved(level, state):
    return all(b in level.goals for b in state[1])


def corner_deadlock(level, state):
    """The textbook 'obvious' deadlock: a box off-goal wedged in a corner."""
    w = level.walls
    for r, c in state[1]:
        if (r, c) in level.goals:
            continue
        if ((r - 1, c) in w or (r + 1, c) in w) and ((r, c - 1) in w or (r, c + 1) in w):
            return True
    return False


def explore(level, start, max_states):
    index, states, children = {start: 0}, [start], []
    i = 0
    while i < len(states):
        row = []
        for action in ACTIONS:
            nxt, _ = step(level, states[i], action)
            j = index.get(nxt)
            if j is None:
                j = len(states)
                if j >= max_states:
                    return None
                index[nxt] = j
                states.append(nxt)
            row.append(j)
        children.append(row)
        i += 1
    parents = [[] for _ in states]
    for i, row in enumerate(children):
        for j in row:
            if j != i:
                parents[j].append(i)
    dist = [INF] * len(states)
    queue = deque()
    for i, s in enumerate(states):
        if is_solved(level, s):
            dist[i] = 0
            queue.append(i)
    while queue:
        j = queue.popleft()
        for i in parents[j]:
            if dist[i] == INF:
                dist[i] = dist[j] + 1
                queue.append(i)
    return states, children, dist


def optimal_actions(children, dist, i):
    return [a for a, j in zip(ACTIONS, children[i]) if dist[i] < INF and dist[j] == dist[i] - 1]


TILE = {"wall": "#", "floor": ".", "goal": "g", "player": "P", "player_goal": "p", "box": "B", "box_goal": "X"}
CHAR_SET = set(TILE.values())


def render(level, state):
    player, boxes = state
    rows = []
    for r in range(level.size):
        cells = []
        for c in range(level.size):
            pos, goal = (r, c), (r, c) in level.goals
            if pos in level.walls:
                ch = "#"
            elif pos in boxes:
                ch = "X" if goal else "B"
            elif pos == player:
                ch = "p" if goal else "P"
            else:
                ch = "g" if goal else "."
            cells.append(ch)
        rows.append(" ".join(cells))
    return "\n".join(rows) + "\n"


def parse_board(text, level):
    """Strictly parse a generated board. Returns (state, None) or (None, reason)."""
    lines = [line for line in text.split("\n") if line.strip()][: level.size]
    if len(lines) != level.size:
        return None, "format"
    walls, goals, boxes, players = set(), set(), [], []
    for r, line in enumerate(lines):
        cells = line.split()
        if len(cells) != level.size or any(ch not in CHAR_SET for ch in cells):
            return None, "format"
        for c, ch in enumerate(cells):
            pos = (r, c)
            if ch == "#":
                walls.add(pos)
            if ch in "gpX":
                goals.add(pos)
            if ch in "BX":
                boxes.append(pos)
            if ch in "Pp":
                players.append(pos)
    if walls != level.walls or goals != level.goals:
        return None, "static_changed"
    if len(players) != 1 or len(boxes) != len(level.goals):
        return None, "object_count"
    return (players[0], tuple(sorted(boxes))), None


def layout_key(level):
    """Canonical key of walls+goals under the 8 square symmetries (for leak-free splits)."""
    n = level.size - 1
    maps = [
        lambda r, c: (r, c), lambda r, c: (c, n - r), lambda r, c: (n - r, n - c), lambda r, c: (n - c, r),
        lambda r, c: (r, n - c), lambda r, c: (n - r, c), lambda r, c: (c, r), lambda r, c: (n - c, n - r),
    ]
    return min(
        (tuple(sorted(f(*p) for p in level.walls)), tuple(sorted(f(*p) for p in level.goals))) for f in maps
    )


# %% [markdown]
# ## 2. Procedural levels with exact labels
#
# Levels are random walls plus random box/goal/player placement, kept only if solvable with an
# optimal length in range. Each worker samples labelled states and discards the full graph.
# States come half from noisy-optimal play (relevant positions) and half uniformly from the
# reachable set (including many dead ones), so the model sees both good and ruinous moves.
# Every dynamics state is branched across **all four actions** for counterfactual coverage.

# %%
def random_level(rng, size, n_boxes, wall_p):
    border = {(r, c) for r in range(size) for c in range(size) if r in (0, size - 1) or c in (0, size - 1)}
    interior = [(r, c) for r in range(1, size - 1) for c in range(1, size - 1)]
    walls = border | {p for p in interior if rng.random() < wall_p}
    floor = [p for p in interior if p not in walls]
    if len(floor) < 3 * n_boxes + 6:
        return None, None
    seen, stack = {floor[0]}, [floor[0]]
    floor_set = set(floor)
    while stack:
        r, c = stack.pop()
        for dr, dc in DELTA.values():
            q = (r + dr, c + dc)
            if q in floor_set and q not in seen:
                seen.add(q)
                stack.append(q)
    if len(seen) != len(floor):
        return None, None
    goals = rng.sample(floor, n_boxes)
    boxes = rng.sample([p for p in floor if p not in goals], n_boxes)
    player = rng.choice([p for p in floor if p not in boxes])
    level = Level(size, frozenset(walls), frozenset(goals))
    return level, (player, tuple(sorted(boxes)))


def sample_indices(rng, pool, k):
    if not pool or k <= 0:
        return []
    return [rng.choice(pool) for _ in range(k)]


def build_level(args):
    seed, n_boxes, len_range, counts = args
    rng = random.Random(seed)
    for _ in range(2000):
        level, start = random_level(rng, CFG["size"], n_boxes, CFG["wall_p"])
        if level is None:
            continue
        graph = explore(level, start, CFG["max_states"])
        if graph is None:
            continue
        states, children, dist = graph
        if not (len_range[0] <= dist[0] <= len_range[1]):
            continue
        # Noisy-optimal play from the start: the positions a reasonable player visits.
        near = set()
        for _ in range(12):
            i = 0
            for _ in range(dist[0] + 6):
                near.add(i)
                opt = optimal_actions(children, dist, i)
                action = rng.choice(opt) if opt and rng.random() > 0.3 else rng.choice(ACTIONS)
                i = children[i][ACTIONS.index(action)]
        near = sorted(near)
        every = list(range(len(states)))
        alive = [i for i in every if 0 < dist[i] < INF]
        dead = [i for i in every if dist[i] == INF]
        near_alive = [i for i in near if 0 < dist[i] < INF]

        k = counts["dyn"]
        dyn = sample_indices(rng, near, k // 2) + sample_indices(rng, every, k - k // 2)
        k = counts["value"]
        n_dead = round(k * CFG["dead_frac"]) if dead else 0
        value = (sample_indices(rng, dead, n_dead) + sample_indices(rng, near_alive, (k - n_dead) // 2)
                 + sample_indices(rng, alive, k - n_dead - (k - n_dead) // 2))
        k = counts["policy"]
        policy = sample_indices(rng, near_alive, k // 2) + sample_indices(rng, alive, k - k // 2)
        return dict(
            level=level, start=start, opt_len=dist[0], key=layout_key(level), n_states=len(states),
            dyn=[states[i] for i in dyn],
            value=[(states[i], None if dist[i] == INF else dist[i]) for i in value],
            policy=[(states[i], optimal_actions(children, dist, i)) for i in policy],
        )
    return None


def generate_split(name, n, n_boxes, len_range, counts, seed, exclude):
    ctx = mp.get_context("fork")
    jobs = [(seed * 100_003 + i, n_boxes, len_range, counts) for i in range(int(n * 1.15) + 4)]
    t0 = time.time()
    with ctx.Pool(CFG["workers"]) as pool:
        bundles = [b for b in pool.map(build_level, jobs, chunksize=4) if b is not None]
    keep, keys = [], set()
    for b in bundles:
        if b["key"] in exclude or b["key"] in keys:
            continue
        keys.add(b["key"])
        keep.append(b)
        if len(keep) == n:
            break
    print(f"{name:9s} {len(keep):5d} levels  boxes={n_boxes}  "
          f"opt_len mean={np.mean([b['opt_len'] for b in keep]):.1f}  {time.time() - t0:.0f}s")
    return keep, keys


S = CFG["seed"]
train, train_keys = generate_split("train", CFG["n_train_levels"], CFG["train_boxes"], CFG["id_len"],
                                   CFG["train_per_level"], S + 1, set())
dev, dev_keys = generate_split("dev", CFG["n_dev_levels"], CFG["train_boxes"], CFG["id_len"],
                               CFG["eval_per_level"], S + 2, train_keys)
test_id, id_keys = generate_split("test_id", CFG["n_test_id_levels"], CFG["train_boxes"], CFG["id_len"],
                                  CFG["eval_per_level"], S + 3, train_keys | dev_keys)
test_ood, _ = generate_split("test_ood", CFG["n_test_ood_levels"], CFG["ood_boxes"], CFG["ood_len"],
                             CFG["eval_per_level"], S + 4, train_keys | dev_keys | id_keys)
SPLITS = {"ID (2 boxes)": test_id, "OOD (3 boxes)": test_ood}

print("\nexample training level (optimal", train[0]["opt_len"], "moves):")
print(render(train[0]["level"], train[0]["start"]))

# %% [markdown]
# ## 3. Text formats
#
# Cells are space-separated so the Qwen tokenizer emits **one token per cell** (unspaced rows
# merge into irregular chunks like `.B` and `PB`). Legend: `#` wall, `.` floor, `g` goal,
# `P` player, `p` player on goal, `B` box, `X` box on goal.

# %%
def dyn_prompt(level, state, action):
    return f"TASK DYNAMICS\nBOARD\n{render(level, state)}ACTION {action}\nNEXT\n"


def value_prompt(level, state):
    return f"TASK VALUE\nBOARD\n{render(level, state)}MOVES_TO_SOLVE "


def policy_prompt(level, state):
    return f"TASK POLICY\nBOARD\n{render(level, state)}BEST_ACTION "


def value_text(dist):
    return "DEAD" if dist is None else str(dist)


def parse_value(text):
    tok = (text.strip().split() or [""])[0]
    if tok == "DEAD":
        return INF
    return int(tok) if tok.isdigit() else None


def parse_action(text):
    tok = (text.strip().split() or [""])[0]
    return tok if tok in ACTIONS else None


rows = []
for b in train:
    lvl = b["level"]
    for s in b["dyn"]:
        for a in ACTIONS:
            rows.append(("dyn", dyn_prompt(lvl, s, a), render(lvl, step(lvl, s, a)[0])))
    for s, d in b["value"]:
        rows.append(("value", value_prompt(lvl, s), value_text(d)))
    for s, opt in b["policy"]:
        # Several actions can be optimal; train on a random one, score against the set.
        rows.append(("policy", policy_prompt(lvl, s), random.choice(opt)))
random.shuffle(rows)
print(Counter(r[0] for r in rows))

# %% [markdown]
# ## 4. Model
#
# `Qwen3-0.6B-Base`: a standard dense transformer (runs on T4 with SDPA), Apache-2.0, not
# gated, and digits tokenize individually. Full fine-tuning with fp32 master weights and fp16
# autocast fits comfortably in a T4's 15 GB because the loss only projects the
# **target positions** through the 152k-word output layer.

# %%
tokenizer = AutoTokenizer.from_pretrained(CFG["model_id"])
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
EOS = tokenizer.eos_token_id

# Recent transformers load checkpoints in their saved dtype (bf16 for Qwen3). Training needs
# fp32 master weights: a T4 has no bf16, and the fp16 GradScaler can't unscale bf16 grads.
TV = tuple(int(x) for x in transformers.__version__.split(".")[:2])
FP32 = {"dtype": torch.float32} if TV >= (4, 56) else {"torch_dtype": torch.float32}

if CFG["resume_from"]:
    model = AutoModelForCausalLM.from_pretrained(CFG["resume_from"], **FP32)
elif SMOKE:
    cfg = AutoConfig.from_pretrained(CFG["model_id"])
    for k, v in dict(hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                     num_attention_heads=2, num_key_value_heads=1, head_dim=32).items():
        setattr(cfg, k, v)
    model = AutoModelForCausalLM.from_config(cfg)
else:
    model = AutoModelForCausalLM.from_pretrained(CFG["model_id"], **FP32)
model.float().to(DEVICE)  # from_config also inherits the checkpoint's bf16 dtype
assert next(model.parameters()).dtype == torch.float32, next(model.parameters()).dtype
print(f"params: {sum(p.numel() for p in model.parameters()) / 1e6:.0f}M")

BOARD_TOKENS = len(tokenizer(render(train[0]["level"], train[0]["start"]), add_special_tokens=False)["input_ids"])


def autocast():
    return torch.autocast("cuda", dtype=AMP_DTYPE) if AMP_DTYPE else torch.autocast("cpu", enabled=False)


@torch.no_grad()
def generate(prompts, max_new_tokens):
    """Greedy batched generation; returns completions in input order."""
    model.eval()
    tokenizer.padding_side = "left"
    order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
    out = [None] * len(prompts)
    for start in range(0, len(order), CFG["gen_batch"]):
        idx = order[start:start + CFG["gen_batch"]]
        enc = tokenizer([prompts[i] for i in idx], return_tensors="pt", padding=True,
                        add_special_tokens=False).to(DEVICE)
        with autocast():
            gen = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                                 eos_token_id=EOS, pad_token_id=tokenizer.pad_token_id)
        texts = tokenizer.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        for i, t in zip(idx, texts):
            out[i] = t
    return out


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (centre - half, centre + half)


def eval_dynamics(bundles, label, prompt_fn=dyn_prompt, max_new=None):
    rows_ = [(b["level"], s, a) for b in bundles for s in b["dyn"] for a in ACTIONS]
    texts = generate([prompt_fn(l, s, a) for l, s, a in rows_], max_new or BOARD_TOKENS + 4)
    records = []
    for (l, s, a), t in zip(rows_, texts):
        true_next, event = step(l, s, a)
        pred, reason = parse_board(t, l)
        records.append(dict(split=label, event=event, exact=pred == true_next,
                            copy_exact=s == true_next, invalid=reason or ""))
    return pd.DataFrame(records)


# %% [markdown]
# ## 5. Baseline: the pretrained model, few-shot
#
# What does pretraining alone know about pushing boxes? Three worked examples plus a legend.

# %%
LEGEND = ("Sokoban. Symbols: # wall, . floor, g goal, P player, p player on goal, B box, X box on goal. "
          "The player moves one cell. Walking into a box pushes it one cell if the cell beyond is "
          "floor or goal; otherwise nothing moves. Walls never move.\n\n")
shots = [r for r in rows if r[0] == "dyn"][:3]
FEWSHOT_PREFIX = LEGEND + "".join(p + t + "\n" for _, p, t in shots)


def fewshot_prompt(level, state, action):
    return FEWSHOT_PREFIX + dyn_prompt(level, state, action)


fs_bundles = [dict(b, dyn=b["dyn"][:1]) for b in test_id][: max(1, CFG["fewshot_n"] // 4)]
pre_cfg = dict(CFG)
CFG["gen_batch"] = max(4, CFG["gen_batch"] // 4)  # long prompts
fewshot_df = eval_dynamics(fs_bundles, "ID (2 boxes)", fewshot_prompt)
CFG["gen_batch"] = pre_cfg["gen_batch"]
print(f"pretrained few-shot exact next board: {fewshot_df.exact.mean():.1%} "
      f"(copy-current-board baseline: {fewshot_df.copy_exact.mean():.1%}), n={len(fewshot_df)}")

# %% [markdown]
# ## 6. Fine-tune (masked causal-LM loss on the target only)

# %%
def encode(row):
    task, prompt, target = row
    p = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    t = tokenizer(target, add_special_tokens=False)["input_ids"] + [EOS]
    return task, p + t, [-100] * len(p) + t


TASK_ID = {"dyn": 0, "value": 1, "policy": 2}


def make_batches(encoded, bs, rng):
    # Length-grouped batches cut padding; batch order is then shuffled.
    idx = list(range(len(encoded)))
    rng.shuffle(idx)
    batches = []
    for start in range(0, len(idx), bs * 50):
        chunk = sorted(idx[start:start + bs * 50], key=lambda i: len(encoded[i][1]))
        batches += [chunk[i:i + bs] for i in range(0, len(chunk), bs)]
    rng.shuffle(batches)
    return batches


def collate(items):
    width = max(len(ids) for _, ids, _ in items)
    ids = torch.full((len(items), width), tokenizer.pad_token_id)
    labels = torch.full((len(items), width), -100)
    mask = torch.zeros((len(items), width), dtype=torch.long)
    for i, (_, x, y) in enumerate(items):
        ids[i, :len(x)] = torch.tensor(x)
        labels[i, :len(y)] = torch.tensor(y)
        mask[i, :len(x)] = 1
    tasks = torch.tensor([TASK_ID[t] for t, _, _ in items])
    return ids.to(DEVICE), labels.to(DEVICE), mask.to(DEVICE), tasks.to(DEVICE)


def masked_loss(ids, labels, mask, tasks):
    decoder = model.get_decoder() if hasattr(model, "get_decoder") else model.model
    hidden = decoder(input_ids=ids, attention_mask=mask).last_hidden_state
    shift_labels = labels[:, 1:]
    keep = shift_labels != -100
    logits = model.get_output_embeddings()(hidden[:, :-1][keep]).float()
    token_loss = F.cross_entropy(logits, shift_labels[keep], reduction="none")
    row = keep.nonzero()[:, 0]
    per_task = {}
    for name, tid in TASK_ID.items():
        sel = tasks[row] == tid
        if sel.any():
            per_task[name] = token_loss[sel].mean().item()
    return token_loss.mean(), per_task


history = []
if CFG["resume_from"]:
    print("resumed from", CFG["resume_from"], "- skipping training")
else:
    encoded = [encode(r) for r in rows]
    rng = random.Random(CFG["seed"])
    batches = [b for _ in range(CFG["epochs"]) for b in make_batches(encoded, CFG["batch_size"], rng)]
    total = len(batches)
    print(f"{len(encoded)} rows, {sum(len(e[1]) for e in encoded) / 1e6:.1f}M tokens, {total} steps")

    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    opt = torch.optim.AdamW(model.parameters(), lr=CFG["lr"], betas=(0.9, 0.95), weight_decay=0.0,
                            fused=DEVICE == "cuda")
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / CFG["warmup"])
        * 0.5 * (1 + math.cos(math.pi * min(1.0, s / total))))
    scaler = torch.amp.GradScaler("cuda", enabled=AMP_DTYPE == torch.float16)
    dev_probe = [dict(b, dyn=b["dyn"][:1]) for b in dev]
    t0, running = time.time(), defaultdict(list)
    for step_i, batch in enumerate(batches, 1):
        model.train()
        ids, labels, mask, tasks = collate([encoded[i] for i in batch])
        with autocast():
            loss, per_task = masked_loss(ids, labels, mask, tasks)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)
        sched.step()
        for k, v in per_task.items():
            running[k].append(v)
        if step_i % CFG["log_every"] == 0 or step_i == total:
            entry = dict(step=step_i, **{k: float(np.mean(v)) for k, v in running.items()})
            running.clear()
            if step_i % CFG["eval_every"] == 0 or step_i == total:
                model.config.use_cache = True
                entry["dev_dyn_exact"] = eval_dynamics(dev_probe, "dev").exact.mean()
                model.config.use_cache = False
            history.append(entry)
            eta = (time.time() - t0) / step_i * (total - step_i) / 60
            print(" ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in entry.items()),
                  f"eta={eta:.0f}m")
    model.gradient_checkpointing_disable()
    model.config.use_cache = True
    del opt, scaler, sched  # sched references opt; keep it and AdamW state stays on the GPU
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    print(f"training took {(time.time() - t0) / 60:.1f} min")

# Inference in half precision if it stays numerically finite on a real batch.
if DEVICE == "cuda":
    model.to(AMP_DTYPE)
    probe = tokenizer([dyn_prompt(dev[0]["level"], dev[0]["start"], "UP")], return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        if not torch.isfinite(model(**probe).logits).all():
            print("half-precision logits not finite; falling back to fp32 inference")
            model.float()
            AMP_DTYPE = None
if not SMOKE:
    model.save_pretrained(os.path.join(OUT, "ckpt"))
    tokenizer.save_pretrained(os.path.join(OUT, "ckpt"))

# %% [markdown]
# ## 7. Consequences: one-step dynamics, by event type
#
# The copy baseline ("nothing changes") is right exactly when the move is blocked, so it shows
# which events are trivial. The interesting rows are `push` and `push_on_goal`.

# %%
dyn_df = pd.concat([eval_dynamics(b, name) for name, b in SPLITS.items()])
dyn_table = dyn_df.groupby(["split", "event"]).agg(
    n=("exact", "size"), model_exact=("exact", "mean"), copy_exact=("copy_exact", "mean"))
print(dyn_table.to_string(float_format="{:.1%}".format))
print("\ninvalid outputs:", dict(Counter(dyn_df.invalid[dyn_df.invalid != ""])))

# %% [markdown]
# ## 8. Free-running rollouts
#
# Only the first board comes from the environment; afterwards the model consumes its own
# predictions. A trajectory counts as alive at step *h* only if every board up to *h* is exact.

# %%
def rollout_survival(bundles, label):
    rng = random.Random(CFG["seed"] + 7)
    cases = []
    for _ in range(CFG["rollout_seqs"]):
        b = rng.choice(bundles)
        cases.append(dict(level=b["level"], true=b["start"], pred=b["start"], alive=True,
                          actions=[rng.choice(ACTIONS) for _ in range(CFG["rollout_len"])]))
    records = []
    for h in range(CFG["rollout_len"]):
        live = [c for c in cases if c["alive"]]
        texts = generate([dyn_prompt(c["level"], c["pred"], c["actions"][h]) for c in live], BOARD_TOKENS + 4)
        for c, t in zip(live, texts):
            c["true"] = step(c["level"], c["true"], c["actions"][h])[0]
            c["pred"], _ = parse_board(t, c["level"])
            c["alive"] = c["pred"] == c["true"]
        records.append(dict(split=label, horizon=h + 1, survival=np.mean([c["alive"] for c in cases])))
    return pd.DataFrame(records)


rollout_df = pd.concat([rollout_survival(b, name) for name, b in SPLITS.items()])
print(rollout_df.pivot(index="horizon", columns="split", values="survival").to_string(float_format="{:.1%}".format))

# %% [markdown]
# ## 9. Strategy: value and deadlock foresight
#
# A *corner* deadlock is visible locally. A *non-corner* deadlock (boxes jammed against each
# other or along a wall with no goal) requires reasoning about future consequences. The corner
# rule gets 100% of the first kind and 0% of the second by construction; the gap is what the
# model's foresight adds.

# %%
def eval_value(bundles, label):
    items = [(b["level"], s, d) for b in bundles for s, d in b["value"]]
    texts = generate([value_prompt(l, s) for l, s, _ in items], 6)
    records = []
    for (l, s, d), t in zip(items, texts):
        pred = parse_value(t)
        dead = d is None
        kind = "alive" if not dead else ("dead_corner" if corner_deadlock(l, s) else "dead_noncorner")
        records.append(dict(split=label, kind=kind, true=d, pred=pred, invalid=pred is None,
                            pred_dead=pred == INF, rule_dead=corner_deadlock(l, s)))
    return pd.DataFrame(records)


value_df = pd.concat([eval_value(b, name) for name, b in SPLITS.items()])
value_table = value_df.groupby(["split", "kind"]).agg(
    n=("pred_dead", "size"), model_says_dead=("pred_dead", "mean"), corner_rule_says_dead=("rule_dead", "mean"))
print(value_table.to_string(float_format="{:.1%}".format))
alive_ok = value_df[(value_df.kind == "alive") & value_df.pred.notna() & (value_df.pred < INF)]
for name, g in alive_ok.groupby("split"):
    print(f"{name}: moves-to-solve MAE={np.mean(np.abs(g.pred - g.true)):.2f} "
          f"spearman={g.pred.rank().corr(g.true.rank()):.2f}  (n={len(g)})")

# %% [markdown]
# ## 10. Direct policy: imitation without a world model

# %%
def eval_policy_accuracy(bundles, label):
    items = [(b["level"], s, opt) for b in bundles for s, opt in b["policy"]]
    texts = generate([policy_prompt(l, s) for l, s, _ in items], 4)
    return dict(split=label, n=len(items),
                optimal_action_rate=np.mean([parse_action(t) in opt for (_, _, opt), t in zip(items, texts)]))


policy_acc = pd.DataFrame([eval_policy_accuracy(b, name) for name, b in SPLITS.items()])
print(policy_acc.to_string(float_format="{:.1%}".format))

# %% [markdown]
# ## 11. Planning: where does competence come from?
#
# One beam search (width `beam_width`, no revisits). Only its two plug-ins change:
#
# - **dynamics**: exact simulator vs. the learned model (predicted boards that fail to parse
#   are pruned; nothing is repaired),
# - **evaluator**: a hand heuristic (box-goal distance + corner-deadlock pruning) vs. the
#   learned `MOVES_TO_SOLVE` value (`DEAD` prunes the branch).
#
# A plan found in imagination is then executed in the real environment. *Hallucinated* plans
# are those the model believed solved the level but that don't.

# %%
def heuristic_value(level, state):
    if corner_deadlock(level, state):
        return INF
    boxes = state[1]
    unplaced = [b for b in boxes if b not in level.goals]
    h = sum(min(abs(b[0] - g[0]) + abs(b[1] - g[1]) for g in level.goals) for b in boxes)
    if unplaced:
        p = state[0]
        h += 0.1 * min(abs(p[0] - b[0]) + abs(p[1] - b[1]) for b in unplaced)
    return h


dyn_cache = {}


def learned_dynamics(queries):
    keys = [(render(l, s), a) for l, s, a in queries]
    missing = list({k: q for k, q in zip(keys, queries) if k not in dyn_cache}.items())
    texts = generate([dyn_prompt(*q) for _, q in missing], BOARD_TOKENS + 4)
    for (k, q), t in zip(missing, texts):
        dyn_cache[k] = parse_board(t, q[0])[0]
    return [dyn_cache[k] for k in keys]


def exact_dynamics(queries):
    return [step(l, s, a)[0] for l, s, a in queries]


value_cache = {}


def learned_value(items):
    keys = [render(l, s) for l, s in items]
    missing = list({k: it for k, it in zip(keys, items) if k not in value_cache}.items())
    texts = generate([value_prompt(*it) for _, it in missing], 6)
    for (k, _), t in zip(missing, texts):
        pred = parse_value(t)
        # Unparseable output is "unknown", not "dead": rank it last but don't prune it.
        value_cache[k] = 10**6 if pred is None else pred
    return [value_cache[k] for k in keys]


def hand_value(items):
    return [heuristic_value(l, s) for l, s in items]


def plan_batch(bundles, dynamics_fn, value_fn, max_depth):
    problems = [dict(level=b["level"], beams=[(b["start"], [])], seen={b["start"]}, plan=None, done=False)
                for b in bundles]
    invalid = 0
    for _ in range(max_depth):
        active = [p for p in problems if not p["done"]]
        if not active:
            break
        queries, owners = [], []
        for p in active:
            for s, path in p["beams"]:
                for a in ACTIONS:
                    queries.append((p["level"], s, a))
                    owners.append((p, s, path, a))
        candidates = defaultdict(list)
        for (p, s, path, a), nxt in zip(owners, dynamics_fn(queries)):
            if nxt is None:
                invalid += 1
                continue
            if p["done"] or nxt in p["seen"]:
                continue
            p["seen"].add(nxt)
            if is_solved(p["level"], nxt):
                p["plan"], p["done"] = path + [a], True
            else:
                candidates[id(p)].append((nxt, path + [a]))
        live = [p for p in active if not p["done"]]
        flat = [(p, c) for p in live for c in candidates[id(p)]]
        scores = value_fn([(p["level"], c[0]) for p, c in flat]) if flat else []
        ranked = defaultdict(list)
        for (p, c), v in zip(flat, scores):
            if v < INF:
                ranked[id(p)].append((v, len(ranked[id(p)]), c))
        for p in live:
            best = sorted(ranked[id(p)])[: CFG["beam_width"]]
            p["beams"] = [c for _, _, c in best]
            if not p["beams"]:
                p["done"] = True
    return problems, invalid


def execute(level, start, plan):
    state = start
    for a in plan:
        state = step(level, state, a)[0]
    return is_solved(level, state)


def run_policy(bundles):
    games = [dict(level=b["level"], state=b["start"], steps=0, done=False, solved=False) for b in bundles]
    limit = max(b["opt_len"] for b in bundles) * 2 + 10
    for _ in range(limit):
        live = [g for g in games if not g["done"]]
        if not live:
            break
        texts = generate([policy_prompt(g["level"], g["state"]) for g in live], 4)
        for g, t in zip(live, texts):
            a = parse_action(t)
            if a is None:
                g["done"] = True
                continue
            g["state"] = step(g["level"], g["state"], a)[0]
            g["steps"] += 1
            if is_solved(g["level"], g["state"]):
                g["done"] = g["solved"] = True
    return games


PLANNERS = {
    "exact dynamics + heuristic": (exact_dynamics, hand_value),
    "exact dynamics + learned value": (exact_dynamics, learned_value),
    "learned dynamics + heuristic": (learned_dynamics, hand_value),
    "learned dynamics + learned value": (learned_dynamics, learned_value),
}
plan_records = []
for split, bundles in SPLITS.items():
    bundles = bundles[: CFG["plan_levels"]]
    depth = max(b["opt_len"] for b in bundles) + 8
    for name, (dyn_fn, val_fn) in PLANNERS.items():
        t0 = time.time()
        problems, invalid = plan_batch(bundles, dyn_fn, val_fn, depth)
        for b, p in zip(bundles, problems):
            real = p["plan"] is not None and execute(b["level"], b["start"], p["plan"])
            plan_records.append(dict(split=split, method=name, solved=real,
                                     hallucinated=p["plan"] is not None and not real,
                                     length_ratio=len(p["plan"]) / b["opt_len"] if real else np.nan))
        print(f"{split:14s} {name:34s} {time.time() - t0:6.0f}s  invalid predictions={invalid}")
    for b, g in zip(bundles, run_policy(bundles)):
        plan_records.append(dict(split=split, method="direct policy (no search)", solved=g["solved"],
                                 hallucinated=False,
                                 length_ratio=g["steps"] / b["opt_len"] if g["solved"] else np.nan))

plan_df = pd.DataFrame(plan_records)
plan_table = plan_df.groupby(["split", "method"], sort=False).agg(
    n=("solved", "size"), solved=("solved", "mean"), hallucinated=("hallucinated", "mean"),
    length_vs_optimal=("length_ratio", "mean"))
plan_table["solved_95ci"] = [
    "[{:.0%}, {:.0%}]".format(*wilson(int(r.solved * r.n), int(r.n))) for r in plan_table.itertuples()]
print(plan_table.to_string(float_format="{:.2f}".format))

# %% [markdown]
# ## 12. Figures and artifacts

# %%
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
ax = axes[0]
evt = dyn_table.reset_index()
events = ["move", "blocked_wall", "blocked_box", "push", "push_on_goal"]
x = np.arange(len(events))
for k, (split, g) in enumerate(evt.groupby("split")):
    g = g.set_index("event").reindex(events)
    ax.bar(x + (k - 0.5) * 0.38, g.model_exact * 100, 0.38, label=split)
ax.scatter(x, evt.groupby("event").copy_exact.mean().reindex(events) * 100, color="k", marker="_", s=300,
           label="copy baseline", zorder=3)
ax.set_xticks(x, events, rotation=20)
ax.set_ylabel("exact next board (%)")
ax.set_title("Consequences: one step")
ax.legend(fontsize=8)

ax = axes[1]
for split, g in rollout_df.groupby("split"):
    ax.plot(g.horizon, g.survival * 100, marker="o", label=split)
ax.set_xlabel("steps of free-running imagination")
ax.set_ylabel("trajectories still exact (%)")
ax.set_title("Consequences: compounding error")
ax.legend(fontsize=8)

ax = axes[2]
pt = plan_table.reset_index()
methods = list(dict.fromkeys(pt.method))
y = np.arange(len(methods))
for k, (split, g) in enumerate(pt.groupby("split")):
    g = g.set_index("method").reindex(methods)
    ax.barh(y + (k - 0.5) * 0.38, g.solved * 100, 0.38, label=split)
ax.set_yticks(y, methods, fontsize=8)
ax.invert_yaxis()
ax.set_xlabel("levels solved in the real environment (%)")
ax.set_title("Strategy: planning")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "sokoban_cwm_results.png"), dpi=150)
plt.show()

results = dict(
    config={k: v for k, v in CFG.items() if k != "workers"},
    train_rows=dict(Counter(r[0] for r in rows)),
    fewshot_pretrained_exact=float(fewshot_df.exact.mean()),
    history=history,
    dynamics=dyn_table.reset_index().to_dict("records"),
    rollouts=rollout_df.to_dict("records"),
    value=value_table.reset_index().to_dict("records"),
    policy=policy_acc.to_dict("records"),
    planning=plan_table.reset_index().to_dict("records"),
)
with open(os.path.join(OUT, "sokoban_cwm_results.json"), "w") as f:
    json.dump(results, f, indent=2, default=float)
plan_df.to_csv(os.path.join(OUT, "planning_per_level.csv"), index=False)
value_df.to_csv(os.path.join(OUT, "value_per_state.csv"), index=False)
print("saved to", OUT)

# %% [markdown]
# ## 13. Reading the results
#
# - **Few-shot vs. fine-tuned dynamics:** how much of Sokoban's physics pretraining supplies.
# - **Push rows vs. blocked rows:** blocked moves are solvable by copying; pushes are the real
#   test of learned consequences.
# - **Rollout curve:** one-step accuracy that collapses by step 8-16 is not good enough to plan with.
# - **`exact + heuristic` vs. `exact + learned value`:** the value of learned *strategy* with
#   perfect physics. Look at non-corner deadlock recall in section 9 for the mechanism.
# - **`exact + X` vs. `learned + X`:** the cost of imperfect *consequences*, plus the
#   hallucinated-plan rate.
# - **Direct policy vs. search:** whether explicit look-ahead adds anything over imitation.
# - **ID vs. OOD:** whether the rules transfer to a third box (compositional generalisation) or
#   were memorised as two-box patterns.
