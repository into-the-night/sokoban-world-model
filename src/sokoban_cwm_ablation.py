# %% [markdown]
# # Does learning policy or value help a world model learn physics?
#
# The first run reached ~100% next-board accuracy with 57.6k dynamics rows, so it could not show
# whether the extra tasks *help* dynamics. Here every arm gets the **same 10k two-box dynamics
# rows** (2,500 states x 4 actions), seen the same number of times, and is tested on a ladder of
# increasingly out-of-distribution boards where errors have room to appear.
#
# | Arm | Trained on |
# |---|---|
# | `D` | dynamics only (pure world model) |
# | `D+P` | dynamics + policy (`BEST_ACTION`) |
# | `D+V` | dynamics + value (`MOVES_TO_SOLVE`) |
# | `D+V+P` | all three |
#
# Policy and value rows are **added on top**, never swapped for dynamics rows. Each arm runs with
# two training seeds; the data is identical across arms and seeds, so seeds measure training noise.
#
# **Kaggle:** GPU T4 x2 (one used), Internet On. Estimated ~6 GPU-hours for 4 arms x 2 seeds
# (~30-40 min training + ~10 min evaluation per run, scaled from the first run's throughput).
# Set `train_seeds=[0]` to halve it. Results are saved after every run.
# `SOKOBAN_SMOKE=1` runs a tiny CPU end-to-end check with a random-init model.

# %%
import os

# Reduce fragmentation: the last run OOMed with 2 GB reserved but unusable.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import gc
import json
import math
import multiprocessing as mp
import os
import random
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
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", "transformers>=4.51"], check=True)
    raise SystemExit("Upgraded transformers; restart the kernel and run again.")

from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

SMOKE = os.environ.get("SOKOBAN_SMOKE") == "1"
OUT = "/kaggle/working" if os.path.isdir("/kaggle/working") else os.path.abspath("sokoban_ablation_out")
os.makedirs(OUT, exist_ok=True)

CFG = dict(
    model_id="Qwen/Qwen3-0.6B-Base",
    data_seed=20261020,
    train_seeds=[0, 1],
    arms=["D", "D+P", "D+V", "D+V+P"],
    wall_p=0.12,
    train_len=(8, 30),
    max_states=150_000,
    n_train_levels=625,                           # 625 levels x 4 states x 4 actions = 10k rows
    train_per_level=dict(dyn=4, value=6, policy=3),  # -> 3,750 value rows, 1,875 policy rows
    dead_frac=0.35,
    dyn_epochs=2,                                  # every arm sees each dynamics row twice
    lr=5e-5,
    batch_size=16,
    warmup=50,
    curve_every=250,                               # learning-curve probe interval (steps)
    curve_probe=48,                                # transitions per rung in each probe
    rung_levels=40,
    rung_states=3,                                 # x 4 actions per state
    push_rows=400,
    rollout_seqs=48,
    rollout_len=16,
    gen_batch=96,
    workers=max(1, (os.cpu_count() or 2)),
)
# (name, board size incl. walls, boxes). Rung 0 is in-distribution.
RUNGS = [("R0 8x8 2box (ID)", 8, 2), ("R1 8x8 3box", 8, 3), ("R2 8x8 4box", 8, 4),
         ("R3 9x9 2box", 9, 2), ("R4 10x10 3box", 10, 3)]
if SMOKE:
    CFG.update(train_seeds=[0], n_train_levels=16, dyn_epochs=1, batch_size=8, warmup=5,
               curve_every=10, curve_probe=4, rung_levels=3, rung_states=1, push_rows=8,
               rollout_seqs=4, rollout_len=3, gen_batch=16)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if DEVICE == "cuda":
    AMP_DTYPE = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
else:
    AMP_DTYPE = None
print("device:", DEVICE, torch.cuda.get_device_name(0) if DEVICE == "cuda" else "", "amp:", AMP_DTYPE)

# %% [markdown]
# ## 1. Sokoban (same rules and text format as the first notebook)

# %%
ACTIONS = ("UP", "DOWN", "LEFT", "RIGHT")
DELTA = {"UP": (-1, 0), "DOWN": (1, 0), "LEFT": (0, -1), "RIGHT": (0, 1)}
INF = 10**9
CHAR_SET = set("#.gPpBX")


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
    lines = [line for line in text.split("\n") if line.strip()][: level.size]
    if len(lines) != level.size:
        return None, "format"
    walls, goals, boxes, players = set(), set(), [], []
    for r, line in enumerate(lines):
        cells = line.split()
        if len(cells) != level.size or any(ch not in CHAR_SET for ch in cells):
            return None, "format"
        for c, ch in enumerate(cells):
            if ch == "#":
                walls.add((r, c))
            if ch in "gpX":
                goals.add((r, c))
            if ch in "BX":
                boxes.append((r, c))
            if ch in "Pp":
                players.append((r, c))
    if walls != level.walls or goals != level.goals:
        return None, "static_changed"
    if len(players) != 1 or len(boxes) != len(level.goals):
        return None, "object_count"
    return (players[0], tuple(sorted(boxes))), None


def layout_key(level):
    n = level.size - 1
    maps = [
        lambda r, c: (r, c), lambda r, c: (c, n - r), lambda r, c: (n - r, n - c), lambda r, c: (n - c, r),
        lambda r, c: (r, n - c), lambda r, c: (n - r, c), lambda r, c: (c, r), lambda r, c: (n - c, n - r),
    ]
    return (level.size,) + min(
        (tuple(sorted(f(*p) for p in level.walls)), tuple(sorted(f(*p) for p in level.goals))) for f in maps
    )


def random_level(rng, size, n_boxes, wall_p):
    border = {(r, c) for r in range(size) for c in range(size) if r in (0, size - 1) or c in (0, size - 1)}
    interior = [(r, c) for r in range(1, size - 1) for c in range(1, size - 1)]
    walls = border | {p for p in interior if rng.random() < wall_p}
    floor = [p for p in interior if p not in walls]
    if len(floor) < 3 * n_boxes + 6:
        return None, None
    seen, stack, floor_set = {floor[0]}, [floor[0]], set(floor)
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
    return Level(size, frozenset(walls), frozenset(goals)), (player, tuple(sorted(boxes)))


# %% [markdown]
# ## 2. Training data: 10k two-box dynamics rows (+ value and policy rows for the arms that use them)
#
# Training levels are solvable 8x8 two-box levels with exact solver labels, generated exactly as in
# the first notebook. Each level contributes 4 states; every state is branched across all 4 actions.

# %%
def sample_indices(rng, pool, k):
    return [rng.choice(pool) for _ in range(k)] if pool and k > 0 else []


def build_train_level(seed):
    rng = random.Random(seed)
    counts = CFG["train_per_level"]
    for _ in range(2000):
        level, start = random_level(rng, 8, 2, CFG["wall_p"])
        if level is None:
            continue
        graph = explore(level, start, CFG["max_states"])
        if graph is None:
            continue
        states, children, dist = graph
        if not (CFG["train_len"][0] <= dist[0] <= CFG["train_len"][1]):
            continue
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
        return dict(level=level, key=layout_key(level),
                    dyn=[states[i] for i in dyn],
                    value=[(states[i], None if dist[i] == INF else dist[i]) for i in value],
                    policy=[(states[i], optimal_actions(children, dist, i)) for i in policy])
    return None


t0 = time.time()
with mp.get_context("fork").Pool(CFG["workers"]) as pool:
    candidates = [b for b in pool.map(build_train_level,
                                      [CFG["data_seed"] * 1009 + i for i in range(int(CFG["n_train_levels"] * 1.1) + 4)],
                                      chunksize=4) if b]
train, train_keys = [], set()
for b in candidates:
    if b["key"] not in train_keys:
        train_keys.add(b["key"])
        train.append(b)
    if len(train) == CFG["n_train_levels"]:
        break
print(f"train levels: {len(train)} ({time.time() - t0:.0f}s)")


def dyn_prompt(level, state, action):
    return f"TASK DYNAMICS\nBOARD\n{render(level, state)}ACTION {action}\nNEXT\n"


def value_prompt(level, state):
    return f"TASK VALUE\nBOARD\n{render(level, state)}MOVES_TO_SOLVE "


def policy_prompt(level, state):
    return f"TASK POLICY\nBOARD\n{render(level, state)}BEST_ACTION "


data_rng = random.Random(CFG["data_seed"])
TASK_ROWS = {"dyn": [], "value": [], "policy": []}
for b in train:
    lvl = b["level"]
    for s in b["dyn"]:
        for a in ACTIONS:
            TASK_ROWS["dyn"].append(("dyn", dyn_prompt(lvl, s, a), render(lvl, step(lvl, s, a)[0])))
    for s, d in b["value"]:
        TASK_ROWS["value"].append(("value", value_prompt(lvl, s), "DEAD" if d is None else str(d)))
    for s, opt in b["policy"]:
        TASK_ROWS["policy"].append(("policy", policy_prompt(lvl, s), data_rng.choice(opt)))
print({k: len(v) for k, v in TASK_ROWS.items()})
train_events = Counter(step(b["level"], s, a)[1] for b in train for s in b["dyn"] for a in ACTIONS)
print("training dynamics events:", dict(train_events))

# %% [markdown]
# ## 3. The OOD ladder
#
# Dynamics labels come straight from the simulator, so ladder levels need no solver: random
# layouts, then states reached by random walks from the start (which also move boxes around).
# No ladder layout appears in training, even up to rotation or reflection. Rung 5 pools pushes and
# box-blocked moves from every rung: the events where physics is non-trivial.

# %%
def ladder_level(rng, size, n_boxes):
    while True:
        level, start = random_level(rng, size, n_boxes, CFG["wall_p"])
        if level is not None and layout_key(level) not in train_keys:
            return level, start


def walk(rng, level, state, n):
    for _ in range(n):
        state = step(level, state, rng.choice(ACTIONS))[0]
    return state


ladder_rng = random.Random(CFG["data_seed"] + 99)
LADDER = {}            # rung -> list of (level, state, action)
ROLL_STARTS = {}       # rung -> list of (level, state)
push_pool = []
for name, size, boxes in RUNGS:
    rows, starts = [], []
    for _ in range(CFG["rung_levels"]):
        level, start = ladder_level(ladder_rng, size, boxes)
        starts.append((level, start))
        for _ in range(CFG["rung_states"]):
            s = walk(ladder_rng, level, start, ladder_rng.randint(0, 40))
            rows += [(level, s, a) for a in ACTIONS]
        for _ in range(30):  # extra states to harvest pushes
            s = walk(ladder_rng, level, start, ladder_rng.randint(0, 40))
            push_pool += [(level, s, a) for a in ACTIONS if step(level, s, a)[1] in ("push", "push_on_goal", "blocked_box")]
    LADDER[name] = rows
    ROLL_STARTS[name] = starts
ladder_rng.shuffle(push_pool)
LADDER["R5 pushes (all rungs)"] = push_pool[: CFG["push_rows"]]
for name, rows in LADDER.items():
    print(f"{name:24s} {len(rows):4d} transitions  events={dict(Counter(step(l, s, a)[1] for l, s, a in rows))}")

# Fixed small probe per rung for learning curves.
probe_rng = random.Random(CFG["data_seed"] + 7)
PROBE = {name: probe_rng.sample(rows, min(len(rows), CFG["curve_probe"])) for name, rows in LADDER.items()}

# %% [markdown]
# ## 4. Model utilities
#
# Same setup as the first notebook: Qwen3-0.6B-Base, fp32 master weights loaded explicitly,
# fp16 autocast with loss scaling on the T4, and loss only on target tokens.

# %%
tokenizer = AutoTokenizer.from_pretrained(CFG["model_id"])
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
EOS = tokenizer.eos_token_id
TV = tuple(int(x) for x in transformers.__version__.split(".")[:2])
FP32 = {"dtype": torch.float32} if TV >= (4, 56) else {"torch_dtype": torch.float32}
TASK_ID = {"dyn": 0, "value": 1, "policy": 2}
model = None


def fresh_model(seed):
    torch.manual_seed(seed)
    if SMOKE:
        cfg = AutoConfig.from_pretrained(CFG["model_id"])
        for k, v in dict(hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                         num_attention_heads=2, num_key_value_heads=1, head_dim=32).items():
            setattr(cfg, k, v)
        m = AutoModelForCausalLM.from_config(cfg)
    else:
        m = AutoModelForCausalLM.from_pretrained(CFG["model_id"], **FP32)
    m.float().to(DEVICE)  # from_config also inherits the checkpoint's bf16 dtype
    assert next(m.parameters()).dtype == torch.float32
    return m


def autocast():
    return torch.autocast("cuda", dtype=AMP_DTYPE) if AMP_DTYPE else torch.autocast("cpu", enabled=False)


def board_tokens(size):
    return size * size + 4  # one token per cell (newline merges into the last), plus slack


@torch.no_grad()
def generate(prompts, max_new_tokens):
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
        for i, t in zip(idx, tokenizer.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)):
            out[i] = t
    return out


FAILS = defaultdict(Counter)        # board size -> parse outcome counts (current run)
RAW_SAMPLES = defaultdict(list)     # board size -> a few unparseable raw outputs


def predict_next(rows):
    """rows: (level, state, action). Returns parsed predicted states (None if unparseable)."""
    preds = [None] * len(rows)
    by_size = defaultdict(list)
    for i, (l, _, _) in enumerate(rows):
        by_size[l.size].append(i)
    for size, idx in by_size.items():
        texts = generate([dyn_prompt(*rows[i]) for i in idx], board_tokens(size))
        for i, t in zip(idx, texts):
            preds[i], reason = parse_board(t, rows[i][0])
            FAILS[size][reason or "ok"] += 1
            if reason and len(RAW_SAMPLES[size]) < 3:
                RAW_SAMPLES[size].append(dict(reason=reason, prompt_board=render(rows[i][0], rows[i][1]), output=t))
    return preds


def one_step_exact(rows):
    preds = predict_next(rows)
    return [p == step(l, s, a)[0] for (l, s, a), p in zip(rows, preds)]


def encode(row):
    task, prompt, target = row
    p = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    t = tokenizer(target, add_special_tokens=False)["input_ids"] + [EOS]
    return task, p + t, [-100] * len(p) + t


ENCODED = {task: [encode(r) for r in rows] for task, rows in TASK_ROWS.items()}


def make_batches(encoded, bs, rng):
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
    hidden = model.get_decoder()(input_ids=ids, attention_mask=mask).last_hidden_state
    shift_labels = labels[:, 1:]
    keep = shift_labels != -100
    logits = model.get_output_embeddings()(hidden[:, :-1][keep]).float()
    token_loss = F.cross_entropy(logits, shift_labels[keep], reduction="none")
    row = keep.nonzero()[:, 0]
    per_task = {name: token_loss[tasks[row] == tid].mean().item()
                for name, tid in TASK_ID.items() if (tasks[row] == tid).any()}
    return token_loss.mean(), per_task


# %% [markdown]
# ## 5. Train each arm and seed, then evaluate on the ladder
#
# Each arm's data is: dynamics rows x `dyn_epochs`, plus its auxiliary rows x `dyn_epochs`,
# shuffled together. Learning curves are indexed by **dynamics rows seen**, so arms with extra
# tasks (and therefore more steps) are compared at equal physics exposure.

# %%
def arm_tasks(arm):
    return ["dyn"] + (["value"] if "V" in arm else []) + (["policy"] if "P" in arm else [])


def rollout_survival(starts, seed):
    rng = random.Random(seed)
    cases = []
    for _ in range(CFG["rollout_seqs"]):
        level, start = rng.choice(starts)
        cases.append(dict(level=level, true=start, pred=start, alive=True,
                          actions=[rng.choice(ACTIONS) for _ in range(CFG["rollout_len"])]))
    for h in range(CFG["rollout_len"]):
        live = [c for c in cases if c["alive"]]
        if not live:
            break
        preds = predict_next([(c["level"], c["pred"], c["actions"][h]) for c in live])
        for c, p in zip(live, preds):
            c["true"] = step(c["level"], c["true"], c["actions"][h])[0]
            c["pred"] = p
            c["alive"] = p == c["true"]
    return float(np.mean([c["alive"] for c in cases]))


def run_arm(arm, seed):
    global model
    model = fresh_model(seed)
    rng = random.Random(seed * 7919 + len(arm))
    encoded = [e for _ in range(CFG["dyn_epochs"]) for t in arm_tasks(arm) for e in ENCODED[t]]
    batches = make_batches(encoded, CFG["batch_size"], rng)
    total = len(batches)
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    opt = torch.optim.AdamW(model.parameters(), lr=CFG["lr"], betas=(0.9, 0.95), weight_decay=0.0,
                            fused=DEVICE == "cuda")
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / CFG["warmup"]) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / total))))
    scaler = torch.amp.GradScaler("cuda", enabled=AMP_DTYPE == torch.float16)
    curve, dyn_seen, losses, t0 = [], 0, defaultdict(list), time.time()
    FAILS.clear()
    RAW_SAMPLES.clear()

    def probe():
        model.config.use_cache = True
        entry = dict(arm=arm, seed=seed, dyn_rows_seen=dyn_seen)
        for name, rows in PROBE.items():
            entry[name] = float(np.mean(one_step_exact(rows)))
        model.config.use_cache = False
        curve.append(entry)

    for step_i, batch in enumerate(batches, 1):
        model.train()
        items = [encoded[i] for i in batch]
        dyn_seen += sum(t == "dyn" for t, _, _ in items)
        ids, labels, mask, tasks = collate(items)
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
            losses[k].append(v)
        if step_i % CFG["curve_every"] == 0 or step_i == total:
            probe()
            msg = " ".join(f"{k}={np.mean(v):.4f}" for k, v in losses.items())
            losses.clear()
            print(f"  [{arm} s{seed}] step {step_i}/{total} dyn_seen={dyn_seen} {msg} "
                  f"probe R0={curve[-1][RUNGS[0][0]]:.2f} R3={curve[-1][RUNGS[3][0]]:.2f} "
                  f"R4={curve[-1][RUNGS[4][0]]:.2f} {(time.time() - t0) / 60:.1f}m")
    model.gradient_checkpointing_disable()
    model.config.use_cache = True
    # The LR scheduler holds a reference to the optimizer; drop all three or ~5 GB of
    # AdamW state stays on the GPU during evaluation (this caused the D+P OOM).
    del opt, scaler, sched
    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    train_min = (time.time() - t0) / 60

    final = []
    for name, rows in LADDER.items():
        exact = one_step_exact(rows)
        events = [step(l, s, a)[1] for l, s, a in rows]
        for ev in sorted(set(events)):
            sel = [e for e, ev_ in zip(exact, events) if ev_ == ev]
            final.append(dict(arm=arm, seed=seed, rung=name, event=ev, n=len(sel), exact=float(np.mean(sel))))
        final.append(dict(arm=arm, seed=seed, rung=name, event="ALL", n=len(exact), exact=float(np.mean(exact))))
    rollouts = [dict(arm=arm, seed=seed, rung=name,
                     survival16=rollout_survival(ROLL_STARTS[name], CFG["data_seed"] + i))
                for i, (name, _, _) in enumerate(RUNGS)]
    print(f"[{arm} s{seed}] trained {train_min:.1f}m, steps={total}; "
          + ", ".join(f"{r['rung'].split()[0]}={r['exact']:.3f}" for r in final if r["event"] == "ALL"))
    model = None
    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    parse = {size: dict(c) for size, c in FAILS.items()}
    print(f"[{arm} s{seed}] parse outcomes by board size: {parse}")
    for size, samples in RAW_SAMPLES.items():
        print(f"  example unparseable output, {size}x{size} ({samples[0]['reason']}):\n{samples[0]['output']}")
    return curve, final, rollouts, dict(arm=arm, seed=seed, steps=total, train_min=train_min,
                                         parse=parse, raw_samples=dict(RAW_SAMPLES))


ALL_CURVES, ALL_FINAL, ALL_ROLLOUTS, ALL_RUNS = [], [], [], []
for seed in CFG["train_seeds"]:
    for arm in CFG["arms"]:
        curve, final, rollouts, run = run_arm(arm, seed)
        ALL_CURVES += curve
        ALL_FINAL += final
        ALL_ROLLOUTS += rollouts
        ALL_RUNS.append(run)
        # Save after every arm so a timeout still leaves usable partial results.
        with open(os.path.join(OUT, "ablation_results.json"), "w") as f:
            json.dump(dict(config=CFG, rungs=RUNGS, runs=ALL_RUNS, curves=ALL_CURVES,
                           final=ALL_FINAL, rollouts=ALL_ROLLOUTS), f, indent=1, default=str)

# %% [markdown]
# ## 6. Results
#
# Means over seeds, with each seed's value shown so seed-to-seed spread is visible. A gap between
# arms only means something if it is larger than the spread between seeds of the same arm.

# %%
final_df = pd.DataFrame(ALL_FINAL)
roll_df = pd.DataFrame(ALL_ROLLOUTS)
curve_df = pd.DataFrame(ALL_CURVES)
final_df.to_csv(os.path.join(OUT, "ablation_final.csv"), index=False)
roll_df.to_csv(os.path.join(OUT, "ablation_rollouts.csv"), index=False)
curve_df.to_csv(os.path.join(OUT, "ablation_curves.csv"), index=False)


def mean_with_seeds(frame, value, index):
    agg = frame.groupby(index + ["arm"])[value].agg(lambda v: f"{v.mean():.1%} (" + " / ".join(f"{x:.1%}" for x in v) + ")")
    return agg.unstack("arm")[CFG["arms"]]


print("One-step exact next board, mean (per seed):")
print(mean_with_seeds(final_df[final_df.event == "ALL"], "exact", ["rung"]).to_string())
print("\nOne-step exact on the hard events only:")
hard = final_df[final_df.event.isin(["push", "push_on_goal", "blocked_box"])]
print(mean_with_seeds(hard, "exact", ["rung", "event"]).to_string())
print("\n16-step free-running rollouts still exact:")
print(mean_with_seeds(roll_df, "survival16", ["rung"]).to_string())
print("\nTraining runs:")
print(pd.DataFrame(ALL_RUNS).drop(columns=["parse", "raw_samples"]).to_string(index=False))

# %%
import matplotlib.pyplot as plt

rung_names = list(LADDER)
colors = {"D": "#2F5D8C", "D+P": "#D19032", "D+V": "#3E8E5E", "D+V+P": "#8C3E7A"}
fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))

ax = axes[0]
x = np.arange(len(rung_names))
width = 0.8 / len(CFG["arms"])
summary = final_df[final_df.event == "ALL"]
for k, arm in enumerate(CFG["arms"]):
    g = summary[summary.arm == arm]
    means = g.groupby("rung").exact.mean().reindex(rung_names)
    ax.bar(x + (k - 1.5) * width, means * 100, width, color=colors[arm], label=arm)
    for xi, name in enumerate(rung_names):
        ax.scatter([xi + (k - 1.5) * width] * len(g[g.rung == name]), g[g.rung == name].exact * 100,
                   color="k", s=8, zorder=3)
ax.set_xticks(x, [n.split(" (")[0] for n in rung_names], rotation=25, ha="right")
ax.set_ylabel("exact next board (%)")
ax.set_title("One step, by OOD rung (dots = seeds)")
ax.legend(fontsize=8)

ax = axes[1]
for arm in CFG["arms"]:
    g = roll_df[roll_df.arm == arm].groupby("rung").survival16.mean().reindex([r[0] for r in RUNGS])
    ax.plot(range(len(RUNGS)), g * 100, marker="o", color=colors[arm], label=arm)
ax.set_xticks(range(len(RUNGS)), [r[0].split(" (")[0] for r in RUNGS], rotation=25, ha="right")
ax.set_ylabel("trajectories exact after 16 steps (%)")
ax.set_title("Compounding error, by OOD rung")
ax.legend(fontsize=8)

ax = axes[2]
focus = RUNGS[4][0]
for arm in CFG["arms"]:
    g = curve_df[curve_df.arm == arm].groupby("dyn_rows_seen")[focus].mean()
    ax.plot(g.index, g * 100, marker=".", color=colors[arm], label=arm)
ax.set_xlabel("dynamics rows seen")
ax.set_ylabel("probe exact (%)")
ax.set_title(f"Learning speed on {focus}")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "ablation_results.png"), dpi=150)
plt.show()

# %% [markdown]
# ## 7. Reading the results
#
# - **R0 (ID)** should be near the ceiling for every arm; if not, 10k rows is too few for any arm.
# - **`D+P` vs `D`** answers "does learning what to do help learn what happens?" Look first at
#   R3/R4 (new board sizes) and R5 (pushes), where the pure world model has room to fail.
# - **`D+V` vs `D`** is the same question for "is this position good?".
# - **Learning curves** can differ even when final accuracies tie: a helpful auxiliary task
#   reaches a given accuracy with fewer dynamics rows.
# - **Seeds:** treat any arm gap smaller than the seed spread within an arm as no effect.
