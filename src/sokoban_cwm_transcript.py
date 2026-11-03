# %% [markdown]
# # Sokoban world model with a self-fed transcript
#
# The model sees the initial board once, then writes the state after each action. Every
# prediction becomes part of its own context: nothing is rebuilt, repaired or re-rendered by code.
# That lets us watch what it does with its own mistakes.
#
# ```
# TASK TRANSCRIPT
# SIZE 8x8
# r0: # # # # # # # #
# r1: # . . P B . g #
# ...
# WALLS [[0,0],[0,1],...,[2,5],...]                 <- every wall cell as a coordinate
# STATE {"player":[1,3],"boxes":{"A":[1,4]}}        <- given (observed)
# ACTION RIGHT
# STATE {"player":[1,4],"boxes":{"A":[1,5]}}        <- model's own output, fed back as-is
# ACTION RIGHT
# STATE ...
# ```
#
# Representation v2 (after the first run): wall bumps were the weakest rule (78% even with the true
# history), plausibly because checking a wall meant locating a row and counting columns in the grid.
# `WALLS` turns that into a coordinate-membership check, like the box check that already worked.
# Boxes carry fixed IDs, so a push edits one named box instead of re-sorting a list (the first run
# duplicated boxes when it copied the wrong list entry).
#
# **Training:** 6x6-8x8 boards, 1-3 boxes, transcripts of 1-16 actions, actions biased so ~30%
# of steps involve a box (~18% real pushes, vs ~6-9% in the earlier notebooks). Loss on every predicted `STATE` line (true states in training).
#
# **Testing:** in-range boards, then beyond the range (9x9, 10x10, 4-5 boxes), all for 32 actions
# (2x the trained length). Each step is compared with two references:
#
# - **truth**: the simulator's real next state,
# - **self-consistent**: the simulator applied to the model's *own previous* prediction.
#
# | previous step | this step matches | category |
# |---|---|---|
# | right | truth | correct |
# | right | neither | new error (physics failed here) |
# | wrong | self-consistent | follows its own wrong path |
# | wrong | truth | recovers |
# | wrong (but possible) | neither | confused |
# | impossible (e.g. player in a wall) | not truth | after impossible state (no rule applies) |
#
# A **teacher-forced** pass (true states in context) gives the pure physics accuracy at every
# step, separating "doesn't know the rule" from "derailed by its own earlier error".
#
# **Kaggle:** GPU T4 x2 (one used), Internet On. `SOKOBAN_SMOKE=1` runs a tiny CPU check.

# %%
import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import gc
import json
import math
import random
import subprocess
import sys
import time
from collections import Counter, defaultdict
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
OUT = "/kaggle/working" if os.path.isdir("/kaggle/working") else os.path.abspath("sokoban_transcript_out")
os.makedirs(OUT, exist_ok=True)

CFG = dict(
    model_id="Qwen/Qwen3-0.6B-Base",
    resume_from=None,          # e.g. "/kaggle/input/<dataset>/ckpt" to skip training and only evaluate
    seed=20261021,
    train_sizes=(6, 7, 8),
    train_boxes=(1, 2, 3),
    train_horizon=16,
    n_train=12_000,            # transcripts; ~100k supervised states
    n_dev=96,
    box_bias=0.9,              # prob. of steering toward / into a box -> ~30% box steps, ~18% pushes
    wall_p=0.12,
    lr=5e-5,
    batch_size=16,
    epochs=1,
    warmup=100,
    log_every=100,
    eval_every=500,
    test_horizon=32,
    test_episodes=40,          # 60 took 34 min of eval last run; effects were large enough
    gen_batch=64,              # max transcripts per generate() call
    gen_token_budget=24_000,   # max context tokens per call: caps the KV cache (~112 KB/token) at ~2.7 GB
)
# (name, sizes, boxes, in training range?)
RUNGS = [
    ("in-range 6-8, 1-3 box", (6, 7, 8), (1, 2, 3), True),
    ("8x8 4 box", (8,), (4,), False),
    ("8x8 5 box", (8,), (5,), False),
    ("9x9 2 box", (9,), (2,), False),
    ("10x10 3 box", (10,), (3,), False),
]
if SMOKE:
    CFG.update(n_train=64, n_dev=8, batch_size=8, warmup=5, log_every=4, eval_every=8,
               test_horizon=6, test_episodes=4, gen_batch=8)

random.seed(CFG["seed"])
torch.manual_seed(CFG["seed"])
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if DEVICE == "cuda":
    AMP_DTYPE = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
else:
    AMP_DTYPE = None
print("device:", DEVICE, torch.cuda.get_device_name(0) if DEVICE == "cuda" else "", "amp:", AMP_DTYPE)

# %% [markdown]
# ## 1. Sokoban

# %%
ACTIONS = ("UP", "DOWN", "LEFT", "RIGHT")
DELTA = {"UP": (-1, 0), "DOWN": (1, 0), "LEFT": (0, -1), "RIGHT": (0, 1)}
BOX_EVENTS = ("push", "push_on_goal", "blocked_box")


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
        i = boxes.index(ahead)  # boxes keep their IDs: only box i moves, order never changes
        moved = boxes[:i] + (beyond,) + boxes[i + 1:]
        return (ahead, moved), ("push_on_goal" if beyond in level.goals else "push")
    return (ahead, boxes), "move"


def valid_state(level, state, n_boxes):
    """Physically possible: player and boxes on distinct non-wall cells, right box count."""
    if state is None:
        return False
    player, boxes = state
    cells = [player, *boxes]
    inside = all(0 <= r < level.size and 0 <= c < level.size for r, c in cells)
    return inside and len(boxes) == n_boxes and len(set(cells)) == len(cells) and not any(
        p in level.walls for p in cells)


def render_grid(level, state):
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
        rows.append(f"r{r}: " + " ".join(cells))
    return "\n".join(rows) + "\n"


BOX_IDS = "ABCDEFGH"


def state_json(state):
    player, boxes = state
    return json.dumps({"player": list(player), "boxes": {BOX_IDS[i]: list(b) for i, b in enumerate(boxes)}},
                      separators=(",", ":"))


def walls_json(level):
    return json.dumps([list(p) for p in sorted(level.walls)], separators=(",", ":"))


def parse_state(text):
    """Boxes are read in ID order; a missing, extra or renamed ID makes the state unequal to truth."""
    try:
        d = json.loads(text.strip())
        player = tuple(d["player"])
        if not isinstance(d["boxes"], dict):
            return None
        boxes = tuple(tuple(d["boxes"][k]) for k in sorted(d["boxes"]))
        if list(sorted(d["boxes"])) != list(BOX_IDS[:len(boxes)]):
            return None
        if len(player) != 2 or any(len(b) != 2 for b in boxes):
            return None
        if not all(isinstance(v, int) for v in (*player, *[x for b in boxes for x in b])):
            return None
        return player, boxes
    except Exception:
        return None


def layout_key(level):
    n = level.size - 1
    maps = [
        lambda r, c: (r, c), lambda r, c: (c, n - r), lambda r, c: (n - r, n - c), lambda r, c: (n - c, r),
        lambda r, c: (r, n - c), lambda r, c: (n - r, c), lambda r, c: (c, r), lambda r, c: (n - c, n - r),
    ]
    return (level.size,) + min(
        (tuple(sorted(f(*p) for p in level.walls)), tuple(sorted(f(*p) for p in level.goals))) for f in maps)


def random_level(rng, size, n_boxes):
    border = {(r, c) for r in range(size) for c in range(size) if r in (0, size - 1) or c in (0, size - 1)}
    interior = [(r, c) for r in range(1, size - 1) for c in range(1, size - 1)]
    for _ in range(1000):
        walls = border | {p for p in interior if rng.random() < CFG["wall_p"]}
        floor = [p for p in interior if p not in walls]
        if len(floor) < 2 * n_boxes + 4:
            continue
        seen, stack, floor_set = {floor[0]}, [floor[0]], set(floor)
        while stack:
            r, c = stack.pop()
            for dr, dc in DELTA.values():
                q = (r + dr, c + dc)
                if q in floor_set and q not in seen:
                    seen.add(q)
                    stack.append(q)
        if len(seen) != len(floor):
            continue
        goals = rng.sample(floor, n_boxes)
        boxes = rng.sample(floor, n_boxes)
        player = rng.choice([p for p in floor if p not in boxes])
        return Level(size, frozenset(walls), frozenset(goals)), (player, tuple(sorted(boxes)))
    raise RuntimeError(f"could not build a {size}x{size} level with {n_boxes} boxes")


def biased_action(rng, level, state):
    """Steer toward boxes so ~30% of steps involve one (mostly real pushes); else a random action."""
    if rng.random() < CFG["box_bias"]:
        events = {a: step(level, state, a)[1] for a in ACTIONS}
        pushes = [a for a in ACTIONS if events[a] in ("push", "push_on_goal")]
        blocked = [a for a in ACTIONS if events[a] == "blocked_box"]
        # Mostly real pushes; occasionally a box-on-box block so that rule is also covered.
        if pushes and (not blocked or rng.random() < 0.85):
            return rng.choice(pushes)
        if blocked and rng.random() < 0.3:
            return rng.choice(blocked)
        # Walk toward a cell from which some box can actually be pushed.
        (pr, pc), boxes = state
        blocked_cells = level.walls | set(boxes)
        spots = [(br - dr, bc - dc) for br, bc in boxes for dr, dc in DELTA.values()
                 if (br + dr, bc + dc) not in blocked_cells and (br - dr, bc - dc) not in blocked_cells]
        if spots:
            tr, tc = min(spots, key=lambda q: abs(q[0] - pr) + abs(q[1] - pc))
            toward = [a for a in ACTIONS if events[a] == "move"
                      and abs(tr - (pr + DELTA[a][0])) + abs(tc - (pc + DELTA[a][1])) < abs(tr - pr) + abs(tc - pc)]
            if toward:
                return rng.choice(toward)
    return rng.choice(ACTIONS)


def make_episode(rng, sizes, box_counts, horizon, exclude):
    while True:
        level, start = random_level(rng, rng.choice(sizes), rng.choice(box_counts))
        if layout_key(level) not in exclude:
            break
    states, actions, events = [start], [], []
    for _ in range(horizon):
        a = biased_action(rng, level, states[-1])
        nxt, ev = step(level, states[-1], a)
        actions.append(a)
        events.append(ev)
        states.append(nxt)
    return dict(level=level, states=states, actions=actions, events=events)


# %% [markdown]
# ## 2. Data
#
# Training transcripts have random lengths 1-16. Test episodes run 32 actions on layouts never
# seen in training (up to rotation/reflection).

# %%
rng = random.Random(CFG["seed"])
train_eps = [make_episode(rng, CFG["train_sizes"], CFG["train_boxes"], rng.randint(1, CFG["train_horizon"]), set())
             for _ in range(CFG["n_train"])]
train_keys = {layout_key(e["level"]) for e in train_eps}
dev_eps = [make_episode(rng, CFG["train_sizes"], CFG["train_boxes"], CFG["train_horizon"], train_keys)
           for _ in range(CFG["n_dev"])]
TEST = {name: [make_episode(rng, sizes, boxes, CFG["test_horizon"], train_keys) for _ in range(CFG["test_episodes"])]
        for name, sizes, boxes, _ in RUNGS}

ev = Counter(e for ep in train_eps for e in ep["events"])
print(f"train transcripts: {len(train_eps)}, supervised states: {sum(ev.values())}")
print("train events:", {k: f"{v / sum(ev.values()):.1%}" for k, v in ev.most_common()})
print(f"box-involving steps: {sum(ev[k] for k in BOX_EVENTS) / sum(ev.values()):.1%}")
for name, eps in TEST.items():
    tev = Counter(e for ep in eps for e in ep["events"])
    print(f"{name:24s} box-involving {sum(tev[k] for k in BOX_EVENTS) / sum(tev.values()):.1%}")

# %% [markdown]
# ## 3. Tokenisation as pieces
#
# Transcripts are built from token-id pieces, and in rollouts the model's generated **token ids**
# are appended directly. Its own output is fed back exactly, never re-tokenised or cleaned up.

# %%
tokenizer = AutoTokenizer.from_pretrained(CFG["model_id"])
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
PAD = tokenizer.pad_token_id
NEWLINE_IDS = [i for i in range(len(tokenizer)) if "\n" in tokenizer.decode([i])]
_cache = {}


def tok(text):
    if text not in _cache:
        _cache[text] = tokenizer(text, add_special_tokens=False)["input_ids"]
    return _cache[text]


def header_ids(ep):
    level, s0 = ep["level"], ep["states"][0]
    return tok(f"TASK TRANSCRIPT\nSIZE {level.size}x{level.size}\n{render_grid(level, s0)}"
               f"WALLS {walls_json(level)}\nSTATE {state_json(s0)}\n")


def action_ids(a):
    return tok(f"ACTION {a}\nSTATE ")


def target_ids(state):
    return tokenizer(state_json(state) + "\n", add_special_tokens=False)["input_ids"]


def encode_episode(ep, n_steps=None):
    """Teacher-forced transcript: ids, labels (loss on states only), and target spans per step."""
    ids, labels, spans = list(header_ids(ep)), [-100] * len(header_ids(ep)), []
    for k, a in enumerate(ep["actions"][:n_steps]):
        ai, ti = action_ids(a), target_ids(ep["states"][k + 1])
        ids += ai
        labels += [-100] * len(ai)
        spans.append((len(ids), len(ids) + len(ti)))
        ids += ti
        labels += ti
    return ids, labels, spans


# %% [markdown]
# ## 4. Model and training

# %%
TV = tuple(int(x) for x in transformers.__version__.split(".")[:2])
FP32 = {"dtype": torch.float32} if TV >= (4, 56) else {"torch_dtype": torch.float32}
if SMOKE:
    cfg = AutoConfig.from_pretrained(CFG["model_id"])
    for k, v in dict(hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                     num_attention_heads=2, num_key_value_heads=1, head_dim=32).items():
        setattr(cfg, k, v)
    model = AutoModelForCausalLM.from_config(cfg)
else:
    model = AutoModelForCausalLM.from_pretrained(CFG["resume_from"] or CFG["model_id"], **FP32)
model.float().to(DEVICE)
assert next(model.parameters()).dtype == torch.float32
print(f"params: {sum(p.numel() for p in model.parameters()) / 1e6:.0f}M")


def autocast():
    return torch.autocast("cuda", dtype=AMP_DTYPE) if AMP_DTYPE else torch.autocast("cpu", enabled=False)


def pad_right(seqs, value):
    width = max(len(s) for s in seqs)
    return torch.tensor([s + [value] * (width - len(s)) for s in seqs])


# Most transcript tokens are targets, so projecting them all through the 152k-word output layer at
# once needs several GB (the first Kaggle run OOMed on a 3.16 GiB logits tensor). Work in chunks,
# and recompute each chunk's logits in backward instead of storing them.
VOCAB_CHUNK = 512


def target_hidden(ids, mask, keep):
    hidden = model.get_decoder()(input_ids=ids, attention_mask=mask, use_cache=False).last_hidden_state
    return hidden[:, :-1][keep]


def _chunk_ce(h, y):
    return F.cross_entropy(model.get_output_embeddings()(h).float(), y, reduction="sum")


def chunked_loss(h, y):
    from torch.utils.checkpoint import checkpoint
    total = sum(checkpoint(_chunk_ce, h[i:i + VOCAB_CHUNK], y[i:i + VOCAB_CHUNK], use_reentrant=False)
                for i in range(0, len(y), VOCAB_CHUNK))
    return total / len(y)


def chunked_argmax(h):
    return torch.cat([model.get_output_embeddings()(h[i:i + VOCAB_CHUNK]).argmax(-1)
                      for i in range(0, len(h), VOCAB_CHUNK)])


@torch.no_grad()
def teacher_forced(eps):
    """Per step: does greedy decoding reproduce the true state, given the true transcript so far?
    With true states in context, greedy generation == argmax at every target position."""
    model.eval()
    out = []
    for start in range(0, len(eps), 16):
        chunk = eps[start:start + 16]
        enc = [encode_episode(ep) for ep in chunk]
        ids = pad_right([e[0] for e in enc], PAD).to(DEVICE)
        labels = pad_right([e[1] for e in enc], -100).to(DEVICE)
        mask = pad_right([[1] * len(e[0]) for e in enc], 0).to(DEVICE)
        shift = labels[:, 1:]
        keep = shift != -100
        with autocast():
            pred = chunked_argmax(target_hidden(ids, mask, keep))
        hit = torch.zeros_like(shift, dtype=torch.bool)
        hit[keep] = pred == shift[keep]
        hit = hit.cpu()
        for row, (ep, (_, _, spans)) in enumerate(zip(chunk, enc)):
            out.append([bool(hit[row, a - 1:b - 1].all()) for a, b in spans])
    return out


history = []
batch_rng = random.Random(CFG["seed"] + 1)
encoded = [encode_episode(ep)[:2] for ep in train_eps]
order = [i for _ in range(CFG["epochs"]) for i in batch_rng.sample(range(len(encoded)), len(encoded))]
batches = []
for s in range(0, len(order), CFG["batch_size"] * 50):  # length-grouped, then shuffled
    chunk = sorted(order[s:s + CFG["batch_size"] * 50], key=lambda i: len(encoded[i][0]))
    batches += [chunk[i:i + CFG["batch_size"]] for i in range(0, len(chunk), CFG["batch_size"])]
batch_rng.shuffle(batches)
total = len(batches)
if CFG["resume_from"]:
    print("resuming from", CFG["resume_from"], "- skipping training")
    batches = []
print(f"{total} steps, {sum(len(e[0]) for e in encoded) / 1e6:.1f}M tokens, "
      f"max length {max(len(e[0]) for e in encoded)}")

model.config.use_cache = False
model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
opt = torch.optim.AdamW(model.parameters(), lr=CFG["lr"], betas=(0.9, 0.95), weight_decay=0.0,
                        fused=DEVICE == "cuda")
sched = torch.optim.lr_scheduler.LambdaLR(
    opt, lambda s: min(1.0, (s + 1) / CFG["warmup"]) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / total))))
scaler = torch.amp.GradScaler("cuda", enabled=AMP_DTYPE == torch.float16)
t0, running = time.time(), []
for step_i, batch in enumerate(batches, 1):
    model.train()
    ids = pad_right([encoded[i][0] for i in batch], PAD).to(DEVICE)
    labels = pad_right([encoded[i][1] for i in batch], -100).to(DEVICE)
    mask = pad_right([[1] * len(encoded[i][0]) for i in batch], 0).to(DEVICE)
    keep = labels[:, 1:] != -100
    with autocast():
        loss = chunked_loss(target_hidden(ids, mask, keep), labels[:, 1:][keep])
    scaler.scale(loss).backward()
    scaler.unscale_(opt)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(opt)
    scaler.update()
    opt.zero_grad(set_to_none=True)
    sched.step()
    running.append(loss.item())
    if step_i % CFG["log_every"] == 0 or step_i == total:
        entry = dict(step=step_i, loss=float(np.mean(running)))
        running = []
        if step_i % CFG["eval_every"] == 0 or step_i == total:
            entry["dev_tf_step_acc"] = float(np.mean([x for r in teacher_forced(dev_eps) for x in r]))
        history.append(entry)
        eta = (time.time() - t0) / step_i * (total - step_i) / 60
        print(" ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in entry.items()),
              f"eta={eta:.0f}m")
model.gradient_checkpointing_disable()
model.config.use_cache = True
del opt, scaler, sched
for _name in ("ids", "labels", "mask", "keep", "loss"):
    globals().pop(_name, None)
gc.collect()
if DEVICE == "cuda":
    torch.cuda.empty_cache()
print(f"training took {(time.time() - t0) / 60:.1f} min")

# Save before any evaluation, so an evaluation failure never costs the training run.
if not SMOKE and not CFG["resume_from"]:
    model.save_pretrained(os.path.join(OUT, "ckpt"))
    tokenizer.save_pretrained(os.path.join(OUT, "ckpt"))
    print("checkpoint saved to", os.path.join(OUT, "ckpt"))

# Evaluate in half precision (halves weight memory, no autocast weight copies) if it stays finite.
if DEVICE == "cuda":
    model.to(AMP_DTYPE)
    probe_ids = torch.tensor([encode_episode(dev_eps[0])[0]], device=DEVICE)
    with torch.no_grad():
        if not torch.isfinite(model(input_ids=probe_ids, use_cache=False).logits).all():
            print("half-precision logits not finite; evaluating in fp32")
            model.float()
    print(f"eval dtype {next(model.parameters()).dtype}, "
          f"GPU memory in use {torch.cuda.memory_allocated() / 2**30:.1f} GiB")

# %% [markdown]
# ## 5. Free-running transcripts
#
# Each step appends the model's own generated tokens (cut at its first newline) to its context.
# If a line doesn't parse, the raw tokens are still fed back: the transcript stays honest.

# %%
@torch.no_grad()
def generate_lines(contexts):
    """Greedy-generate one line per context. Returns the generated token ids (through the newline)."""
    model.eval()
    out = [None] * len(contexts)
    order = sorted(range(len(contexts)), key=lambda i: len(contexts[i]))
    # Batches are capped by total tokens, not just count: transcripts grow every step and the
    # KV cache for 64 x ~1,100 tokens (~8 GB) caused the previous run's OOM.
    groups, cur = [], []
    for i in order:
        if cur and (len(cur) >= CFG["gen_batch"]
                    or (len(cur) + 1) * (len(contexts[i]) + 48) > CFG["gen_token_budget"]):
            groups.append(cur)
            cur = []
        cur.append(i)
    groups.append(cur)
    for idx in groups:
        width = max(len(contexts[i]) for i in idx)
        ids = torch.tensor([[PAD] * (width - len(contexts[i])) + contexts[i] for i in idx], device=DEVICE)
        mask = torch.tensor([[0] * (width - len(contexts[i])) + [1] * len(contexts[i]) for i in idx], device=DEVICE)
        with autocast():
            gen = model.generate(input_ids=ids, attention_mask=mask, max_new_tokens=48, do_sample=False,
                                 eos_token_id=NEWLINE_IDS, pad_token_id=PAD)
        for i, row in zip(idx, gen[:, width:].tolist()):
            line = []
            for t in row:
                line.append(t)
                if t in NEWLINE_SET:
                    break
            out[i] = line
    return out


NEWLINE_SET = set(NEWLINE_IDS)


def free_run(eps):
    contexts = [list(header_ids(ep)) for ep in eps]
    preds = [[] for _ in eps]
    raw = [[] for _ in eps]
    for k in range(len(eps[0]["actions"])):
        for c, ep in zip(contexts, eps):
            c += action_ids(ep["actions"][k])
        lines = generate_lines(contexts)
        for i, (c, line) in enumerate(zip(contexts, lines)):
            c += line
            text = tokenizer.decode(line)
            raw[i].append(text)
            preds[i].append(parse_state(text))
    return preds, raw


def classify(ep, preds):
    """Four-way category for every step, against truth and against the model's own previous belief."""
    rows, n_boxes = [], len(ep["states"][0][1])
    prev_pred, prev_right = ep["states"][0], True
    for k, (a, ev, pred) in enumerate(zip(ep["actions"], ep["events"], preds)):
        true_next = ep["states"][k + 1]
        self_next = step(ep["level"], prev_pred, a)[0] if valid_state(ep["level"], prev_pred, n_boxes) else None
        right = pred == true_next
        if prev_right:
            cat = "correct" if right else "new_error"
        elif right:
            cat = "recovers"
        elif self_next is None:
            cat = "after_impossible"  # previous belief broke the rules; there is no consistent next state
        elif pred == self_next:
            cat = "follows_own"
        else:
            cat = "confused"
        rows.append(dict(step=k + 1, event=ev, category=cat, right=right,
                         parsed=pred is not None, physically_valid=valid_state(ep["level"], pred, n_boxes)))
        prev_pred, prev_right = pred, right
    return rows


records, examples, tf_records = [], {}, []
t0 = time.time()
for name, sizes, boxes, in_range in RUNGS:
    eps = TEST[name]
    preds, raw = free_run(eps)
    for ep_i, (ep, p) in enumerate(zip(eps, preds)):
        for r in classify(ep, p):
            records.append(dict(rung=name, in_range=in_range, episode=ep_i, **r))
    for ep_i, (ep, hits) in enumerate(zip(eps, teacher_forced(eps))):
        for k, (hit, ev) in enumerate(zip(hits, ep["events"])):
            tf_records.append(dict(rung=name, episode=ep_i, step=k + 1, event=ev, tf_right=hit))
    # Keep the first episode that derails, for reading.
    for ep, p, r in zip(eps, preds, raw):
        cats = classify(ep, p)
        first_bad = next((c["step"] for c in cats if not c["right"]), None)
        if first_bad:
            lo = max(0, first_bad - 2)
            examples[name] = [dict(step=k + 1, action=ep["actions"][k], event=ep["events"][k],
                                   truth=state_json(ep["states"][k + 1]), model=r[k].strip(),
                                   category=cats[k]["category"]) for k in range(lo, min(lo + 8, len(r)))]
            break
    print(f"{name:24s} done ({time.time() - t0:.0f}s)")

df = pd.DataFrame(records)
tf = pd.DataFrame(tf_records)
df["band"] = np.where(df.step <= CFG["train_horizon"], f"steps 1-{CFG['train_horizon']}",
                      f"steps {CFG['train_horizon'] + 1}-{CFG['test_horizon']}")
tf["band"] = np.where(tf.step <= CFG["train_horizon"], f"steps 1-{CFG['train_horizon']}",
                      f"steps {CFG['train_horizon'] + 1}-{CFG['test_horizon']}")

# %% [markdown]
# ## 6. Results

# %%
rung_order = [r[0] for r in RUNGS]

# Survival: fraction of episodes whose transcript is still exactly right at step k.
surv = (df.assign(alive=df.groupby(["rung", "episode"]).right.cummin())
        .groupby(["rung", "step"]).alive.mean().unstack("rung")[rung_order])
print("Free-running survival (every state so far exact):")
print(surv.loc[[h for h in (1, 4, 8, 16, 24, 32) if h <= CFG["test_horizon"]]].to_string(float_format="{:.0%}".format))

print("\nPer-step accuracy: teacher-forced (true history) vs free-running (own history):")
acc = pd.concat([
    tf.groupby(["rung", "band"]).tf_right.mean().rename("teacher_forced"),
    df.groupby(["rung", "band"]).right.mean().rename("free_running"),
], axis=1).reindex(rung_order, level=0)
print(acc.to_string(float_format="{:.1%}".format))

print("\nTeacher-forced accuracy by event (pure physics):")
print(tf.groupby(["rung", "event"]).tf_right.agg(["mean", "size"]).unstack("event")
      .reindex(rung_order).to_string(float_format="{:.2f}".format))

print("\nFree-running categories (share of steps):")
cats = ["correct", "new_error", "follows_own", "recovers", "confused", "after_impossible"]
cat_table = (df.groupby(["rung", "band"]).category.value_counts(normalize=True).unstack()
             .reindex(columns=cats, fill_value=0).reindex(rung_order, level=0))
print(cat_table.to_string(float_format="{:.1%}".format))

print("\nWhen the model is off track, what does it do next? (share of off-track steps)")
off = df[df.category.isin(["follows_own", "recovers", "confused", "after_impossible"])]
print(off.groupby("rung").category.value_counts(normalize=True).unstack()
      .reindex(columns=["follows_own", "recovers", "confused", "after_impossible"], fill_value=0)
      .reindex(rung_order).to_string(float_format="{:.1%}".format))

print("\nWhere new errors start (rate per on-track step, by event):")
on_track = df[df.category.isin(["correct", "new_error"])]
print(on_track.assign(err=on_track.category == "new_error").groupby(["rung", "event"]).err.mean()
      .unstack("event").reindex(rung_order).to_string(float_format="{:.1%}".format))

print("\nOutput health: parseable / physically valid states:")
print(df.groupby("rung")[["parsed", "physically_valid"]].mean().reindex(rung_order)
      .to_string(float_format="{:.1%}".format))

for name, rows in examples.items():
    print(f"\n--- first derailment, {name} ---")
    for r in rows:
        mark = "  " if r["model"] == r["truth"] else "!!"
        print(f"{mark} step {r['step']:2d} {r['action']:5s} {r['event']:12s} [{r['category']}]")
        print(f"     truth: {r['truth']}\n     model: {r['model']}")

# %%
import matplotlib.pyplot as plt

colors = dict(zip(rung_order, ["#2F5D8C", "#D19032", "#B5562E", "#3E8E5E", "#8C3E7A"]))
fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))
ax = axes[0]
for name in rung_order:
    ax.plot(surv.index, surv[name] * 100, color=colors[name], label=name)
ax.axvline(CFG["train_horizon"], color="k", ls=":", lw=1)
ax.set_xlabel("step")
ax.set_ylabel("transcripts still exact (%)")
ax.set_title("Free-running survival (dotted: trained length)")
ax.legend(fontsize=8)

ax = axes[1]
tf_curve = tf.groupby(["rung", "step"]).tf_right.mean().unstack("rung")
fr_curve = df.groupby(["rung", "step"]).right.mean().unstack("rung")
for name in rung_order:
    ax.plot(tf_curve.index, tf_curve[name] * 100, color=colors[name], ls="--")
    ax.plot(fr_curve.index, fr_curve[name] * 100, color=colors[name], label=name)
ax.axvline(CFG["train_horizon"], color="k", ls=":", lw=1)
ax.set_xlabel("step")
ax.set_ylabel("step accuracy (%)")
ax.set_title("Dashed: true history. Solid: own history")

ax = axes[2]
share = df.groupby("rung").category.value_counts(normalize=True).unstack().reindex(
    index=rung_order, columns=cats, fill_value=0)
left = np.zeros(len(share))
cat_colors = dict(zip(cats, ["#9DB4C9", "#C0392B", "#D19032", "#3E8E5E", "#555555", "#A9A9A9"]))
for c in cats:
    ax.barh(range(len(share)), share[c] * 100, left=left, color=cat_colors[c], label=c)
    left += share[c].values * 100
ax.set_yticks(range(len(share)), rung_order, fontsize=8)
ax.invert_yaxis()
ax.set_xlabel("share of steps (%)")
ax.set_title("What each step was")
ax.legend(fontsize=8, loc="lower right")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "transcript_results.png"), dpi=150)
plt.show()

df.to_csv(os.path.join(OUT, "transcript_steps.csv"), index=False)
tf.to_csv(os.path.join(OUT, "transcript_teacher_forced.csv"), index=False)
with open(os.path.join(OUT, "transcript_results.json"), "w") as f:
    json.dump(dict(config=CFG, rungs=RUNGS, history=history, examples=examples), f, indent=1, default=str)
print("saved to", OUT)

# %% [markdown]
# ## 7. Reading the results
#
# - **Teacher-forced vs free-running:** the gap is the cost of reading its own history. A small gap
#   means errors are rare and isolated; a large one means early errors derail later steps.
# - **`follows_own` high:** the physics is right, it just trusts its own mistaken state. Errors are
#   *consistent*: a wrong belief, correctly updated.
# - **`recovers` high:** it re-derives the state from the initial board and actions, and overrides
#   its own written history.
# - **`confused` high:** once off track, it stops applying the rules coherently.
# - **`after_impossible`:** the previous belief broke the rules (player in a wall, two boxes on one
#   cell, missing box), so no rule defines a consistent next state. Separated from `confused` after
#   the first run, where these steps inflated it.
# - **New errors by event:** which rule breaks first: pushes, box-on-box blocks, or plain moves.
# - **Steps 17-32 vs 1-16:** generalisation to transcripts longer than any seen in training.
# - **Beyond-range rungs:** more boxes, and interior coordinates (7 on 9x9; 7-8 on 10x10) never
#   used in training outputs (8x8 interiors only span 1-6).
