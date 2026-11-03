# %% [markdown]
# # Is the text representation biased? LLM transcript model vs grid CNN
#
# Same Sokoban transitions, same test episodes, same metrics; two very different world models:
#
# | | LLM (v2 transcript model) | Grid CNN |
# |---|---|---|
# | State in | text: labelled grid + `WALLS` coordinates + JSON state | 4 one-hot planes: wall, goal, player, box |
# | State out | JSON `{"player":[r,c],"boxes":{"A":[r,c],...}}`, token by token | per-cell "player here?" / "box here?" maps |
# | Position | absolute coordinates as number tokens | none: a 5x5 local view per cell (translation-invariant) |
# | Memory | whole transcript (attention) | Markov: current state only |
# | Size | 596M parameters (pretrained Qwen3-0.6B) | ~0.2M parameters (from scratch) |
#
# This is **not** a compute-matched contest. It asks which failures come from the representation.
#
# **Fairness choices**
#
# - Both are trained on the same ~102k transitions (true previous state -> next state).
# - Both receive walls/goals only from the initial observation, and both feed their **own raw
#   outputs** back during free-running rollouts (JSON tokens for the LLM; binarised maps for the
#   CNN). Nothing is repaired.
# - The CNN has no box IDs, so the main metric compares states as **sets** (player position + box
#   positions in any order) for both models. The LLM's stricter ID-aware score is reported too.
# - Both are scored by generation: "teacher-forced" = predict step k from the true history;
#   "free-running" = from its own history.
#
# **The key test: the shifted rung.** The in-range test episodes again, with **two extra rings of
# wall** around each board. The physics is identical; only the coordinates (+2) and board size
# change. A translation-equivariant CNN should barely notice. A drop for the LLM is direct evidence
# that its coordinate representation, not the physics, is what fails.
#
# **Setup on Kaggle:** GPU T4 x2, Internet On, and attach the output of the v2 transcript notebook
# (the run with `checkpoint saved to /kaggle/working/ckpt`) as an input. The notebook finds `ckpt/`
# automatically. Without it, only the CNN half runs. `SOKOBAN_SMOKE=1` runs a tiny CPU check.

# %%
import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import glob
import gc
import hashlib
import json
import math
import random
import time
from collections import Counter, defaultdict
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

SMOKE = os.environ.get("SOKOBAN_SMOKE") == "1"
OUT = "/kaggle/working" if os.path.isdir("/kaggle/working") else os.path.abspath("sokoban_compare_out")
os.makedirs(OUT, exist_ok=True)

# Data keys must stay identical to the v2 transcript notebook: the episodes are regenerated, not loaded.
CFG = dict(
    model_id="Qwen/Qwen3-0.6B-Base",
    llm_ckpt=None,             # None = search /kaggle/input/**/ckpt
    seed=20261021,
    train_sizes=(6, 7, 8),
    train_boxes=(1, 2, 3),
    train_horizon=16,
    n_train=12_000,
    n_dev=96,
    box_bias=0.9,
    wall_p=0.12,
    test_horizon=32,
    test_episodes=40,
    shift_rings=2,             # extra wall rings for the shifted rung
    # CNN
    cnn_channels=128,
    cnn_blocks=4,              # per-cell (1x1) residual blocks after the two 3x3 convolutions
    cnn_epochs=12,
    cnn_batch=512,
    cnn_lr=2e-3,
    # LLM generation
    gen_batch=64,
    gen_token_budget=24_000,
)
RUNGS = [
    ("in-range 6-8, 1-3 box", (6, 7, 8), (1, 2, 3), True),
    ("8x8 4 box", (8,), (4,), False),
    ("8x8 5 box", (8,), (5,), False),
    ("9x9 2 box", (9,), (2,), False),
    ("10x10 3 box", (10,), (3,), False),
]
SHIFTED = "in-range shifted +2"
V2_FINGERPRINT = "df3b78954a51d047"   # test episodes of the v2 run (40 per rung)
if SMOKE:
    CFG.update(n_train=64, n_dev=8, test_horizon=6, test_episodes=4, cnn_epochs=2, cnn_batch=64, gen_batch=8)

random.seed(CFG["seed"])
torch.manual_seed(CFG["seed"])
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", DEVICE, torch.cuda.get_device_name(0) if DEVICE == "cuda" else "")

# %% [markdown]
# ## 1. Sokoban (copied verbatim from the v2 transcript notebook)

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
# ## 2. Data: regenerated exactly as in v2, plus the shifted rung

# %%
rng = random.Random(CFG["seed"])
train_eps = [make_episode(rng, CFG["train_sizes"], CFG["train_boxes"], rng.randint(1, CFG["train_horizon"]), set())
             for _ in range(CFG["n_train"])]
train_keys = {layout_key(e["level"]) for e in train_eps}
dev_eps = [make_episode(rng, CFG["train_sizes"], CFG["train_boxes"], CFG["train_horizon"], train_keys)
           for _ in range(CFG["n_dev"])]
TEST = {name: [make_episode(rng, sizes, boxes, CFG["test_horizon"], train_keys) for _ in range(CFG["test_episodes"])]
        for name, sizes, boxes, _ in RUNGS}

fp = hashlib.sha256()
for name, eps in TEST.items():
    for ep in eps:
        fp.update(json.dumps([sorted(ep["level"].walls), ep["states"], ep["actions"]], default=list).encode())
fingerprint = fp.hexdigest()[:16]
print(f"test-set fingerprint {fingerprint} (v2 run: {V2_FINGERPRINT}) ->",
      "IDENTICAL to v2" if fingerprint == V2_FINGERPRINT else "DIFFERENT from v2 (expected only in smoke mode)")


def shift_episode(ep, k):
    """Same episode inside k extra rings of wall: physics unchanged, coordinates +k, board +2k."""
    lvl, size = ep["level"], ep["level"].size + 2 * k
    ring = {(r, c) for r in range(size) for c in range(size)
            if r < k or c < k or r >= size - k or c >= size - k}
    sh = lambda p: (p[0] + k, p[1] + k)
    level = Level(size, frozenset({sh(p) for p in lvl.walls} | ring), frozenset(sh(p) for p in lvl.goals))
    states = [(sh(pl), tuple(sh(b) for b in bx)) for pl, bx in ep["states"]]
    out = dict(ep, level=level, states=states)
    assert all(step(level, s, a) == (n, e) for s, a, n, e in zip(states, ep["actions"], states[1:], ep["events"]))
    return out


TEST[SHIFTED] = [shift_episode(ep, CFG["shift_rings"]) for ep in TEST[RUNGS[0][0]]]
RUNG_ORDER = [RUNGS[0][0], SHIFTED] + [r[0] for r in RUNGS[1:]]
N_BOXES = {name: [len(ep["states"][0][1]) for ep in eps] for name, eps in TEST.items()}
# Largest coordinate ever written for a player or box in training (8x8 interior = 1..6).
MAX_TRAIN_COORD = max(max(p[0], p[1], *[x for b in s[1] for x in b])
                      for ep in train_eps for s in ep["states"] for p in [s[0]])
print("max player/box coordinate in training states:", MAX_TRAIN_COORD)
print("transitions available for training:", sum(len(ep["actions"]) for ep in train_eps))

# %% [markdown]
# ## 3. Shared scoring
#
# States are compared as **sets**: (player, sorted boxes). The six categories are the v2 ones,
# applied identically to both models.

# %%
def norm(state):
    return None if state is None else (tuple(state[0]), tuple(sorted(tuple(b) for b in state[1])))


def new_coord(ep, k):
    """Does step k's intended destination use a coordinate larger than any in training states?"""
    (pr, pc), _ = ep["states"][k]
    dr, dc = DELTA[ep["actions"][k]]
    return max(pr + dr, pc + dc) > MAX_TRAIN_COORD


def score_free(model_name, rung, eps, preds, strict=None):
    rows = []
    for ep_i, (ep, p) in enumerate(zip(eps, preds)):
        n_boxes, level = len(ep["states"][0][1]), ep["level"]
        prev_pred, prev_right = norm(ep["states"][0]), True
        for k, (a, ev, pred) in enumerate(zip(ep["actions"], ep["events"], p)):
            pred, true_next = norm(pred), norm(ep["states"][k + 1])
            self_next = norm(step(level, prev_pred, a)[0]) if valid_state(level, prev_pred, n_boxes) else None
            right = pred == true_next
            if prev_right:
                cat = "correct" if right else "new_error"
            elif right:
                cat = "recovers"
            elif self_next is None:
                cat = "after_impossible"
            elif pred == self_next:
                cat = "follows_own"
            else:
                cat = "confused"
            rows.append(dict(model=model_name, rung=rung, episode=ep_i, step=k + 1, event=ev, category=cat,
                             right=right, valid=valid_state(level, pred, n_boxes),
                             box_count_ok=pred is not None and len(pred[1]) == n_boxes,
                             new_coord=new_coord(ep, k),
                             right_strict=None if strict is None else strict[ep_i][k] == ep["states"][k + 1]))
            prev_pred, prev_right = pred, right
    return rows


def score_tf(model_name, rung, eps, preds):
    rows = []
    for ep_i, (ep, p) in enumerate(zip(eps, preds)):
        for k, pred in enumerate(p):
            rows.append(dict(model=model_name, rung=rung, episode=ep_i, step=k + 1, event=ep["events"][k],
                             tf_right=norm(pred) == norm(ep["states"][k + 1]),
                             tf_right_strict=pred == ep["states"][k + 1], new_coord=new_coord(ep, k)))
    return rows


FREE, TF = [], []

# %% [markdown]
# ## 4. Grid CNN world model
#
# Input planes: wall, goal, player, box, plus the action as 4 constant planes. Every Sokoban rule
# depends only on cells within 2 steps (player, box, cell beyond), so the network sees exactly a
# **5x5 neighbourhood** per cell: two 3x3 convolutions, then per-cell (1x1) residual layers.
# Boards sit inside a 2-cell wall margin, so zero padding at the tensor edge is never within any
# cell's view. Together these make the CNN translation-invariant **by construction**: it cannot know
# absolute position. That locality is the built-in assumption being compared with the LLM's absolute
# coordinates. (A first design with a 27-cell receptive field could sense the tensor edge and
# dropped 21 points on the shifted rung in a local check.)
#
# Output: per-cell logits for "player here" and "box here". Decoding is a plain 0.5 threshold:
# nothing forces exactly one player or the right number of boxes.

# %%
MARGIN = 2  # wall cells around every board, so zero padding is outside every cell's 5x5 view


def planes(level, player_cells, box_cells, size):
    """(4, size+2*MARGIN, size+2*MARGIN) planes; board at offset MARGIN, everything else is wall."""
    x = np.zeros((4, size + 2 * MARGIN, size + 2 * MARGIN), dtype=np.float32)
    x[0] = 1.0
    o, n = MARGIN, level.size
    x[0, o:o + n, o:o + n] = 0.0
    for r, c in level.walls:
        x[0, o + r, o + c] = 1.0
    for r, c in level.goals:
        x[1, o + r, o + c] = 1.0
    for r, c in player_cells:
        x[2, o + r, o + c] = 1.0
    for r, c in box_cells:
        x[3, o + r, o + c] = 1.0
    return x


def board_mask(level, size):
    m = np.zeros((size + 2 * MARGIN, size + 2 * MARGIN), dtype=np.float32)
    m[MARGIN:MARGIN + level.size, MARGIN:MARGIN + level.size] = 1.0
    return m


class Block(nn.Module):
    """Per-cell residual block (1x1 convolutions): mixes features without widening the view."""
    def __init__(self, ch):
        super().__init__()
        self.a = nn.Conv2d(ch, ch, 1)
        self.b = nn.Conv2d(ch, ch, 1)

    def forward(self, x):
        return x + self.b(F.relu(self.a(F.relu(x))))


class GridWM(nn.Module):
    def __init__(self, ch, blocks):
        super().__init__()
        self.local = nn.Sequential(nn.Conv2d(8, ch // 2, 3, padding=1), nn.ReLU(),
                                   nn.Conv2d(ch // 2, ch, 3, padding=1))  # receptive field 5x5
        self.blocks = nn.Sequential(*[Block(ch) for _ in range(blocks)])
        self.head = nn.Conv2d(ch, 2, 1)

    def forward(self, x, action):
        a = F.one_hot(action, 4).float()[:, :, None, None].expand(-1, -1, x.shape[2], x.shape[3])
        return self.head(F.relu(self.blocks(self.local(torch.cat([x, a], 1)))))


S_TRAIN = max(CFG["train_sizes"])
xs, acts, ys, masks = [], [], [], []
for ep in train_eps:
    lvl = ep["level"]
    for k, a in enumerate(ep["actions"]):
        (p, bx), (p2, bx2) = ep["states"][k], ep["states"][k + 1]
        xs.append(planes(lvl, [p], bx, S_TRAIN))
        ys.append(planes(lvl, [p2], bx2, S_TRAIN)[2:4])
        masks.append(board_mask(lvl, S_TRAIN))
        acts.append(ACTIONS.index(a))
X, A_, Y, M = (torch.tensor(np.stack(xs)), torch.tensor(acts), torch.tensor(np.stack(ys)), torch.tensor(np.stack(masks)))
del xs, ys, masks
print("CNN training transitions:", len(X))

torch.manual_seed(CFG["seed"])
cnn = GridWM(CFG["cnn_channels"], CFG["cnn_blocks"]).to(DEVICE)
print(f"CNN params: {sum(p.numel() for p in cnn.parameters()) / 1e6:.2f}M")
opt = torch.optim.AdamW(cnn.parameters(), lr=CFG["cnn_lr"], weight_decay=1e-4)
steps_per_epoch = math.ceil(len(X) / CFG["cnn_batch"])
total = steps_per_epoch * CFG["cnn_epochs"]
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=CFG["cnn_lr"], total_steps=total)
t0 = time.time()
for epoch in range(CFG["cnn_epochs"]):
    perm = torch.randperm(len(X))
    losses = []
    cnn.train()
    for i in range(0, len(X), CFG["cnn_batch"]):
        idx = perm[i:i + CFG["cnn_batch"]]
        x, a, y, m = X[idx].to(DEVICE), A_[idx].to(DEVICE), Y[idx].to(DEVICE), M[idx].to(DEVICE)
        loss = (F.binary_cross_entropy_with_logits(cnn(x, a), y, reduction="none") * m[:, None]).sum() / m.sum()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        losses.append(loss.item())
    print(f"CNN epoch {epoch + 1}/{CFG['cnn_epochs']} loss={np.mean(losses):.5f} ({time.time() - t0:.0f}s)")
cnn.eval()
del X, A_, Y, M, opt, sched


@torch.no_grad()
def cnn_predict(levels, player_maps, box_maps, actions):
    """One step for a batch. Maps are lists of cell lists; returns predicted (player_cells, box_cells)."""
    size = max(l.size for l in levels)
    x = torch.tensor(np.stack([planes(l, p, b, size) for l, p, b in zip(levels, player_maps, box_maps)]),
                     device=DEVICE)
    a = torch.tensor([ACTIONS.index(act) for act in actions], device=DEVICE)
    out = (cnn(x, a) > 0).cpu().numpy()
    res = []
    m = MARGIN
    for l, o in zip(levels, out):
        players = [tuple(map(int, rc)) for rc in np.argwhere(o[0][m:m + l.size, m:m + l.size])]
        boxes = [tuple(map(int, rc)) for rc in np.argwhere(o[1][m:m + l.size, m:m + l.size])]
        res.append((players, boxes))
    return res


def as_state(players, boxes):
    return (players[0], tuple(sorted(boxes))) if len(players) == 1 else None


def cnn_free_run(eps):
    """Feeds back its own binarised maps, even if they contain 0 or 2 players."""
    cur = [([ep["states"][0][0]], list(ep["states"][0][1])) for ep in eps]
    preds = [[] for _ in eps]
    for k in range(len(eps[0]["actions"])):
        cur = cnn_predict([ep["level"] for ep in eps], [c[0] for c in cur], [c[1] for c in cur],
                          [ep["actions"][k] for ep in eps])
        for i, c in enumerate(cur):
            preds[i].append(as_state(*c))
    return preds


def cnn_teacher_forced(eps):
    preds = [[] for _ in eps]
    for k in range(len(eps[0]["actions"])):
        out = cnn_predict([ep["level"] for ep in eps], [[ep["states"][k][0]] for ep in eps],
                          [list(ep["states"][k][1]) for ep in eps], [ep["actions"][k] for ep in eps])
        for i, c in enumerate(out):
            preds[i].append(as_state(*c))
    return preds


for name in RUNG_ORDER:
    FREE += score_free("CNN", name, TEST[name], cnn_free_run(TEST[name]))
    TF += score_tf("CNN", name, TEST[name], cnn_teacher_forced(TEST[name]))
print(f"CNN evaluated ({time.time() - t0:.0f}s total)")
del cnn
gc.collect()
if DEVICE == "cuda":
    torch.cuda.empty_cache()

# %% [markdown]
# ## 5. LLM transcript model (the trained v2 checkpoint)
#
# Prompt format, tokenisation-as-pieces and generation are exactly the v2 code. Teacher-forced
# predictions are generated (not read off as argmax) so both models are scored the same way.

# %%
ckpt = CFG["llm_ckpt"] or next(iter(sorted(glob.glob("/kaggle/input/**/ckpt/config.json", recursive=True))), None)
ckpt = os.path.dirname(ckpt) if ckpt and ckpt.endswith("config.json") else ckpt
HAVE_LLM = bool(ckpt) or SMOKE
print("LLM checkpoint:", ckpt or ("none (smoke: tiny random model)" if SMOKE else
                                  "NOT FOUND - attach the v2 transcript notebook output; skipping the LLM half"))

if HAVE_LLM:
    tokenizer = AutoTokenizer.from_pretrained(ckpt or CFG["model_id"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    PAD = tokenizer.pad_token_id
    NEWLINE_IDS = [i for i in range(len(tokenizer)) if "\n" in tokenizer.decode([i])]
    NEWLINE_SET = set(NEWLINE_IDS)
    if SMOKE:
        cfg = AutoConfig.from_pretrained(CFG["model_id"])
        for k, v in dict(hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                         num_attention_heads=2, num_key_value_heads=1, head_dim=32).items():
            setattr(cfg, k, v)
        llm = AutoModelForCausalLM.from_config(cfg).float().to(DEVICE)
    else:
        llm = AutoModelForCausalLM.from_pretrained(ckpt).to(DEVICE)
        llm.half() if DEVICE == "cuda" else llm.float()
    llm.eval()
    print(f"LLM params: {sum(p.numel() for p in llm.parameters()) / 1e6:.0f}M, dtype {next(llm.parameters()).dtype}")

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


@torch.no_grad()
def generate_lines(contexts):
    out = [None] * len(contexts)
    order = sorted(range(len(contexts)), key=lambda i: len(contexts[i]))
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
        gen = llm.generate(input_ids=ids, attention_mask=mask, max_new_tokens=48, do_sample=False,
                           eos_token_id=NEWLINE_IDS, pad_token_id=PAD)
        for i, row in zip(idx, gen[:, width:].tolist()):
            line = []
            for t in row:
                line.append(t)
                if t in NEWLINE_SET:
                    break
            out[i] = line
    return out


def llm_free_run(eps):
    contexts = [list(header_ids(ep)) for ep in eps]
    preds = [[] for _ in eps]
    for k in range(len(eps[0]["actions"])):
        for c, ep in zip(contexts, eps):
            c += action_ids(ep["actions"][k])
        for i, (c, line) in enumerate(zip(contexts, generate_lines(contexts))):
            c += line
            preds[i].append(parse_state(tokenizer.decode(line)))
    return preds


def llm_teacher_forced(eps):
    contexts, where = [], []
    for i, ep in enumerate(eps):
        ctx = list(header_ids(ep))
        for k, a in enumerate(ep["actions"]):
            contexts.append(ctx + action_ids(a))
            where.append((i, k))
            ctx = ctx + action_ids(a) + target_ids(ep["states"][k + 1])
    preds = [[None] * len(ep["actions"]) for ep in eps]
    for (i, k), line in zip(where, generate_lines(contexts)):
        preds[i][k] = parse_state(tokenizer.decode(line))
    return preds


if HAVE_LLM:
    t0 = time.time()
    for name in RUNG_ORDER:
        p = llm_free_run(TEST[name])
        FREE += score_free("LLM", name, TEST[name], p, strict=p)
        TF += score_tf("LLM", name, TEST[name], llm_teacher_forced(TEST[name]))
        print(f"LLM {name:24s} done ({time.time() - t0:.0f}s)")

# %% [markdown]
# ## 6. Results

# %%
free = pd.DataFrame(FREE)
tf = pd.DataFrame(TF)
H = CFG["train_horizon"]
free["band"] = np.where(free.step <= H, f"1-{H}", f"{H + 1}-{CFG['test_horizon']}")
tf["band"] = np.where(tf.step <= H, f"1-{H}", f"{H + 1}-{CFG['test_horizon']}")
MODELS = [m for m in ("LLM", "CNN") if m in set(free.model)]
pct = "{:.1%}".format

surv = (free.assign(alive=free.groupby(["model", "rung", "episode"]).right.cummin())
        .groupby(["model", "rung", "step"]).alive.mean())
horizons = sorted({h for h in (8, 16, 32) if h <= CFG["test_horizon"]} | {CFG["test_horizon"]})
head = pd.concat({
    "TF acc": tf.groupby(["rung", "model"]).tf_right.mean(),
    "free acc": free.groupby(["rung", "model"]).right.mean(),
    **{f"exact@{h}": surv.xs(h, level="step").swaplevel() for h in horizons},
    "valid": free.groupby(["rung", "model"]).valid.mean(),
    "box count ok": free.groupby(["rung", "model"]).box_count_ok.mean(),
}, axis=1).reindex(RUNG_ORDER, level=0)
print("Head to head (states compared as sets; TF = true history, free = own history):")
print(head.to_string(float_format=pct))

print("\nThe shifted rung: same episodes, +2 wall rings. Change vs unshifted in-range:")
base, shifted = RUNG_ORDER[0], SHIFTED
delta = pd.DataFrame({m: {col: head.loc[(shifted, m), col] - head.loc[(base, m), col] for col in head.columns}
                      for m in MODELS}).T
print(delta.to_string(float_format=lambda v: f"{v:+.1%}"))

print("\nTeacher-forced accuracy by event:")
print(tf.groupby(["rung", "model", "event"]).tf_right.mean().unstack("event")
      .reindex(RUNG_ORDER, level=0).to_string(float_format=pct))

print(f"\nCoordinate test: TF accuracy on moves/pushes whose destination uses a coordinate > {MAX_TRAIN_COORD}")
moving = tf[tf.event.isin(["move", "push", "push_on_goal"])]
print(moving.groupby(["rung", "model", "new_coord"]).tf_right.agg(["mean", "size"]).unstack("new_coord")
      .reindex(RUNG_ORDER, level=0).to_string(float_format="{:.2f}".format))

print("\nOff-track behaviour (share of off-track steps):")
off = free[free.category.isin(["follows_own", "recovers", "confused", "after_impossible"])]
print(off.groupby(["rung", "model"]).category.value_counts(normalize=True).unstack()
      .reindex(RUNG_ORDER, level=0).to_string(float_format=pct))

if "LLM" in MODELS:
    print("\nLLM: set-equality vs strict (box IDs must match too):")
    llm_free = free[free.model == "LLM"]
    llm_tf = tf[tf.model == "LLM"]
    print(pd.DataFrame({
        "TF set": llm_tf.groupby("rung").tf_right.mean(), "TF strict": llm_tf.groupby("rung").tf_right_strict.mean(),
        "free set": llm_free.groupby("rung").right.mean(),
        "free strict": llm_free.groupby("rung").right_strict.mean(),
    }).reindex(RUNG_ORDER).to_string(float_format=pct))

# %%
import matplotlib.pyplot as plt

colors = {"LLM": "#8C3E7A", "CNN": "#2F5D8C"}
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
ax = axes[0]
x = np.arange(len(RUNG_ORDER))
for j, m in enumerate(MODELS):
    vals = [head.loc[(r, m), "TF acc"] * 100 for r in RUNG_ORDER]
    ax.bar(x + (j - 0.5) * 0.38, vals, 0.38, color=colors[m], label=m)
ax.set_xticks(x, RUNG_ORDER, rotation=25, ha="right", fontsize=8)
ax.set_ylabel("per-step accuracy, true history (%)")
ax.set_title("Physics given the truth")
ax.legend()

ax = axes[1]
for m in MODELS:
    for r, ls in [(RUNG_ORDER[0], "-"), (SHIFTED, "--")]:
        s = surv.loc[(m, r)]
        ax.plot(s.index, s.values * 100, color=colors[m], ls=ls, label=f"{m} {'shifted' if r == SHIFTED else 'in-range'}")
ax.set_xlabel("step")
ax.set_ylabel("transcripts still exact (%)")
ax.set_title("Same episodes, +2 wall rings (dashed)")
ax.legend(fontsize=8)

ax = axes[2]
for j, m in enumerate(MODELS):
    vals = [head.loc[(r, m), f"exact@{horizons[-1]}"] * 100 for r in RUNG_ORDER]
    ax.bar(x + (j - 0.5) * 0.38, vals, 0.38, color=colors[m], label=m)
ax.set_xticks(x, RUNG_ORDER, rotation=25, ha="right", fontsize=8)
ax.set_ylabel(f"exact after {horizons[-1]} own steps (%)")
ax.set_title("Free-running survival")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "compare_llm_cnn.png"), dpi=150)
plt.show()

free.to_csv(os.path.join(OUT, "compare_free_running.csv"), index=False)
tf.to_csv(os.path.join(OUT, "compare_teacher_forced.csv"), index=False)
head.to_csv(os.path.join(OUT, "compare_head_to_head.csv"))
with open(os.path.join(OUT, "compare_config.json"), "w") as f:
    json.dump(dict(config=CFG, fingerprint=fingerprint, llm_ckpt=ckpt, models=MODELS), f, indent=1, default=str)
print("saved to", OUT)

# %% [markdown]
# ## 7. Reading the results
#
# - **Shifted rung:** for the CNN it is a sanity check (it is translation-invariant by construction,
#   so its shifted and unshifted numbers should match exactly). For the LLM it is the cleanest test of
#   representational bias: if it drops, the failure is in how positions are written (absolute number
#   tokens), not in Sokoban physics, because nothing else changes.
# - **Coordinate test:** for the LLM, a gap between moves onto new coordinates and moves onto
#   familiar ones on the same boards isolates the "7 = wall" shortcut. The CNN has no coordinates, so
#   any gap it shows comes from something else (e.g. more open space on bigger boards).
# - **4-5 boxes:** the CNN has no IDs and can output any number of boxes. If it keeps the box count
#   right where the LLM drops boxes, the box-count failure belongs to the ID representation.
# - **TF vs free gap, off-track behaviour:** the CNN is Markov (no memory). If both models compound
#   errors and neither recovers, that's a property of step-by-step world models, not of text.
# - **Where the LLM wins:** the CNN has no history and far fewer parameters. If the LLM is better in
#   range, the text model is not worse at physics; its weaknesses are specifically about transfer.
