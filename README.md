# donksolver

A tabular Monte Carlo CFR solver for no-limit hold'em. It solves an abstract
game of up to six players on CPU, freezes the average strategy into a versioned
bundle, and serves actions for real game states through a thread-safe batch
server.

```bash
pip install donksolver
```

The core is C++17 behind a plain C ABI, loaded from Python with ctypes. Because
it is not a CPython extension, one wheel per OS/CPU serves every Python version
(free-threaded builds included): Linux x86_64/aarch64 (glibc and musl), macOS
x86_64/arm64, Windows AMD64/ARM64. Elsewhere pip builds the sdist, which needs
CMake and a C++17 compiler.

**What it is not.** It is not a Nash certificate for 6-max: with more than two
players CFR converges to a coarse correlated equilibrium at best. Exact
exploitability is available for the validation games (Kuhn, Leduc) only.

## Quick start

```bash
donksolver abstraction --buckets 50,50,50 --threads 2 --out abs.dab      # about a minute
donksolver size  --preset 6max-lite --stack-bb 100
donksolver solve --preset 6max-lite --stack-bb 100 --abstraction abs.dab \
    --iterations 50000000 --threads 6 --out runs/6max-lite_100bb          # resumable (state.bin)
donksolver bundle --abstraction abs.dab --out runs/bundle.json runs/*/policy.dpol
donksolver toy --game leduc --players 2 --iterations 1000000              # exact exploitability
```

`python -m donksolver ...` is the same command line.

From Python:

```python
from donksolver import Abstraction, Solver, Solve, Bundle, preset, toy

s = Solver(toy("leduc", 2))
s.run(10**6, threads=2)
print(s.exploitability())          # {'ev': ..., 'gains': ..., 'nashconv': ...}

Abstraction.build((50, 50, 50), threads=2).save("abs.dab")
s = Solver.nlhe(preset("6max-lite", stack_bb=100), "abs.dab", Solve())
s.solve("runs/6max-lite_100bb", iterations=50_000_000, threads=6)

b = Bundle("runs/bundle.json")
out = b.act(stacks, blinds, hole, board, histories, uniforms)   # kind/chips/prob/status per row
```

## Serving real states

A bundle holds one card abstraction and one policy per stack class. For each
row `Bundle.act`:

1. picks the class nearest the actor's effective stack (log scale);
2. replays the real action history through that class's tree, mapping each real
   raise to a neighbouring abstract size with the pseudo-harmonic mapping
   (Ganzfried & Sandholm, 2013). The mapping's coin is a hash of what the
   acting seat knows, so a state is always read the same way;
3. buckets the actor's cards, samples the strategy row with the caller's
   uniform, and converts the abstract action back to chips.

Every row reports a status: `ok`, `offtree` (translated, still served),
`no_node` (the abstraction has no node here, such as multiway postflop),
`desync`, `bad_input`. Rows at `no_node` and above carry no usable action; the
caller decides what plays them.

`donksolver.engine.Player` does the packing for hands played on
[pokerfast](https://pypi.org/project/pokerfast/)'s `FastHand`
(`pip install donksolver[pokerfast]`):

```python
from pokerfast import FastHand
from donksolver.engine import Player

player = Player("runs/bundle.json")
hand = FastHand()
actions, chips, probs, status = player.act([hand])
```

It reads the public state and the acting seat's own hole cards, nothing else.

## Layout

```
native/include/donk/   header-only C++17 core
  cards.hpp            card ids, 7-card evaluator, RNG
  betting.hpp          no-limit betting rules, side pots, rake
  tree.hpp             abstract trees: no-limit menu builder, fixed-limit builder
  abstraction.hpp      169 preflop classes; flop/turn E[HS^2] percentile buckets; river HS
  games.hpp            Kuhn / Leduc (validation) / NLHE games; range deal sampler
  solver.hpp           MCCFR: es / cs traversal, linear | linear_avg | vanilla weighting
  policy.hpp           frozen policy files, stack-class bundles, translation, batch act
native/src/capi.cpp    the C ABI (one shared library, no Python headers)
native/CMakeLists.txt  builds donksolver_core.{dll,so,dylib}
src/donksolver/        Python package (ctypes): Solver, Abstraction, Bundle, CLI
  engine.py            adapter: pokerfast FastHand -> action server
tests/                 audits (each with a negative control); pytest or run_all.py
```

## Algorithms

| piece | options | notes |
|---|---|---|
| tree | per-street, per-raise-level menus of pot-after-call fractions; `kappa_bet` drops bets that commit most of the stack; all-in by rule (`always`, `threshold`, `never`) | preflop `0.6` at the open is a 2.5bb raise |
| deals | uniform for a blueprint; a rejection sampler for weighted ranges | whole-deal restart on a card collision, audited against the enumerated law |
| traversal | `algo=es` external sampling (default), `algo=cs` chance sampling with pruning at sigma <= 1e-7 | ES makes a 6-max iteration affordable |
| weighting | `linear` (regrets and average weighted by t; Linear CFR, Brown & Sandholm 2019), default; `linear_avg` (uniform regrets, t-weighted average); `vanilla` | `linear_avg` has no proven bound and measured about 3.6x worse on Leduc |
| average | `avg_mode=weighted` (own reach, opponents full width) or `sampled` (own action sampled, as in Pluribus), each run with probability `p_avg` per iteration | both are consistent, audited |
| doneness | root EV per chunk with a standard error, one seat per deal | iterations per node is logged, but it is a budget and not a regret bound |
| cards | 169 preflop classes; E[HS^2] percentiles on flop and turn; HS percentiles on the river (Johanson et al. 2013) | imperfect recall |
| threads | shared tables, striped spinlocks on writes; `hogwild=1` drops them | |

The betting rules are pokerkit's cash-game no-limit semantics, checked step by
step against pokerfast's engine: every raise re-queues the other live players,
short all-ins accumulate toward reopening the betting, and the minimum raise is
capped by the effective stack. Blinds default to 100/200. Side pots merge on
winner sets and odd chips go to the lowest seat. The default rake is 5% of the
pot with a cap by stake and players dealt, no flop no drop (`rake.py`); set
`Game.rake_pct = 0` for none.

## Building from source

```bash
git clone https://github.com/wesboyt/donksolver
cd donksolver
pip install ".[test]"
DONKSOLVER_TEST_INSTALLED=1 pytest
```

Or build the library by hand and point the package at it:

```bash
cmake -S native -B native/build
cmake --build native/build --config Release
export DONKSOLVER_LIB=$PWD/native/build/donksolver_core.so     # Release/donksolver_core.dll with MSVC
PYTHONPATH=src pytest
```

AVX2 is opt-in (`-DDONK_AVX2=ON`) and off in every published wheel. A paired,
interleaved A/B on a 6-max-lite solve measured a 1.025x mean ratio, which is
noise (pairs 1.07/0.99/0.94/1.10), and MSVC 14.43 `/arch:AVX2` miscompiled a
masked loop in the abstraction builder (see `docs/RESULTS.md`). An AVX2 build
must re-run `tests/audit_abstraction.py`.

## Audits

`pytest`, or `python tests/run_all.py`. Every check that could pass vacuously
has a negative control: a deliberately broken variant that must be detected.
The card abstraction and two small solved policies are built on first use and
cached under `artifacts/` (about two minutes).

| audit | what | control |
|---|---|---|
| audit_eval | evaluator order == textbook best-of-21 and == pokerfast's eval7 | straight/flush swap |
| audit_payoff | replay == pokerfast FastHand at every step; settlement + rake == FastHand + pokerfast.rake | rake preflop; exact splits |
| audit_deal_law | range sampler law, chi-square vs enumeration | seat-only redraw |
| audit_abstraction | river HS exact; suit isomorphism lossless; bucket mass; buckets predict strength | hole-only suit permutation |
| audit_cs_regret | chance-sampled regret is unbiased, Leduc 2p/3p, exact SE | no opponent reach |
| audit_average | sampled average -> exact linear average | opponent-weighted average |
| audit_convergence | Kuhn value -1/18; Leduc LCFR O(1/sqrt T); `linear_avg` not better; 3p gains fall | current strategy (no average) |
| audit_translation | on-tree round trip; pseudo-harmonic frequency; legality in FastHand | linear mapping |
| audit_serving | served frequencies == table; mixed-seat hands; throughput | permuted table |

## Measured

`docs/RESULTS.md` has every number and its conditions. Headlines, on 2 threads
of a laptop CPU:

- abstraction build in about 30 s, byte-reproducible;
- 6-max-lite 40bb ES solve at about 28k iterations/s;
- serving at 250k rows/s;
- Leduc 2p LCFR NashConv of about 0.02 at 1e6 iterations.

## Limitations and known biases

- **Stack classes are symmetric.** A state with unequal stacks is served by the
  class nearest the actor's effective stack, and real sizes are translated by
  pot fraction. The cost of that translation is not measured.
- **Multiway postflop is not solved.** When more than `postflop_max_players`
  (default 2) see a flop, the tree ends in a check-down leaf. That biases
  preflop values toward position-less equity, and those states come back
  `no_node` in serving.
- **The abstraction has imperfect recall.** Flop and turn buckets are E[HS^2]
  percentiles, not potential-aware or texture buckets.
- **No NLHE best response.** For hold'em, doneness is root-EV stability only.
- `precision=float` halves table memory but loses increments once a row sees
  about 1e7 visits under linear weighting. The default is double.
- A policy is tied to its abstraction's hash. A rebuilt abstraction with
  different parameters invalidates the bundles made with the old one.

## License

MIT. See `LICENSE`.
