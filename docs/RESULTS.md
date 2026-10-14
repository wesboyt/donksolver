# donksolver 0.1.0: measured results

Audit results were recorded on 2026-10-13 from the installed
`py3-none-win_amd64` wheel (MSVC 2022, x86-64 baseline, no AVX2), 2 threads.
Throughput and sizes were recorded on 2026-09-12 on an 8-thread, 16 GB laptop
CPU using 2 threads while another job held the remaining cores, so they are
lower bounds for an idle machine.

## Audits: 83/83 checks pass (`tests/run_all.py`)

| audit | result | negative control |
|---|---|---|
| evaluator | order == best-of-21 reference (200k pairs) and == pokerfast eval7 (50k pairs), 0 mismatches | straight/flush swap: 543 mismatches |
| betting + payoff vs pokerfast FastHand | 67,998 steps and 6,000 hands (5,823 with side pots): 0 mismatches | preflop rake: 22 hands; exact split: 132 hands |
| range deal law | chi2 = 150, df = 168, z = -1.01 | seat-only redraw: z = 293 |
| abstraction | suit invariance exact on all streets; bucket mass 1.80-2.25% (target 2%); Spearman(bucket, river HS) 0.990 / 0.991 | hole-only permutation changes 630 / 401 / 82 |
| nested abstraction | turn and flop e1, e2 == brute force over every runout (worst diff 4e-16); 50 boards re-binned here, 0 mismatches; board class x span exact on 2,000 flop/turn rows and 600 rivers; suit invariance exact on all streets | equal-mass bins: 44,058 holdings differ; sub-bins cut by equity: 19,337; class ignoring suits: 1,480 rows; hole-only permutation changes 520 / 349 / 67 |
| potential sub-bins | on two-tone flops, flush draws sit 0.48 sub-bins above no-draw holdings of the same equity bin (per bin +1.13, +0.80, -0.49) | |
| chance-sampled regret (exact SE, 400k deals) | Leduc 2p chi2/df 0.865, 0.515; Leduc 3p 1.126, 0.998, 1.054; max abs z 3.89 against a Bonferroni bound of 5.31 | no opponent reach: chi2/df > 4.6e5 |
| sampled average | worst row error 0.268 -> 0.118 -> 0.045 at T = 4k/16k/64k | opponent-weighted: 0.71 / 0.95 / 0.76 |
| convergence | Kuhn at 4e6 iterations ev0 = -0.05556 (-1/18), NashConv 0.0006; Leduc 2p LCFR slope -0.49/decade, NashConv at 1e6 = 0.022-0.027 | current strategy: slope -0.02 |
| `linear_avg` vs `linear` | NashConv ratio at 1e6 = 3.58x (median of 3 seeds: 3.58, 3.22, 7.42) | |
| Leduc 3p (coarse correlated sense) | max seat best-response gain 0.260 -> 0.048 | |
| translation | 1,332 on-tree lines: 0 mismatches; pseudo-harmonic 0.3878 vs 0.3824 (se 0.0063); 3,656 served rows, 0 illegal | linear mapping 0.7647 rejected |
| serving | first-to-act frequencies vs table chi2 z = 0.03; 1,500 mixed-seat hands, 0 illegal actions | permuted table z = 332 |
| multiway postflop (3 players, 15bb, nested buckets) | full tree: 0 check-down terminals, 135 / 297 / 540 three-way flop / turn / river decision nodes; facing a bet three-way on the flop the weakest equity bin folds 68%, the strongest 30%; 1,500 hands, 14,505 served rows (2,489 / 2,290 / 1,696 of them three-way on the flop / turn / river), 0 illegal, 0 no_node | tree cut at two players: 7 check-down terminals, no three-way nodes, 7,362 no_node rows |

The multi-threaded solves are not bit-reproducible, so the convergence and
serving figures move a little from run to run.

## Throughput and sizes

- Abstraction build (1,755 flops, 16,432 turns, 20k river boards): 28-37 s.
  The output is byte-identical for 2 and 3 threads.
- Kuhn 2.6M iterations/s, Leduc 2p ES 1.15M iterations/s, Leduc 3p ES 149k iterations/s.
- 6-max-lite 40bb ES solve: about 28k iterations/s (6 traversals each), plus sampled averaging at p = 0.02.
- Serving: 250k rows/s at batch 64 and batch 512 (4.0 us/row, packing
  included); 71 us for a single-row call.

| preset | stack | nodes | entries | tables (double) |
|---|---|---|---|---|
| 6max-lite | 40bb | 267,932 | 25.5M | 0.38 GB |
| 6max-lite | 100bb | 1,111,798 | 103.5M | 1.54 GB |
| 6max-lite | 200bb | 2,310,060 | 188.6M | 2.81 GB |
| 6max | 40bb | 487,040 | 37.9M | 0.56 GB |
| 6max | 100bb | 2,205,540 | 168.7M | 2.51 GB |
| 6max | 200bb | 5,245,672 | 338.3M | 5.04 GB |

A served policy file holds 4 bytes per entry, so a 6-max-lite 40/100/200
bundle takes about 1.3 GB of RAM.

## Serving rates against random-legal players (fixture, not a blueprint)

ok 64.1%, offtree 19.8% (translated, still served), no_node 16.2% (multiway
postflop), desync 0%. Rates against other players will differ;
`donksolver.engine.Player` counts them.

## Findings and corrections during the build

1. **MSVC `/arch:AVX2` miscompile.** In `Abstraction::compute_street`, the
   masked loop `if (cnt[h]) row[h] = acc[h]/cnt[h]` wrote 0.0 into 642 of 1,176
   flop values under MSVC 14.43. The result was correct without AVX2, and
   correct with `loop(no_vector)` on that loop alone; this was bisected in a
   standalone program. The first abstraction file was garbage (54% of mass in
   one bucket, suit invariance broken), and `audit_abstraction` is what caught
   it. AVX2 bought nothing (paired ABBA ratios 1.07/0.99/0.94/1.10), so the
   default build is baseline and AVX2 is opt-in.
2. **The first regret-unbiasedness test design was wrong, not the solver.** It
   flagged Leduc 3p at chi2/df 1.2-1.35 using SEs estimated from 40 batch
   means. Those are unreliable when entries depend on a few of the 1,680 deals.
   With the exact second moment, the same code gives 1.126 / 0.998 / 1.054.
   **The second design was still wrong.** It compared chi2/df with a
   chi-square quantile (z_chi < 4). Entries share deals, so the z values are
   dependent and chi2/df is far wider than chi-square: on Leduc 3p seat 0 it
   reads 1.126, 1.013, 1.059, 0.975, 1.415, 0.874 over six sampling seeds
   (z_chi from -4.3 to +11.9), with mean 1.0 and no growth at four times the
   samples (0.976, 0.866). The check passed on one platform at z_chi = 3.94
   and failed on another at 4.09 with identical code. The test now requires
   chi2/df < 2 (a bias adds N b^2, so this detects b > 0.0016 standard
   deviations per entry at N = 400k) and keeps the Bonferroni bound on the
   largest z, which is valid under any dependence.
3. **The first pseudo-harmonic test design was wrong.** It read 4.9 SE off
   because rows repeated the same holdings, and the mapping coin is (by design)
   a function of what the acting seat knows. With independent rows it reads
   0.86 SE.
4. **The abstraction build was nondeterministic.** River threshold boards were
   seeded per thread. They are now seeded per board index, so the file and its
   hash, which identifies every policy made with it, are reproducible.
5. The deal-law negative control could loop forever when a seat had no
   disjoint combo left. It is now bounded.
6. **Two audit thresholds sat inside the run-to-run spread.** Kuhn NashConv
   was checked against 0.005 at 2e5 iterations, where 16 runs (8 seeds, 1 and
   2 threads) gave up to 0.0054; it failed on two CI machines. At 4e6
   iterations the same 16 runs give at most 0.0012, and the audit now runs
   there. The first multiway audit required a three-way river row from the
   solved policy's own play, which produced 0 to 3 of them in 1,500 hands; it
   now mixes in checks and calls so that thousands of three-way rows are
   served on every street.
7. **A statically linked libstdc++ crashed on Linux.** Built with
   `-static-libstdc++` by gcc 13.3 on Ubuntu 24.04, every call that formatted
   a number through `std::ostringstream` segfaulted inside libstdc++'s locale
   code once the library was loaded with dlopen. The same source linked
   dynamically passed, and an AddressSanitizer + UBSan build was clean, so the
   fault is in the link mode and not in the solver. Linux now links libstdc++
   dynamically.

## Not done

- No NLHE best-response or exploitability measure. Doneness is root-EV
  stability only.
- The cost of serving unequal stacks from symmetric stack classes is not
  measured.
- No 6-max multiway blueprint has been solved. The multiway audit is a
  three-player, 15bb game.
- Nested against percentile buckets at equal table size: not compared in play.
- The Linux, macOS and Windows ARM64 builds are exercised by CI only.
