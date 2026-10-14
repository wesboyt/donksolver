// abstraction.hpp -- card abstraction for hold'em.
//
// HS is the win+tie/2 probability against ONE uniformly random opponent
// holding, with exact card removal. Preflop is always the 169 isomorphic
// classes (lossless). Two kinds of postflop buckets:
//
// percentile  one global ranking per street
//   flop    : percentile buckets of E[HS^2] over all turn+river runouts
//   turn    : percentile buckets of E[HS^2] over all rivers
//   river   : percentile buckets of HS
//   (Johanson et al. 2013). The board is not part of the bucket key.
//
// nested      buckets are relative to the board, and split by potential
//   equity   e1 : turn  E_river[HS]          flop  E_turn[E_river[HS]]
//   potential e2: turn  E_river[HS^2]        flop  E_turn[(E_river[HS])^2]
//   flop, turn : on each board, N equal-WIDTH bins of e1 between the lowest
//                and highest e1 any holding has on that board; then, inside
//                each bin, K equal-width bins of e2 between the lowest and
//                highest e2 in that bin. bucket = bin * K + sub-bin.
//   river      : N equal-width bins of HS on [0, 1].
//   Two holdings with the same equity but different draws land in different
//   sub-bins, and "the top bin" always means the best holdings on THIS board.
//   Optionally the bucket is also keyed by a board class (suit structure, or
//   suit structure x high-card counts): bucket = class * (N * K) + the above.
//
// Flop and turn values are precomputed once per SUIT-ISOMORPHIC board
// (1,755 flops, 16,432 turns) and stored as one byte per (board, holding).
// River HS costs ~1,081 evaluations, so it is computed when needed.
//
// Buckets are per street and looked up independently, so the abstract game has
// imperfect recall over cards and perfect recall over actions. CFR's
// convergence guarantee assumes perfect recall, so it does not carry over.
#pragma once
#include <algorithm>
#include <atomic>
#include <cstdio>
#include <functional>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "cards.hpp"

namespace donk {

static constexpr int MAXP_ABS = 8;

// MSVC 14.43 with /arch:AVX2 miscompiles the masked average loop in
// compute_street (`if (cnt[h]) row[h] = acc[h] / cnt[h]`): 642 of 1176 flop
// values came out 0.0, correct without AVX2 and correct with this pragma on
// that loop alone (bisected in a standalone harness). The pragma pins it
// scalar; tests/audit_abstraction.py fails if the table degenerates again.
#if defined(_MSC_VER)
#define DONK_NO_VECTOR __pragma(loop(no_vector))
#else
#define DONK_NO_VECTOR
#endif

enum { ABS_PERCENTILE = 0, ABS_NESTED = 1 };
enum { BOARDS_NONE = 0, BOARDS_SUITS = 1, BOARDS_TEXTURE = 2 };

// ---- board classes ------------------------------------------------------------
// Functions of the board as a SET of cards, invariant under suit permutations.
// suits: the flush structure. flop: rainbow / two-tone / monotone; turn:
// rainbow / one pair of a suit / two pairs of suits / three / four; river: no
// flush possible / three / four / five of a suit.
inline int suit_class(const int* board, int n) {
    int c[4] = {0, 0, 0, 0};
    for (int i = 0; i < n; ++i) c[card_suit(board[i])]++;
    std::sort(c, c + 4, [](int a, int b) { return a > b; });
    if (n == 3) return c[0] - 1;
    if (n == 4) return c[0] == 4 ? 4 : c[0] == 3 ? 3 : (c[1] == 2 ? 2 : (c[0] == 2 ? 1 : 0));
    return c[0] >= 5 ? 3 : c[0] == 4 ? 2 : c[0] == 3 ? 1 : 0;
}
inline int n_suit_classes(int n) { return n == 3 ? 3 : n == 4 ? 5 : 4; }
// high cards: a = cards seven or higher, b = cards jack or higher (b <= a).
inline int high_class(const int* board, int n) {
    int a = 0, b = 0;
    for (int i = 0; i < n; ++i) { a += card_rank(board[i]) >= 5; b += card_rank(board[i]) >= 9; }
    return a * (a + 1) / 2 + b;
}
inline int n_high_classes(int n) { return (n + 1) * (n + 2) / 2; }
inline int board_class(int mode, const int* board, int n) {
    if (mode == BOARDS_NONE) return 0;
    if (mode == BOARDS_SUITS) return suit_class(board, n);
    return suit_class(board, n) * n_high_classes(n) + high_class(board, n);
}
inline int n_board_classes(int mode, int n) {
    if (mode == BOARDS_NONE) return 1;
    if (mode == BOARDS_SUITS) return n_suit_classes(n);
    return n_suit_classes(n) * n_high_classes(n);
}

// Equal-width bin of v on [lo, hi]: the first k with v <= lo + (hi-lo)*k/B.
inline int width_bin(double v, int B, double lo, double hi) {
    double w = hi - lo + 1e-8;
    for (int k = 1; k <= B; ++k)
        if (v <= lo + w * ((double)k / B)) return k - 1;
    return B - 1;
}

// ---- suit canonicalisation --------------------------------------------------
struct Canon {
    uint64_t key = 0;
    int perm[4] = {0, 1, 2, 3};  // original suit -> canonical suit
    int map_card(int c) const { return make_card(card_rank(c), perm[card_suit(c)]); }
};

inline Canon canonical_board(const int* board, int n) {
    uint32_t m[4] = {0, 0, 0, 0};
    for (int i = 0; i < n; ++i) m[card_suit(board[i])] |= 1u << card_rank(board[i]);
    int order[4] = {0, 1, 2, 3};
    // sort suits by (popcount desc, mask desc, suit asc); ties are symmetric
    std::sort(order, order + 4, [&](int a, int b) {
        int pa = popc32(m[a]), pb = popc32(m[b]);
        if (pa != pb) return pa > pb;
        if (m[a] != m[b]) return m[a] > m[b];
        return a < b;
    });
    Canon c;
    for (int k = 0; k < 4; ++k) {
        c.perm[order[k]] = k;
        c.key = (c.key << 13) | m[order[k]];
    }
    return c;
}

// ---- hand strength on a complete board -------------------------------------
// hs_out[holding] for every holding disjoint from the board; others untouched.
struct HSWork {
    struct E { uint32_t v; uint16_t h; };
    E e[1326];
    int below_card[52];
    int tie_card[52];
};

inline void hs_all(const int* board5, double* hs_out, double* hs2_acc, int* cnt, HSWork& w) {
    uint64_t bm = 0, dead = 0;
    for (int i = 0; i < 5; ++i) { bm |= sbit(board5[i]); dead |= 1ULL << board5[i]; }
    const auto& H = holdings();
    int live[47], nl = 0;
    for (int c = 0; c < 52; ++c) if (!((dead >> c) & 1)) live[nl++] = c;
    int ne = 0;
    for (int x = 1; x < nl; ++x)
        for (int y = 0; y < x; ++y) {
            int a = live[y], b = live[x];
            w.e[ne].v = eval_mask(bm | sbit(a) | sbit(b));
            w.e[ne].h = (uint16_t)holding_index(a, b);
            ++ne;
        }
    std::sort(w.e, w.e + ne, [](const HSWork::E& p, const HSWork::E& q) { return p.v < q.v; });
    std::memset(w.below_card, 0, sizeof(w.below_card));
    std::memset(w.tie_card, 0, sizeof(w.tie_card));
    int total_below = 0;
    const double inv = 1.0 / 990.0;  // C(45,2) opponent holdings
    for (int g0 = 0; g0 < ne;) {
        int g1 = g0;
        while (g1 < ne && w.e[g1].v == w.e[g0].v) ++g1;
        int gs = g1 - g0;
        for (int k = g0; k < g1; ++k) { w.tie_card[H.lo[w.e[k].h]]++; w.tie_card[H.hi[w.e[k].h]]++; }
        for (int k = g0; k < g1; ++k) {
            int h = w.e[k].h, lo = H.lo[h], hi = H.hi[h];
            int below = total_below - w.below_card[lo] - w.below_card[hi];
            int ties = gs - (w.tie_card[lo] + w.tie_card[hi] - 1);
            double hs = (below + 0.5 * ties) * inv;
            if (hs_out) hs_out[h] = hs;
            if (hs2_acc) { hs2_acc[h] += hs * hs; cnt[h]++; }
        }
        for (int k = g0; k < g1; ++k) {
            w.tie_card[H.lo[w.e[k].h]] = 0; w.tie_card[H.hi[w.e[k].h]] = 0;
            w.below_card[H.lo[w.e[k].h]]++; w.below_card[H.hi[w.e[k].h]]++;
        }
        total_below += gs;
        g0 = g1;
    }
}

// HS for a few specific holdings on one board: one pass, no sort.
inline void hs_players(const int* board5, int np, const int (*hole)[2], double* out) {
    uint64_t bm = 0, dead = 0;
    for (int i = 0; i < 5; ++i) { bm |= sbit(board5[i]); dead |= 1ULL << board5[i]; }
    int live[47], nl = 0;
    for (int c = 0; c < 52; ++c) if (!((dead >> c) & 1)) live[nl++] = c;
    uint32_t pv[MAXP_ABS];
    uint64_t pm[MAXP_ABS];
    int below[MAXP_ABS] = {0}, ties[MAXP_ABS] = {0};
    for (int p = 0; p < np; ++p) {
        pv[p] = eval_mask(bm | sbit(hole[p][0]) | sbit(hole[p][1]));
        pm[p] = (1ULL << hole[p][0]) | (1ULL << hole[p][1]);
    }
    for (int x = 1; x < nl; ++x) {
        int b = live[x];
        uint64_t sb_ = bm | sbit(b);
        for (int y = 0; y < x; ++y) {
            int a = live[y];
            uint64_t hm = (1ULL << a) | (1ULL << b);
            uint32_t v = eval_mask(sb_ | sbit(a));
            for (int p = 0; p < np; ++p) {
                if (hm & pm[p]) continue;
                below[p] += v < pv[p];
                ties[p] += v == pv[p];
            }
        }
    }
    for (int p = 0; p < np; ++p) out[p] = (below[p] + 0.5 * ties[p]) / 990.0;
}

inline uint64_t fnv1a(const uint8_t* p, size_t n, uint64_t h = 1469598103934665603ULL) {
    for (size_t i = 0; i < n; ++i) { h ^= p[i]; h *= 1099511628211ULL; }
    return h;
}

class Abstraction {
public:
    int nb[4] = {169, 0, 0, 0};               // buckets per street, board classes included
    int kind = ABS_PERCENTILE;
    int sub = 1;                              // nested: potential sub-bins per equity bin (flop, turn)
    int nbin[4] = {169, 0, 0, 0};             // nested: equity bins per street
    int bmode[4] = {0, 0, 0, 0};              // nested: board class mode per street
    int span[4] = {169, 0, 0, 0};             // buckets per board class
    std::vector<uint64_t> flop_keys, turn_keys;
    std::vector<uint32_t> flop_mult, turn_mult;
    std::vector<uint8_t> flop_tab, turn_tab;  // n_boards x 1326, the bucket within the board class
    std::vector<uint16_t> flop_cls, turn_cls; // board class per board (derived, not stored)
    std::vector<float> river_thr;             // percentile: nb[3]-1 ascending
    uint64_t hash = 0;

    int flop_index(uint64_t key) const { return index_of(flop_keys, key); }
    int turn_index(uint64_t key) const { return index_of(turn_keys, key); }

    // hi: the holding's index after the board's suit canonicalisation
    int flop_bucket_at(int bi, int hi) const { return flop_cls[(size_t)bi] * span[1] + flop_tab[(size_t)bi * 1326 + hi]; }
    int turn_bucket_at(int bi, int hi) const { return turn_cls[(size_t)bi] * span[2] + turn_tab[(size_t)bi * 1326 + hi]; }

    // bucket for a holding given the board prefix of that street
    int bucket(int street, int h0, int h1, const int* board) const {
        if (street == 0) return preflop_class(h0, h1);
        if (street == 3) {
            int hole[1][2] = {{h0, h1}};
            double hs;
            hs_players(board, 1, hole, &hs);
            return river_bucket(board, hs);
        }
        Canon c = canonical_board(board, street == 1 ? 3 : 4);
        int bi = street == 1 ? flop_index(c.key) : turn_index(c.key);
        if (bi < 0) throw std::runtime_error("board not in abstraction");
        int hi = holding_index(c.map_card(h0), c.map_card(h1));
        return street == 1 ? flop_bucket_at(bi, hi) : turn_bucket_at(bi, hi);
    }

    int river_bucket(const int* board5, double hs) const {
        if (kind == ABS_PERCENTILE)
            return (int)(std::upper_bound(river_thr.begin(), river_thr.end(), (float)hs) - river_thr.begin());
        return board_class(bmode[3], board5, 5) * span[3] + width_bin(hs, nbin[3], 0.0, 1.0 - 1e-8);
    }

    // ---- build ----------------------------------------------------------------
    void build(int nbf, int nbt, int nbr, int threads, uint64_t river_boards, uint64_t seed,
               const std::function<void(const char*, double)>& progress) {
        if (nbf < 1 || nbf > 255 || nbt < 1 || nbt > 255 || nbr < 1 || nbr > 255)
            throw std::runtime_error("bucket counts must be 1..255");
        nb[0] = 169; nb[1] = nbf; nb[2] = nbt; nb[3] = nbr;
        threads = std::max(1, threads);
        enumerate_iso(3, flop_keys, flop_mult);
        enumerate_iso(4, turn_keys, turn_mult);
        std::vector<float> fv, tv;
        compute_street(3, flop_keys, fv, threads, progress);
        compute_street(4, turn_keys, tv, threads, progress);
        flop_tab = quantise(fv, flop_keys, flop_mult, nbf);
        turn_tab = quantise(tv, turn_keys, turn_mult, nbt);
        river_thresholds(nbr, river_boards, seed, threads, progress);
        kind = ABS_PERCENTILE;
        sub = 1;
        for (int s = 1; s < 4; ++s) { nbin[s] = nb[s]; bmode[s] = BOARDS_NONE; }
        derive();
        rehash();
    }

    // Nested buckets: nf/nt/nr equity bins, `sub_bins` potential sub-bins on the
    // flop and turn, board class mode per street. Deterministic: every board is
    // enumerated, nothing is sampled.
    void build_nested(int nf, int nt, int nr, int sub_bins, int bf, int bt, int br, int threads,
                      const std::function<void(const char*, double)>& progress) {
        if (nf < 1 || nt < 1 || nr < 1 || sub_bins < 1 || nf * sub_bins > 255 || nt * sub_bins > 255 || nr > 255)
            throw std::runtime_error("nested buckets: bins * sub-bins must be 1..255 per street");
        int modes[3] = {bf, bt, br};
        for (int m : modes)
            if (m < BOARDS_NONE || m > BOARDS_TEXTURE) throw std::runtime_error("unknown board class mode");
        kind = ABS_NESTED;
        sub = sub_bins;
        nbin[1] = nf; nbin[2] = nt; nbin[3] = nr;
        bmode[1] = bf; bmode[2] = bt; bmode[3] = br;
        threads = std::max(1, threads);
        enumerate_iso(3, flop_keys, flop_mult);
        enumerate_iso(4, turn_keys, turn_mult);
        nested_street(3, flop_keys, flop_tab, nf, threads, progress);
        nested_street(4, turn_keys, turn_tab, nt, threads, progress);
        river_thr.clear();
        derive();
        if (nb[1] > 32767 || nb[2] > 32767 || nb[3] > 32767)
            throw std::runtime_error("more than 32767 buckets on a street");
        rehash();
    }

    // Everything that follows from the stored fields: spans, totals, board classes.
    void derive() {
        nb[0] = 169; span[0] = 169;
        if (kind == ABS_PERCENTILE) {
            for (int s = 1; s < 4; ++s) span[s] = nb[s];
        } else {
            span[1] = nbin[1] * sub; span[2] = nbin[2] * sub; span[3] = nbin[3];
            for (int s = 1; s < 4; ++s) nb[s] = n_board_classes(bmode[s], s + 2) * span[s];
        }
        auto classes = [&](const std::vector<uint64_t>& keys, int n, int mode, std::vector<uint16_t>& out) {
            out.assign(keys.size(), 0);
            if (kind != ABS_NESTED || mode == BOARDS_NONE) return;
            for (size_t i = 0; i < keys.size(); ++i) {
                int bd[5];
                key_cards(keys[i], bd);
                out[i] = (uint16_t)board_class(mode, bd, n);
            }
        };
        classes(flop_keys, 3, bmode[1], flop_cls);
        classes(turn_keys, 4, bmode[2], turn_cls);
    }

    void rehash() {
        uint64_t h = fnv1a((const uint8_t*)nb, sizeof(nb));
        h = fnv1a((const uint8_t*)flop_keys.data(), flop_keys.size() * 8, h);
        h = fnv1a(flop_tab.data(), flop_tab.size(), h);
        h = fnv1a((const uint8_t*)turn_keys.data(), turn_keys.size() * 8, h);
        h = fnv1a(turn_tab.data(), turn_tab.size(), h);
        h = fnv1a((const uint8_t*)river_thr.data(), river_thr.size() * 4, h);
        if (kind != ABS_PERCENTILE) {
            int32_t p[9] = {kind, sub, nbin[1], nbin[2], nbin[3], bmode[1], bmode[2], bmode[3], 0};
            h = fnv1a((const uint8_t*)p, sizeof(p), h);
        }
        hash = h;
    }

    // ---- io -------------------------------------------------------------------
    void save(const std::string& path) const {
        FILE* f = std::fopen(path.c_str(), "wb");
        if (!f) throw std::runtime_error("cannot write " + path);
        auto W = [&](const void* p, size_t n) { if (n && std::fwrite(p, 1, n, f) != n) throw std::runtime_error("write failed"); };
        // Percentile files keep the first format, so their bytes and hashes
        // (and every bundle made with them) do not change.
        if (kind == ABS_PERCENTILE) {
            W("DONKABS1", 8);
        } else {
            W("DONKABS2", 8);
            int32_t p[8] = {kind, sub, nbin[1], nbin[2], nbin[3], bmode[1], bmode[2], bmode[3]};
            W(p, sizeof(p));
        }
        W(nb, sizeof(nb));
        auto vec = [&](const std::vector<uint64_t>& k, const std::vector<uint32_t>& m, const std::vector<uint8_t>& t) {
            uint64_t n = k.size();
            W(&n, 8); W(k.data(), n * 8); W(m.data(), n * 4); W(t.data(), t.size());
        };
        vec(flop_keys, flop_mult, flop_tab);
        vec(turn_keys, turn_mult, turn_tab);
        uint64_t nr = river_thr.size();
        W(&nr, 8); W(river_thr.data(), nr * 4);
        W(&hash, 8);
        std::fclose(f);
    }

    void load(const std::string& path) {
        FILE* f = std::fopen(path.c_str(), "rb");
        if (!f) throw std::runtime_error("cannot open " + path);
        auto R = [&](void* p, size_t n) { if (n && std::fread(p, 1, n, f) != n) throw std::runtime_error("truncated abstraction file"); };
        char magic[8];
        R(magic, 8);
        kind = ABS_PERCENTILE;
        sub = 1;
        if (!std::memcmp(magic, "DONKABS2", 8)) {
            int32_t p[8];
            R(p, sizeof(p));
            kind = p[0]; sub = p[1];
            nbin[1] = p[2]; nbin[2] = p[3]; nbin[3] = p[4];
            bmode[1] = p[5]; bmode[2] = p[6]; bmode[3] = p[7];
            if (kind != ABS_NESTED || sub < 1 || sub > 255) throw std::runtime_error("corrupt abstraction file");
            for (int s = 1; s < 4; ++s)
                if (nbin[s] < 1 || nbin[s] > 255 || bmode[s] < BOARDS_NONE || bmode[s] > BOARDS_TEXTURE)
                    throw std::runtime_error("corrupt abstraction file");
        } else if (std::memcmp(magic, "DONKABS1", 8)) {
            throw std::runtime_error("not a donksolver abstraction file");
        }
        R(nb, sizeof(nb));
        auto vec = [&](std::vector<uint64_t>& k, std::vector<uint32_t>& m, std::vector<uint8_t>& t) {
            uint64_t n;
            R(&n, 8);
            if (n > 100000) throw std::runtime_error("corrupt abstraction file");
            k.resize(n); m.resize(n); t.resize(n * 1326);
            R(k.data(), n * 8); R(m.data(), n * 4); R(t.data(), t.size());
        };
        vec(flop_keys, flop_mult, flop_tab);
        vec(turn_keys, turn_mult, turn_tab);
        uint64_t nr;
        R(&nr, 8);
        if (nr > 256) throw std::runtime_error("corrupt abstraction file");
        river_thr.resize(nr);
        R(river_thr.data(), nr * 4);
        uint64_t stored;
        R(&stored, 8);
        std::fclose(f);
        if (kind == ABS_PERCENTILE)
            for (int s = 1; s < 4; ++s) { nbin[s] = nb[s]; bmode[s] = BOARDS_NONE; }
        derive();
        rehash();
        if (stored != hash) throw std::runtime_error("abstraction file checksum mismatch");
    }

    // Debug/audit: raw street values for the canonical board of `board`
    // (flop n=3: E[HS^2] over turn+river; turn n=4: over rivers), in CANONICAL
    // holding indices, plus the stored buckets. Returns the board index.
    int debug_values(int n, const int* board, float* vals, uint8_t* tab) {
        Canon c = canonical_board(board, n);
        const auto& keys = n == 3 ? flop_keys : turn_keys;
        int bi = index_of(keys, c.key);
        if (bi < 0) return -1;
        std::vector<uint64_t> one(1, c.key);
        std::vector<float> v;
        compute_street(n, one, v, 1, nullptr);
        std::memcpy(vals, v.data(), 1326 * sizeof(float));
        const auto& t = n == 3 ? flop_tab : turn_tab;
        std::memcpy(tab, &t[(size_t)bi * 1326], 1326);
        return bi;
    }

    // Debug/audit, nested kind: e1, e2 and the stored bucket (without the board
    // class) of every holding on `board` (n = 3 or 4), indexed by the ORIGINAL
    // holding index; e1 < 0 marks holdings that share a card with the board.
    int debug_nested(int n, const int* board, double* e1, double* e2, int32_t* tab) const {
        Canon c = canonical_board(board, n);
        const auto& keys = n == 3 ? flop_keys : turn_keys;
        int bi = index_of(keys, c.key);
        if (bi < 0) return -1;
        int bd[5];
        key_cards(c.key, bd);
        NestedWork w;
        std::vector<double> a(1326), b(1326);
        street_metrics(n, bd, a.data(), b.data(), w);
        const auto& t = n == 3 ? flop_tab : turn_tab;
        const auto& H = holdings();
        for (int h = 0; h < 1326; ++h) {
            int ch = holding_index(c.map_card(H.lo[h]), c.map_card(H.hi[h]));
            e1[h] = a[ch]; e2[h] = b[ch];
            tab[h] = a[ch] < 0 ? -1 : t[(size_t)bi * 1326 + ch];
        }
        return bi;
    }

private:
    static int index_of(const std::vector<uint64_t>& keys, uint64_t key) {
        auto it = std::lower_bound(keys.begin(), keys.end(), key);
        if (it == keys.end() || *it != key) return -1;
        return (int)(it - keys.begin());
    }

    static void enumerate_iso(int n, std::vector<uint64_t>& keys, std::vector<uint32_t>& mult) {
        std::vector<uint64_t> all;
        int b[4];
        if (n == 3) {
            for (b[0] = 0; b[0] < 52; ++b[0]) for (b[1] = b[0] + 1; b[1] < 52; ++b[1])
                for (b[2] = b[1] + 1; b[2] < 52; ++b[2]) all.push_back(canonical_board(b, 3).key);
        } else {
            for (b[0] = 0; b[0] < 52; ++b[0]) for (b[1] = b[0] + 1; b[1] < 52; ++b[1])
                for (b[2] = b[1] + 1; b[2] < 52; ++b[2]) for (b[3] = b[2] + 1; b[3] < 52; ++b[3])
                    all.push_back(canonical_board(b, 4).key);
        }
        std::sort(all.begin(), all.end());
        keys.clear(); mult.clear();
        for (size_t i = 0; i < all.size();) {
            size_t j = i;
            while (j < all.size() && all[j] == all[i]) ++j;
            keys.push_back(all[i]);
            mult.push_back((uint32_t)(j - i));
            i = j;
        }
    }

    // canonical key -> the board cards it denotes (suits 0..3 in canonical order)
    static int key_cards(uint64_t key, int* out) {
        int n = 0;
        for (int k = 0; k < 4; ++k) {
            uint32_t m = (uint32_t)((key >> (13 * (3 - k))) & 0x1FFF);
            for (int r = 0; r < 13; ++r) if ((m >> r) & 1) out[n++] = make_card(r, k);
        }
        return n;
    }

    void compute_street(int n, const std::vector<uint64_t>& keys, std::vector<float>& vals, int threads,
                        const std::function<void(const char*, double)>& progress) {
        vals.assign(keys.size() * 1326, -1.0f);
        std::atomic<size_t> next{0}, done{0};
        auto work = [&]() {
            HSWork w;
            std::vector<double> acc(1326);
            std::vector<int> cnt(1326);
            for (;;) {
                size_t bi = next.fetch_add(1);
                if (bi >= keys.size()) break;
                int bd[5];
                key_cards(keys[bi], bd);
                uint64_t dead = 0;
                for (int i = 0; i < n; ++i) dead |= 1ULL << bd[i];
                std::fill(acc.begin(), acc.end(), 0.0);
                std::fill(cnt.begin(), cnt.end(), 0);
                if (n == 3) {
                    for (int t = 0; t < 52; ++t) {
                        if ((dead >> t) & 1) continue;
                        for (int r = t + 1; r < 52; ++r) {
                            if ((dead >> r) & 1) continue;
                            bd[3] = t; bd[4] = r;
                            hs_all(bd, nullptr, acc.data(), cnt.data(), w);
                        }
                    }
                } else {
                    for (int r = 0; r < 52; ++r) {
                        if ((dead >> r) & 1) continue;
                        bd[4] = r;
                        hs_all(bd, nullptr, acc.data(), cnt.data(), w);
                    }
                }
                float* row = &vals[bi * 1326];
                DONK_NO_VECTOR
                for (int h = 0; h < 1326; ++h)
                    if (cnt[h]) row[h] = (float)(acc[h] / cnt[h]);
                size_t d = done.fetch_add(1) + 1;
                if (progress && (d % 64 == 0 || d == keys.size()))
                    progress(n == 3 ? "flop" : "turn", (double)d / keys.size());
            }
        };
        run_threads(threads, work);
    }

    // ---- nested kind ---------------------------------------------------------------
    struct NestedWork {
        HSWork hw;
        std::vector<double> hs, sum, acc1, acc2;
        std::vector<int> cnt, n;
        NestedWork() : hs(1326), sum(52 * 1326), acc1(1326), acc2(1326), cnt(52 * 1326), n(1326) {}
    };

    // e1/e2 of every holding on the board bd[0..n) (see the file header);
    // e1 = e2 = -1 for holdings that share a card with the board.
    static void street_metrics(int n, int* bd, double* e1, double* e2, NestedWork& w) {
        uint64_t dead = 0;
        for (int i = 0; i < n; ++i) dead |= 1ULL << bd[i];
        const auto& H = holdings();
        if (n == 4) {
            std::fill(w.acc1.begin(), w.acc1.end(), 0.0);
            std::fill(w.acc2.begin(), w.acc2.end(), 0.0);
            std::fill(w.n.begin(), w.n.end(), 0);
            for (int r = 0; r < 52; ++r) {
                if ((dead >> r) & 1) continue;
                bd[4] = r;
                std::fill(w.hs.begin(), w.hs.end(), -1.0);
                hs_all(bd, w.hs.data(), nullptr, nullptr, w.hw);
                for (int h = 0; h < 1326; ++h)
                    if (w.hs[h] >= 0) { w.acc1[h] += w.hs[h]; w.acc2[h] += w.hs[h] * w.hs[h]; w.n[h]++; }
            }
            for (int h = 0; h < 1326; ++h) {
                if (w.n[h]) { e1[h] = w.acc1[h] / w.n[h]; e2[h] = w.acc2[h] / w.n[h]; }
                else { e1[h] = -1; e2[h] = -1; }
            }
            return;
        }
        // flop: every runout {t, r} is evaluated once and credited to both of
        // its cards, so sum[t][h] / cnt[t][h] is the equity of h on turn t.
        std::fill(w.sum.begin(), w.sum.end(), 0.0);
        std::fill(w.cnt.begin(), w.cnt.end(), 0);
        for (int t = 0; t < 52; ++t) {
            if ((dead >> t) & 1) continue;
            for (int r = t + 1; r < 52; ++r) {
                if ((dead >> r) & 1) continue;
                bd[3] = t; bd[4] = r;
                std::fill(w.hs.begin(), w.hs.end(), -1.0);
                hs_all(bd, w.hs.data(), nullptr, nullptr, w.hw);
                double* st = &w.sum[(size_t)t * 1326];
                double* sr = &w.sum[(size_t)r * 1326];
                int* ct = &w.cnt[(size_t)t * 1326];
                int* cr = &w.cnt[(size_t)r * 1326];
                for (int h = 0; h < 1326; ++h)
                    if (w.hs[h] >= 0) { st[h] += w.hs[h]; ct[h]++; sr[h] += w.hs[h]; cr[h]++; }
            }
        }
        for (int h = 0; h < 1326; ++h) {
            if (((dead >> H.lo[h]) & 1) || ((dead >> H.hi[h]) & 1)) { e1[h] = -1; e2[h] = -1; continue; }
            double a = 0, b = 0;
            int m = 0;
            for (int t = 0; t < 52; ++t) {
                int c = w.cnt[(size_t)t * 1326 + h];
                if (!c) continue;
                double et = w.sum[(size_t)t * 1326 + h] / c;
                a += et; b += et * et; ++m;
            }
            e1[h] = a / m; e2[h] = b / m;
        }
    }

    // The nested bucket of every holding on one board: equity bin, then the
    // potential sub-bin inside that bin.
    static void nested_row(const double* e1, const double* e2, int N, int K, uint8_t* row) {
        double lo = 2, hi = -1;
        for (int h = 0; h < 1326; ++h)
            if (e1[h] >= 0) { lo = std::min(lo, e1[h]); hi = std::max(hi, e1[h]); }
        int bin[1326];
        std::vector<double> blo((size_t)N, 2.0), bhi((size_t)N, -1.0);
        for (int h = 0; h < 1326; ++h) {
            if (e1[h] < 0) { bin[h] = -1; continue; }
            int k = width_bin(e1[h], N, lo, hi);
            bin[h] = k;
            blo[(size_t)k] = std::min(blo[(size_t)k], e2[h]);
            bhi[(size_t)k] = std::max(bhi[(size_t)k], e2[h]);
        }
        for (int h = 0; h < 1326; ++h) {
            if (bin[h] < 0) { row[h] = 0xFF; continue; }
            int k = bin[h];
            row[h] = (uint8_t)(k * K + width_bin(e2[h], K, blo[(size_t)k], bhi[(size_t)k]));
        }
    }

    void nested_street(int n, const std::vector<uint64_t>& keys, std::vector<uint8_t>& tab, int N, int threads,
                       const std::function<void(const char*, double)>& progress) {
        tab.assign(keys.size() * 1326, 0xFF);
        std::atomic<size_t> next{0}, done{0};
        int K = sub;
        auto work = [&]() {
            NestedWork w;
            std::vector<double> e1(1326), e2(1326);
            for (;;) {
                size_t bi = next.fetch_add(1);
                if (bi >= keys.size()) break;
                int bd[5];
                key_cards(keys[bi], bd);
                street_metrics(n, bd, e1.data(), e2.data(), w);
                nested_row(e1.data(), e2.data(), N, K, &tab[bi * 1326]);
                size_t d = done.fetch_add(1) + 1;
                if (progress && (d % 64 == 0 || d == keys.size()))
                    progress(n == 3 ? "flop" : "turn", (double)d / keys.size());
            }
        };
        run_threads(threads, work);
    }

    static std::vector<uint8_t> quantise(const std::vector<float>& vals, const std::vector<uint64_t>& keys,
                                         const std::vector<uint32_t>& mult, int B) {
        struct P { float v; uint32_t w; };
        std::vector<P> ps;
        ps.reserve(vals.size());
        for (size_t bi = 0; bi < keys.size(); ++bi)
            for (int h = 0; h < 1326; ++h) {
                float v = vals[bi * 1326 + h];
                if (v >= 0) ps.push_back({v, mult[bi]});
            }
        std::sort(ps.begin(), ps.end(), [](const P& a, const P& b) { return a.v < b.v; });
        double tot = 0;
        for (auto& p : ps) tot += p.w;
        std::vector<float> thr;
        double acc = 0;
        size_t k = 0;
        for (int b = 1; b < B; ++b) {
            double target = tot * b / B;
            while (k < ps.size() && acc + ps[k].w <= target) acc += ps[k++].w;
            thr.push_back(k < ps.size() ? ps[k].v : ps.back().v);
        }
        std::vector<uint8_t> tab(vals.size(), 0xFF);
        for (size_t i = 0; i < vals.size(); ++i)
            if (vals[i] >= 0)
                tab[i] = (uint8_t)(std::upper_bound(thr.begin(), thr.end(), vals[i]) - thr.begin());
        return tab;
    }

    void river_thresholds(int B, uint64_t boards, uint64_t seed, int threads,
                          const std::function<void(const char*, double)>& progress) {
        std::vector<std::vector<float>> parts((size_t)threads);
        std::atomic<uint64_t> next{0};
        auto work = [&](int tid) {
            HSWork w;
            std::vector<double> hs(1326);
            auto& out = parts[(size_t)tid];
            for (;;) {
                uint64_t i = next.fetch_add(1);
                if (i >= boards) break;
                // seeded by board index, not thread: the file (and its hash,
                // the frozen-policy identity) must not depend on scheduling
                Rng rng(seed * 1000003ULL + i * 0x9E3779B97F4A7C15ULL);
                int deck[52];
                for (int c = 0; c < 52; ++c) deck[c] = c;
                for (int k = 0; k < 5; ++k) { int j = k + (int)rng.below(52 - k); std::swap(deck[k], deck[j]); }
                std::fill(hs.begin(), hs.end(), -1.0);
                hs_all(deck, hs.data(), nullptr, nullptr, w);
                for (double v : hs) if (v >= 0) out.push_back((float)v);
                if (progress && tid == 0 && i % 256 == 0) progress("river", (double)i / boards);
            }
        };
        std::vector<std::thread> th;
        for (int t = 0; t < threads; ++t) th.emplace_back(work, t);
        for (auto& x : th) x.join();
        std::vector<float> all;
        for (auto& p : parts) all.insert(all.end(), p.begin(), p.end());
        std::sort(all.begin(), all.end());
        river_thr.clear();
        for (int b = 1; b < B; ++b) river_thr.push_back(all[(size_t)((double)all.size() * b / B)]);
        if (progress) progress("river", 1.0);
    }

    static void run_threads(int threads, const std::function<void()>& fn) {
        std::vector<std::thread> th;
        for (int t = 0; t < threads; ++t) th.emplace_back(fn);
        for (auto& x : th) x.join();
    }
};

}  // namespace donk
