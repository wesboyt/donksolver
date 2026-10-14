// cards.hpp -- card ids, the 7-card evaluator, holding indices, RNG.
//
// CARD IDS: id = rank*4 + suit, rank 0..12 = '2'..'A', suit 0..3 =
// 'c','d','h','s'.
//
// The evaluator returns a uint32 whose ORDER is the poker order (bigger wins,
// equal = split). Only the order is part of the contract; the audit
// (tests/audit_eval.py) checks it pairwise against pokerfast's evaluator and
// against an independent best-of-21 reference compiled from this file.
#pragma once
#include <array>
#include <cstdint>
#include <cstring>
#include <string>

namespace donk {

inline int card_rank(int c) { return c >> 2; }
inline int card_suit(int c) { return c & 3; }
inline int make_card(int rank, int suit) { return rank * 4 + suit; }

inline int parse_card(const char* s) {
    static const char* R = "23456789TJQKA";
    static const char* S = "cdhs";
    int r = -1, su = -1;
    for (int i = 0; i < 13; ++i) if (R[i] == s[0] || (R[i] | 32) == s[0]) r = i;
    for (int i = 0; i < 4; ++i) if (S[i] == s[1] || (S[i] & ~32) == s[1]) su = i;
    if (r < 0 || su < 0) return -1;
    return make_card(r, su);
}

inline std::string card_str(int c) {
    static const char* R = "23456789TJQKA";
    static const char* S = "cdhs";
    std::string o;
    o += R[card_rank(c)];
    o += S[card_suit(c)];
    return o;
}

// Suit-layout mask: bit (suit*13 + rank). Suit masks fall out as 13-bit fields.
inline uint64_t sbit(int c) { return 1ULL << (card_suit(c) * 13 + card_rank(c)); }

#if defined(_MSC_VER)
#include <intrin.h>
inline int msb32(uint32_t x) { unsigned long i; _BitScanReverse(&i, x); return (int)i; }
#if defined(_M_ARM64) || defined(_M_ARM)
inline int popc32(uint32_t x) { return (int)_CountOneBits(x); }
inline int popc64(uint64_t x) { return (int)_CountOneBits64(x); }
#else
inline int popc32(uint32_t x) { return (int)__popcnt(x); }
inline int popc64(uint64_t x) { return (int)__popcnt64(x); }
#endif
inline int ctz64(uint64_t x) { unsigned long i; _BitScanForward64(&i, x); return (int)i; }
#else
inline int msb32(uint32_t x) { return 31 - __builtin_clz(x); }
inline int popc32(uint32_t x) { return __builtin_popcount(x); }
inline int popc64(uint64_t x) { return __builtin_popcountll(x); }
inline int ctz64(uint64_t x) { return __builtin_ctzll(x); }
#endif

namespace detail {
struct EvalTables {
    uint8_t straight_hi[8192];  // 0 = none, else top rank + 1 (wheel -> 4)
    EvalTables() {
        for (int m = 0; m < 8192; ++m) {
            int hi = 0;
            for (int h = 12; h >= 4; --h)
                if (((m >> (h - 4)) & 0x1F) == 0x1F) { hi = h + 1; break; }
            if (!hi && (m & 0xF) == 0xF && ((m >> 12) & 1)) hi = 4;
            straight_hi[m] = (uint8_t)hi;
        }
    }
};
inline const EvalTables& tables() { static const EvalTables t; return t; }
inline uint32_t topn(uint32_t mask, int n, int shift) {
    uint32_t v = 0;
    for (int k = 0; k < n && mask; ++k) {
        int r = msb32(mask);
        mask &= ~(1u << r);
        v |= (uint32_t)r << (shift - 4 * k);
    }
    return v;
}
}  // namespace detail

enum : uint32_t { HIGH = 0, PAIR, TWO_PAIR, TRIPS, STRAIGHT, FLUSH, FULL_HOUSE, QUADS, STRAIGHT_FLUSH };

// Rank 5..7 cards given as a suit-layout mask. value = cat<<20 | nibbles.
inline uint32_t eval_mask(uint64_t m) {
    const auto& T = detail::tables();
    uint32_t s0 = (uint32_t)(m & 0x1FFF), s1 = (uint32_t)((m >> 13) & 0x1FFF);
    uint32_t s2 = (uint32_t)((m >> 26) & 0x1FFF), s3 = (uint32_t)((m >> 39) & 0x1FFF);
    uint32_t su[4] = {s0, s1, s2, s3};
    for (int k = 0; k < 4; ++k) {
        if (popc32(su[k]) >= 5) {
            // With at most 7 cards a flush excludes quads and full houses.
            int sf = T.straight_hi[su[k]];
            if (sf) return (STRAIGHT_FLUSH << 20) | ((uint32_t)sf << 16);
            return (FLUSH << 20) | detail::topn(su[k], 5, 16);
        }
    }
    uint32_t ranks = s0 | s1 | s2 | s3;
    uint32_t quads = s0 & s1 & s2 & s3;
    if (quads) {
        int q = msb32(quads);
        uint32_t rest = ranks & ~(1u << q);
        return (QUADS << 20) | ((uint32_t)q << 16) | ((uint32_t)msb32(rest) << 12);
    }
    uint32_t ge3 = (s0 & s1 & s2) | (s0 & s1 & s3) | (s0 & s2 & s3) | (s1 & s2 & s3);
    uint32_t ge2 = (s0 & s1) | (s0 & s2) | (s0 & s3) | (s1 & s2) | (s1 & s3) | (s2 & s3);
    uint32_t pairs = ge2 & ~ge3;
    if (ge3) {
        int t = msb32(ge3);
        uint32_t pc = pairs | (ge3 & ~(1u << t));  // a second trips plays as the pair
        if (pc) return (FULL_HOUSE << 20) | ((uint32_t)t << 16) | ((uint32_t)msb32(pc) << 12);
    }
    int st = T.straight_hi[ranks];
    if (st) return (STRAIGHT << 20) | ((uint32_t)st << 16);
    if (ge3) {
        int t = msb32(ge3);
        return (TRIPS << 20) | ((uint32_t)t << 16) | detail::topn(ranks & ~(1u << t), 2, 12);
    }
    if (pairs) {
        int p1 = msb32(pairs);
        uint32_t rest = pairs & ~(1u << p1);
        if (rest) {
            int p2 = msb32(rest);
            uint32_t k = ranks & ~(1u << p1) & ~(1u << p2);
            return (TWO_PAIR << 20) | ((uint32_t)p1 << 16) | ((uint32_t)p2 << 12) |
                   ((uint32_t)msb32(k) << 8);
        }
        return (PAIR << 20) | ((uint32_t)p1 << 16) | detail::topn(ranks & ~(1u << p1), 3, 12);
    }
    return (HIGH << 20) | detail::topn(ranks, 5, 16);
}

inline uint32_t eval_cards(const int* cards, int n) {
    uint64_t m = 0;
    for (int i = 0; i < n; ++i) m |= sbit(cards[i]);
    return eval_mask(m);
}

// Independent reference: classify every 5-card subset the textbook way and
// keep the best. Slow; used only by the audit, never by the solver.
inline uint32_t eval5_reference(const int* c) {
    int r[5], cnt[13] = {0};
    bool flush = true;
    for (int i = 0; i < 5; ++i) {
        r[i] = card_rank(c[i]);
        cnt[r[i]]++;
        if (card_suit(c[i]) != card_suit(c[0])) flush = false;
    }
    // ranks sorted by (count desc, rank desc)
    int ord[5], n = 0;
    for (int want = 4; want >= 1; --want)
        for (int rk = 12; rk >= 0; --rk)
            if (cnt[rk] == want) ord[n++] = rk;
    int distinct = n;
    bool straight = false;
    int top = -1;
    if (distinct == 5) {
        if (ord[0] - ord[4] == 4) { straight = true; top = ord[0]; }
        else if (ord[0] == 12 && ord[1] == 3 && ord[4] == 0) { straight = true; top = 3; }
    }
    auto pack = [](uint32_t cat, const int* rs, int k) {
        uint32_t v = cat << 20;
        for (int i = 0; i < k; ++i) v |= (uint32_t)rs[i] << (16 - 4 * i);
        return v;
    };
    int one[1];
    if (straight && flush) { one[0] = top + 1; return pack(STRAIGHT_FLUSH, one, 1); }
    int c0 = cnt[ord[0]];
    if (c0 == 4) return pack(QUADS, ord, 2);
    if (c0 == 3 && distinct == 2) return pack(FULL_HOUSE, ord, 2);
    if (flush) return pack(FLUSH, ord, 5);
    if (straight) { one[0] = top + 1; return pack(STRAIGHT, one, 1); }
    if (c0 == 3) return pack(TRIPS, ord, 3);
    if (c0 == 2 && distinct == 3) return pack(TWO_PAIR, ord, 3);
    if (c0 == 2) return pack(PAIR, ord, 4);
    return pack(HIGH, ord, 5);
}

inline uint32_t eval_reference(const int* cards, int n) {
    uint32_t best = 0;
    int sub[5];
    for (int a = 0; a < n; ++a)
        for (int b = a + 1; b < n; ++b)
            for (int c = b + 1; c < n; ++c)
                for (int d = c + 1; d < n; ++d)
                    for (int e = d + 1; e < n; ++e) {
                        sub[0] = cards[a]; sub[1] = cards[b]; sub[2] = cards[c];
                        sub[3] = cards[d]; sub[4] = cards[e];
                        uint32_t v = eval5_reference(sub);
                        if (v > best) best = v;
                    }
    return best;
}

// ---- holdings -------------------------------------------------------------
// 1326 two-card holdings, index(c1<c2) = c2*(c2-1)/2 + c1.
inline int holding_index(int a, int b) {
    if (a > b) { int t = a; a = b; b = t; }
    return b * (b - 1) / 2 + a;
}
struct HoldingTable {
    uint8_t lo[1326], hi[1326];
    HoldingTable() {
        for (int b = 1; b < 52; ++b)
            for (int a = 0; a < b; ++a) {
                int i = holding_index(a, b);
                lo[i] = (uint8_t)a;
                hi[i] = (uint8_t)b;
            }
    }
};
inline const HoldingTable& holdings() { static const HoldingTable t; return t; }

// 169 preflop classes on the 13x13 grid: pair (r,r); suited (hi,lo) above the
// diagonal; offsuit (lo,hi) below it.
inline int preflop_class(int a, int b) {
    int ra = card_rank(a), rb = card_rank(b);
    int hi = ra > rb ? ra : rb, lo = ra > rb ? rb : ra;
    if (hi == lo) return hi * 13 + hi;
    if (card_suit(a) == card_suit(b)) return hi * 13 + lo;
    return lo * 13 + hi;
}

// ---- RNG: xoshiro256++ ----------------------------------------------------
struct Rng {
    uint64_t s[4];
    static uint64_t splitmix(uint64_t& x) {
        uint64_t z = (x += 0x9E3779B97F4A7C15ULL);
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
        return z ^ (z >> 31);
    }
    explicit Rng(uint64_t seed = 1) { seed_with(seed); }
    void seed_with(uint64_t seed) {
        uint64_t x = seed;
        for (auto& v : s) v = splitmix(x);
    }
    static uint64_t rotl(uint64_t x, int k) { return (x << k) | (x >> (64 - k)); }
    uint64_t next() {
        uint64_t r = rotl(s[0] + s[3], 23) + s[0];
        uint64_t t = s[1] << 17;
        s[2] ^= s[0]; s[3] ^= s[1]; s[1] ^= s[2]; s[0] ^= s[3];
        s[2] ^= t; s[3] = rotl(s[3], 45);
        return r;
    }
    double uniform() { return (next() >> 11) * (1.0 / 9007199254740992.0); }
    // Lemire's nearly-divisionless bounded draw, unbiased.
    uint32_t below(uint32_t n) {
        uint64_t x = (uint32_t)next();
        uint64_t m = x * n;
        uint32_t l = (uint32_t)m;
        if (l < n) {
            uint32_t t = (0u - n) % n;
            while (l < t) { x = (uint32_t)next(); m = x * n; l = (uint32_t)m; }
        }
        return (uint32_t)(m >> 32);
    }
};

}  // namespace donk
