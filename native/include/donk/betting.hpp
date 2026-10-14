// betting.hpp -- no-limit betting rules, and settlement.
//
// The rules are pokerkit's cash-game no-limit hold'em semantics, not textbook
// poker. Three rules differ from the textbook and are reproduced:
//   1. every raise re-queues every other live player;
//   2. re-raising is governed by cbra (largest increment), consec (sum of
//      consecutive short all-in increments) and acted-since-full-raise;
//   3. the minimum raise-to is capped by the effective stack.
// Settlement layers side pots by commitment, merges adjacent layers with the
// same WINNER set, gives odd chips to the lowest seat, and applies the rake
// (a share of the high-water pot, capped, no flop no drop, charged to winners
// in proportion to what they won). tests/audit_payoff.py compares all of this
// against an independent engine (pokerfast.FastHand) on random lines.
#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

#include "cards.hpp"

namespace donk {

static constexpr int MAXP = 6;

struct RakeRule {
    double pct = 0.0;           // 0 disables rake
    int64_t cap = 0;            // chips; <= 0 means uncapped
    bool no_flop_no_drop = true;
};

struct BetState {
    int n = 0;
    int64_t sb = 0, bb = 0;
    int64_t stacks[MAXP] = {0}, bets[MAXP] = {0}, committed[MAXP] = {0};
    bool folded[MAXP] = {false}, allin[MAXP] = {false};
    int8_t queue[2 * MAXP] = {0};
    int qlen = 0;
    uint32_t acted = 0;          // bitmask of seats acted since last full raise
    int64_t cbra = 0;
    int64_t consec_sum = 0;
    int consec_n = 0;
    int street = 0;              // 0 preflop .. 3 river
    int raises = 0;              // bets+raises this street (menu level)
    int64_t pot = 0;             // total committed so far (== final pot high-water)
    bool done = false;
    bool showdown = false;

    bool live(int i) const { return !folded[i] && !allin[i]; }
    int in_hand() const { int k = 0; for (int i = 0; i < n; ++i) k += !folded[i]; return k; }
    int live_count() const { int k = 0; for (int i = 0; i < n; ++i) k += live(i); return k; }
    int64_t max_bet() const { int64_t m = 0; for (int i = 0; i < n; ++i) m = std::max(m, bets[i]); return m; }
    int actor() const { return qlen ? queue[0] : -1; }

    void post(int i, int64_t amount) {
        int64_t amt = std::min(amount, stacks[i]);
        bets[i] += amt; stacks[i] -= amt; committed[i] += amt; pot += amt;
        if (stacks[i] == 0) allin[i] = true;
    }

    void begin_betting(int opener) {
        qlen = 0;
        for (int k = 0; k < n; ++k) {
            int s = (opener + k) % n;
            if (live(s)) queue[qlen++] = (int8_t)s;
        }
        acted = 0; cbra = 0; consec_sum = 0; consec_n = 0; raises = 0;
    }

    static BetState start(int n, const int64_t* stk, int64_t sb, int64_t bb) {
        BetState s;
        s.n = n; s.sb = sb; s.bb = bb;
        for (int i = 0; i < n; ++i) s.stacks[i] = stk[i];
        s.post(0, sb);
        s.post(1 % n, bb);
        s.begin_betting(2 % n);
        return s;
    }

    int64_t effective_stack(int i) const {
        // second-largest total (bets+stacks) among players still in the hand
        int64_t a = -1, b = -1;
        int cnt = 0;
        for (int j = 0; j < n; ++j) {
            if (folded[j]) continue;
            ++cnt;
            int64_t t = bets[j] + stacks[j];
            if (t > a) { b = a; a = t; } else if (t > b) { b = t; }
        }
        if (cnt < 2) return stacks[i];
        return std::min(stacks[i], std::max<int64_t>(0, b - bets[i]));
    }

    bool can_raise(int i) const {
        if (consec_n && consec_sum < cbra && ((acted >> i) & 1)) return false;
        int64_t mb = max_bet();
        if (stacks[i] <= mb - bets[i]) return false;
        for (int j = 0; j < n; ++j)
            if (j != i && !folded[j] && stacks[j] + bets[j] > mb) return true;
        return false;
    }

    int64_t to_call(int i) const { return std::min(max_bet() - bets[i], stacks[i]); }

    // min_bet / max_bet in ADDITIONAL chips for seat i
    void raise_bounds(int i, int64_t& mn, int64_t& mx) const {
        int64_t mb = max_bet();
        int64_t amount = std::max(cbra, bb) + mb;
        int64_t min_to = std::min(effective_stack(i) + bets[i], amount);
        mn = min_to - bets[i];
        mx = stacks[i];
    }

    void pop_actor() {
        int i = queue[0];
        for (int k = 1; k < qlen; ++k) queue[k - 1] = queue[k];
        --qlen;
        acted |= 1u << i;
    }

    void fold() {
        int i = queue[0];
        pop_actor();
        folded[i] = true;
        post_action();
    }
    void check_call() {
        int i = queue[0];
        int64_t c = to_call(i);
        if (c) commit(i, c);
        pop_actor();
        post_action();
    }
    // chips = ADDITIONAL amount; clamped to [min,max] exactly like bet_or_raise
    void bet_raise(int64_t chips) {
        int i = queue[0];
        if (!can_raise(i)) { check_call(); return; }
        int64_t mn, mx;
        raise_bounds(i, mn, mx);
        if (chips > mx) chips = mx; else if (chips < mn) chips = mn;
        int64_t prev = max_bet();
        pop_actor();
        commit(i, chips);
        int64_t inc = bets[i] - prev;
        qlen = 0;
        for (int k = 0; k < n; ++k) {
            int s = (i + 1 + k) % n;
            if (s != i && live(s)) queue[qlen++] = (int8_t)s;
        }
        if (inc >= cbra) acted = 1u << i;
        if (inc > cbra) cbra = inc;
        if (stacks[i]) { consec_sum = 0; consec_n = 0; }
        else { consec_sum += inc; ++consec_n; }
        ++raises;
        post_action();
    }

    void commit(int i, int64_t amount) { post(i, amount); }

    void post_action() {
        if (in_hand() <= 1) { done = true; showdown = false; qlen = 0; return; }
        if (qlen) return;
        for (;;) {
            for (int i = 0; i < n; ++i) bets[i] = 0;
            if (street >= 3) { done = true; showdown = true; return; }
            ++street;
            if (live_count() <= 1) continue;
            begin_betting(0);
            if (qlen) return;
        }
    }
};

// ---- settlement -------------------------------------------------------------
// value[i] = hand strength (bigger wins) for seats in hand; ignored for folded.
// integer_chips: whole-chip pots (floor division, odd chips to lowest seat).
// Otherwise the pot is split exactly (the textbook convention for toy games).
inline void settle(int n, const int64_t* committed, const bool* folded, const uint32_t* value,
                   bool showdown, bool saw_flop, const RakeRule& rake, bool integer_chips,
                   double* out) {
    double won[MAXP] = {0};
    int64_t total = 0;
    for (int i = 0; i < n; ++i) total += committed[i];
    int alive[MAXP], na = 0;
    for (int i = 0; i < n; ++i) if (!folded[i]) alive[na++] = i;
    if (na == 1 || !showdown) {
        won[alive[0]] = (double)total;
    } else {
        int64_t levels[MAXP];
        int nl = 0;
        for (int i = 0; i < n; ++i) if (committed[i] > 0) levels[nl++] = committed[i];
        std::sort(levels, levels + nl);
        nl = (int)(std::unique(levels, levels + nl) - levels);
        struct Layer { int64_t pot; uint32_t key; };
        Layer layers[MAXP];
        int nlay = 0;
        int64_t prev = 0;
        for (int li = 0; li < nl; ++li) {
            int64_t lvl = levels[li], pot = 0;
            for (int i = 0; i < n; ++i) {
                int64_t take = std::min(committed[i], lvl) - std::min(committed[i], prev);
                if (take > 0) pot += take;
            }
            uint32_t best = 0, key = 0;
            bool any = false;
            for (int k = 0; k < na; ++k) {
                int i = alive[k];
                if (committed[i] < lvl) continue;
                if (!any || value[i] > best) { best = value[i]; key = 1u << i; any = true; }
                else if (value[i] == best) key |= 1u << i;
            }
            // key 0 = nobody in hand is eligible. Such layers merge with each
            // other and are paid to the seats whose commitment reaches the
            // FINAL level.
            if (nlay && layers[nlay - 1].key == key) layers[nlay - 1].pot += pot;
            else layers[nlay++] = {pot, key};
            prev = lvl;
        }
        uint32_t refund = 0;
        for (int i = 0; i < n; ++i) if (committed[i] >= prev) refund |= 1u << i;
        for (int l = 0; l < nlay; ++l) {
            if (!layers[l].pot) continue;
            if (!layers[l].key) layers[l].key = refund;
            if (!layers[l].key) continue;
            int w = popc32(layers[l].key);
            if (integer_chips) {
                int64_t share = layers[l].pot / w, odd = layers[l].pot % w;
                bool first = true;
                for (int i = 0; i < n; ++i)
                    if ((layers[l].key >> i) & 1) {
                        won[i] += (double)share;
                        if (first && odd) { won[i] += (double)odd; }
                        first = false;
                    }
            } else {
                for (int i = 0; i < n; ++i)
                    if ((layers[l].key >> i) & 1) won[i] += (double)layers[l].pot / w;
            }
        }
    }
    for (int i = 0; i < n; ++i) out[i] = won[i] - (double)committed[i];
    if (rake.pct <= 0.0 || total <= 0) return;
    if (rake.no_flop_no_drop && !saw_flop) return;
    double r = (double)total * rake.pct;
    if (rake.cap > 0) r = std::min(r, (double)rake.cap);
    r = std::min(r, (double)total);
    r = std::floor(r);  // whole chips: int(min(pot*pct, cap, pot))
    if (r <= 0) return;
    double pos = 0;
    for (int i = 0; i < n; ++i) if (out[i] > 0) pos += out[i];
    if (pos <= 0) return;
    for (int i = 0; i < n; ++i) if (out[i] > 0) out[i] -= r * (out[i] / pos);
}

}  // namespace donk
