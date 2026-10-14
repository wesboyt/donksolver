// tree.hpp -- explicit abstract betting trees.
//
// One flat representation serves every game: decision nodes own a block of
// table rows (buckets x actions), terminals carry the commitment vector.
// Two builders: the no-limit builder (per-street, per-raise-level menus of
// pot-after-call fractions on top of the exact betting rules of betting.hpp)
// and a fixed-limit builder for the Kuhn/Leduc validation games.
#pragma once
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "betting.hpp"

namespace donk {

enum ActKind : uint8_t { A_FOLD = 0, A_CALL = 1, A_RAISE = 2, A_ALLIN = 3 };

struct Action {
    uint8_t kind;
    float frac;      // raise: (raise_to - max_bet) / (pot + to_call) at this node
    int64_t chips;   // additional chips at this node (0 for fold/check)
};

struct Node {
    int8_t actor = -1;     // -1 terminal
    uint8_t street = 0;
    uint8_t nA = 0;
    uint8_t in_hand = 0;
    uint32_t child0 = 0;   // index into Tree::child / Tree::act
    uint32_t term = 0;     // terminal index
    uint64_t row = 0;      // first table entry
    int64_t actor_commit = 0;  // actor's total commitment before acting (fold value = -this)
};

struct Terminal {
    int64_t committed[MAXP] = {0};
    uint32_t folded = 0;
    bool showdown = false;
    bool saw_flop = false;
    bool checkdown = false;  // multiway leaf played as check-down to showdown
    double fold_payoff[MAXP] = {0};  // valid when !showdown
};

struct Tree {
    int n = 0;
    int nb[4] = {1, 1, 1, 1};
    std::vector<Node> nodes;
    std::vector<uint32_t> child;
    std::vector<Action> act;
    std::vector<Terminal> terms;
    uint64_t entries = 0;
    uint64_t decision_nodes[4] = {0, 0, 0, 0};
    uint64_t rows_per_street[4] = {0, 0, 0, 0};
    RakeRule rake;
    bool integer_chips = true;
    uint64_t checkdown_terms = 0;

    void assign_rows() {
        entries = 0;
        for (auto& v : decision_nodes) v = 0;
        for (auto& v : rows_per_street) v = 0;
        for (auto& nd : nodes) {
            if (nd.actor < 0) continue;
            nd.row = entries;
            entries += (uint64_t)nb[nd.street] * nd.nA;
            decision_nodes[nd.street]++;
            rows_per_street[nd.street] += (uint64_t)nb[nd.street];
        }
    }
};

inline void fill_fold_payoff(const Tree& t, Terminal& tm) {
    bool folded[MAXP];
    uint32_t dummy[MAXP] = {0};
    for (int i = 0; i < t.n; ++i) folded[i] = (tm.folded >> i) & 1;
    settle(t.n, tm.committed, folded, dummy, false, tm.saw_flop, t.rake, t.integer_chips,
           tm.fold_payoff);
}

// ============================================================================
// No-limit builder
// ============================================================================
struct NLConfig {
    int n = 6;
    int64_t stack = 20000, sb = 100, bb = 200;
    RakeRule rake;
    // menus[street][level] = pot-after-call fractions; the last level repeats
    std::vector<std::vector<double>> menus[4];
    std::vector<std::vector<double>> mw_menus[4];  // >= 3 in hand; empty = menus
    int max_raises[4] = {4, 3, 3, 2};
    int mw_max_raises[4] = {4, 1, 1, 1};
    double kappa_bet = 0.9;     // skip a non-all-in bet committing >= this share of the stack
    int allin_rule = 0;         // 0 always, 1 only if <= kappa_ai pot-fraction raise, 2 never
    double kappa_ai = 1.5;
    int postflop_max_players = 2;  // more in hand when the flop starts -> check-down leaf
    int nb[4] = {169, 50, 50, 50};
    uint64_t max_nodes = 200000000ULL;
};

class NLBuilder {
public:
    explicit NLBuilder(const NLConfig& c) : c_(c) {}

    Tree build() {
        t_ = Tree();
        t_.n = c_.n;
        for (int s = 0; s < 4; ++s) t_.nb[s] = c_.nb[s];
        t_.rake = c_.rake;
        t_.integer_chips = true;
        int64_t stk[MAXP];
        for (int i = 0; i < c_.n; ++i) stk[i] = c_.stack;
        BetState s = BetState::start(c_.n, stk, c_.sb, c_.bb);
        rec(s);
        t_.assign_rows();
        return std::move(t_);
    }

    // The abstract action set at a live state (also used by the server to
    // interpret real states the same way the tree did).
    static int menu(const NLConfig& c, const BetState& s, Action* out) {
        int i = s.actor();
        int nA = 0;
        int64_t coc = s.to_call(i);
        if (coc > 0) out[nA++] = {A_FOLD, 0.0f, 0};
        out[nA++] = {A_CALL, 0.0f, coc};
        if (!s.can_raise(i)) return nA;
        int in_hand = s.in_hand();
        bool mw = in_hand >= 3;
        const auto& M = (mw && !c.mw_menus[s.street].empty()) ? c.mw_menus[s.street] : c.menus[s.street];
        int cap = mw ? c.mw_max_raises[s.street] : c.max_raises[s.street];
        int64_t mn, mx;
        s.raise_bounds(i, mn, mx);
        int64_t mb = s.max_bet();
        double pac = (double)(s.pot + coc);
        int64_t total_stack = s.committed[i] + s.stacks[i];
        int first_raise = nA;
        if (s.raises < cap && !M.empty()) {
            const auto& lvl = M[std::min<size_t>((size_t)s.raises, M.size() - 1)];
            for (double f : lvl) {
                double to = (double)mb + f * pac;
                int64_t chips = (int64_t)std::floor(to - (double)s.bets[i] + 1e-9);
                if (chips < mn) chips = mn;
                if (chips >= mx) continue;  // that is the all-in
                if ((double)(s.committed[i] + chips) >= c.kappa_bet * (double)total_stack) continue;
                bool dup = false;
                for (int k = first_raise; k < nA; ++k) dup |= out[k].chips == chips;
                if (dup) continue;
                double real_f = ((double)(s.bets[i] + chips - mb)) / pac;
                if (nA >= 15) break;
                out[nA++] = {A_RAISE, (float)real_f, chips};
            }
        }
        bool add_ai = false;
        {
            // All-in stays available past the raise cap (it is a separate
            // rule from the menu lists), subject to allin_rule.
            if (c.allin_rule == 0) add_ai = true;
            else if (c.allin_rule == 1) {
                double thr_to = (double)mb + c.kappa_ai * pac;
                add_ai = (double)(s.bets[i] + mx) <= thr_to;
            }
        }
        if (add_ai) {
            double real_f = ((double)(s.bets[i] + mx - mb)) / pac;
            out[nA++] = {A_ALLIN, (float)real_f, mx};
        }
        // raises sorted ascending by chips (already ascending if menus are)
        for (int a = first_raise + 1; a < nA; ++a)
            for (int b = a; b > first_raise && out[b - 1].chips > out[b].chips; --b)
                std::swap(out[b - 1], out[b]);
        return nA;
    }

    static void apply(BetState& s, const Action& a) {
        if (a.kind == A_FOLD) s.fold();
        else if (a.kind == A_CALL) s.check_call();
        else s.bet_raise(a.chips);
    }

private:
    uint32_t rec(const BetState& s) {
        if (t_.nodes.size() >= c_.max_nodes)
            throw std::runtime_error("tree exceeds max_nodes; coarsen the menus");
        uint32_t id = (uint32_t)t_.nodes.size();
        t_.nodes.emplace_back();
        int in_hand = s.in_hand();
        bool leaf_mw = !s.done && s.street >= 1 && in_hand > c_.postflop_max_players;
        if (s.done || leaf_mw) {
            Terminal tm;
            for (int i = 0; i < c_.n; ++i) {
                tm.committed[i] = s.committed[i];
                if (s.folded[i]) tm.folded |= 1u << i;
            }
            tm.showdown = s.done ? s.showdown : true;
            tm.checkdown = leaf_mw;
            if (leaf_mw) t_.checkdown_terms++;
            // A showdown always runs the board out; a fold ends on the street reached.
            tm.saw_flop = tm.showdown || s.street >= 1;
            if (!tm.showdown) fill_fold_payoff(t_, tm);
            Node& nd = t_.nodes[id];
            nd.actor = -1;
            nd.street = (uint8_t)std::min(s.street, 3);
            nd.in_hand = (uint8_t)in_hand;
            nd.term = (uint32_t)t_.terms.size();
            t_.terms.push_back(tm);
            return id;
        }
        Action acts[16];
        int nA = menu(c_, s, acts);
        uint32_t c0 = (uint32_t)t_.child.size();
        for (int a = 0; a < nA; ++a) {
            t_.child.push_back(0);
            t_.act.push_back(acts[a]);
        }
        {
            Node& nd = t_.nodes[id];
            nd.actor = (int8_t)s.actor();
            nd.street = (uint8_t)s.street;
            nd.nA = (uint8_t)nA;
            nd.in_hand = (uint8_t)in_hand;
            nd.child0 = c0;
            nd.actor_commit = s.committed[s.actor()];
        }
        for (int a = 0; a < nA; ++a) {
            BetState ns = s;
            apply(ns, acts[a]);
            uint32_t ch = rec(ns);
            t_.child[c0 + a] = ch;
        }
        return id;
    }

    NLConfig c_;
    Tree t_;
};

// ============================================================================
// Fixed-limit builder (validation games). No stacks; antes; bet sizes per street.
// ============================================================================
struct LimitConfig {
    int n = 2;
    int streets = 1;
    int64_t ante = 1;
    int64_t bet[4] = {1, 1, 1, 1};
    int max_raises[4] = {1, 1, 1, 1};
    int nb[4] = {1, 1, 1, 1};
};

class LimitBuilder {
public:
    explicit LimitBuilder(const LimitConfig& c) : c_(c) {}
    Tree build() {
        t_ = Tree();
        t_.n = c_.n;
        for (int s = 0; s < 4; ++s) t_.nb[s] = c_.nb[s];
        t_.integer_chips = false;
        St s;
        s.n = c_.n;
        for (int i = 0; i < c_.n; ++i) s.committed[i] = c_.ante;
        s.begin(0);
        rec(s);
        t_.assign_rows();
        return std::move(t_);
    }

private:
    struct St {
        int n = 0, street = 0, raises = 0, qlen = 0;
        int64_t committed[MAXP] = {0}, bets[MAXP] = {0};
        uint32_t folded = 0;
        int8_t queue[2 * MAXP] = {0};
        bool done = false, showdown = false;
        int in_hand() const { return n - popc32(folded); }
        int64_t max_bet() const { int64_t m = 0; for (int i = 0; i < n; ++i) m = std::max(m, bets[i]); return m; }
        void begin(int opener) {
            qlen = 0; raises = 0;
            for (int k = 0; k < n; ++k) { int s = (opener + k) % n; if (!((folded >> s) & 1)) queue[qlen++] = (int8_t)s; }
        }
        void pop() { for (int k = 1; k < qlen; ++k) queue[k - 1] = queue[k]; --qlen; }
    };

    void after(St& s, const LimitConfig& c) {
        if (s.in_hand() <= 1) { s.done = true; s.showdown = false; return; }
        if (s.qlen) return;
        for (int i = 0; i < s.n; ++i) s.bets[i] = 0;
        if (s.street + 1 >= c.streets) { s.done = true; s.showdown = true; return; }
        ++s.street;
        s.begin(0);
    }

    uint32_t rec(const St& s) {
        uint32_t id = (uint32_t)t_.nodes.size();
        t_.nodes.emplace_back();
        if (s.done) {
            Terminal tm;
            for (int i = 0; i < c_.n; ++i) tm.committed[i] = s.committed[i];
            tm.folded = s.folded;
            tm.showdown = s.showdown;
            tm.saw_flop = true;
            if (!tm.showdown) fill_fold_payoff(t_, tm);
            Node& nd = t_.nodes[id];
            nd.actor = -1;
            nd.street = (uint8_t)s.street;
            nd.term = (uint32_t)t_.terms.size();
            t_.terms.push_back(tm);
            return id;
        }
        int i = s.queue[0];
        int64_t coc = s.max_bet() - s.bets[i];
        Action acts[3];
        int nA = 0;
        if (coc > 0) acts[nA++] = {A_FOLD, 0.0f, 0};
        acts[nA++] = {A_CALL, 0.0f, coc};
        if (s.raises < c_.max_raises[s.street]) acts[nA++] = {A_RAISE, 0.0f, coc + c_.bet[s.street]};
        uint32_t c0 = (uint32_t)t_.child.size();
        for (int a = 0; a < nA; ++a) { t_.child.push_back(0); t_.act.push_back(acts[a]); }
        {
            Node& nd = t_.nodes[id];
            nd.actor = (int8_t)i;
            nd.street = (uint8_t)s.street;
            nd.nA = (uint8_t)nA;
            nd.in_hand = (uint8_t)s.in_hand();
            nd.child0 = c0;
            nd.actor_commit = s.committed[i];
        }
        for (int a = 0; a < nA; ++a) {
            St ns = s;
            ns.pop();
            if (acts[a].kind == A_FOLD) {
                ns.folded |= 1u << i;
            } else if (acts[a].kind == A_CALL) {
                ns.bets[i] += coc; ns.committed[i] += coc;
            } else {
                ns.bets[i] += acts[a].chips; ns.committed[i] += acts[a].chips;
                ++ns.raises;
                ns.qlen = 0;
                for (int k = 1; k < c_.n; ++k) {
                    int q = (i + k) % c_.n;
                    if (!((ns.folded >> q) & 1)) ns.queue[ns.qlen++] = (int8_t)q;
                }
            }
            after(ns, c_);
            uint32_t ch = rec(ns);
            t_.child[c0 + a] = ch;
        }
        return id;
    }

    LimitConfig c_;
    Tree t_;
};

}  // namespace donk
