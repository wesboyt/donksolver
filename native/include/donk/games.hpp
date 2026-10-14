// games.hpp -- the three games the solver template is instantiated on, and the
// range deal sampler.
//
// A game supplies: tree(), sample_deal(rng, deal), bucket(deal, player, street),
// utility(deal, terminal node, player); enumerable games also supply
// enumerate(fn(deal, prob)) for exact analysis.
#pragma once
#include <functional>
#include <memory>
#include <vector>

#include "abstraction.hpp"
#include "tree.hpp"

namespace donk {

struct Deal {
    int8_t hole[MAXP][2];
    int8_t board[5];
    int16_t bucket[MAXP][4];
    uint32_t value[MAXP];
    bool river_ready;
};

inline double terminal_utility(const Tree& t, const Deal& d, const Node& nd, int i) {
    const Terminal& tm = t.terms[nd.term];
    if (!tm.showdown) return tm.fold_payoff[i];
    bool folded[MAXP];
    for (int p = 0; p < t.n; ++p) folded[p] = (tm.folded >> p) & 1;
    double out[MAXP];
    settle(t.n, tm.committed, folded, d.value, true, tm.saw_flop, t.rake, t.integer_chips, out);
    return out[i];
}

// ---------------------------------------------------------------------------
// Kuhn poker, N players: N+1 cards, ante 1, one bet of 1, one raise.
// ---------------------------------------------------------------------------
class KuhnGame {
public:
    explicit KuhnGame(int n) : n_(n) {
        LimitConfig c;
        c.n = n; c.streets = 1; c.ante = 1; c.bet[0] = 1; c.max_raises[0] = 1; c.nb[0] = n + 1;
        tree_ = LimitBuilder(c).build();
    }
    const Tree& tree() const { return tree_; }
    int n() const { return n_; }
    bool enumerable() const { return true; }
    void sample_deal(Rng& rng, Deal& d) const {
        int deck[MAXP + 1];
        for (int c = 0; c <= n_; ++c) deck[c] = c;
        for (int k = 0; k < n_; ++k) { int j = k + (int)rng.below((uint32_t)(n_ + 1 - k)); std::swap(deck[k], deck[j]); }
        fill(d, deck);
    }
    int bucket(Deal& d, int p, int s) const { return d.bucket[p][s]; }
    double utility(Deal& d, const Node& nd, int p) const { return terminal_utility(tree_, d, nd, p); }
    void enumerate(const std::function<void(Deal&, double)>& fn) const {
        std::vector<int> cards(n_), used(n_ + 1, 0);
        uint64_t total = 1;
        for (int k = 0; k < n_; ++k) total *= (uint64_t)(n_ + 1 - k);
        double p = 1.0 / (double)total;
        std::function<void(int)> rec = [&](int k) {
            if (k == n_) { Deal d; fill(d, cards.data()); fn(d, p); return; }
            for (int c = 0; c <= n_; ++c) if (!used[c]) { used[c] = 1; cards[k] = c; rec(k + 1); used[c] = 0; }
        };
        rec(0);
    }

private:
    void fill(Deal& d, const int* cards) const {
        for (int p = 0; p < n_; ++p) {
            d.hole[p][0] = (int8_t)cards[p]; d.hole[p][1] = -1;
            d.bucket[p][0] = (int16_t)cards[p];
            d.value[p] = (uint32_t)cards[p];
        }
        d.river_ready = true;
    }
    int n_;
    Tree tree_;
};

// ---------------------------------------------------------------------------
// Leduc hold'em, N players: ranks R = N+1, two suits; ante 1; bets 2 then 4;
// two raises per round. Pair with the board wins, else the higher rank.
// ---------------------------------------------------------------------------
class LeducGame {
public:
    explicit LeducGame(int n) : n_(n), R_(n + 1) {
        LimitConfig c;
        c.n = n; c.streets = 2; c.ante = 1;
        c.bet[0] = 2; c.bet[1] = 4; c.max_raises[0] = 2; c.max_raises[1] = 2;
        c.nb[0] = R_; c.nb[1] = R_ * R_;
        tree_ = LimitBuilder(c).build();
    }
    const Tree& tree() const { return tree_; }
    int n() const { return n_; }
    bool enumerable() const { return true; }
    void sample_deal(Rng& rng, Deal& d) const {
        int deck[2 * (MAXP + 1)];
        int D = 2 * R_;
        for (int c = 0; c < D; ++c) deck[c] = c;
        for (int k = 0; k <= n_; ++k) { int j = k + (int)rng.below((uint32_t)(D - k)); std::swap(deck[k], deck[j]); }
        fill(d, deck);
    }
    int bucket(Deal& d, int p, int s) const { return d.bucket[p][s]; }
    double utility(Deal& d, const Node& nd, int p) const { return terminal_utility(tree_, d, nd, p); }
    void enumerate(const std::function<void(Deal&, double)>& fn) const {
        int D = 2 * R_;
        std::vector<int> cards(n_ + 1), used(D, 0);
        uint64_t total = 1;
        for (int k = 0; k <= n_; ++k) total *= (uint64_t)(D - k);
        double p = 1.0 / (double)total;
        std::function<void(int)> rec = [&](int k) {
            if (k == n_ + 1) { Deal d; fill(d, cards.data()); fn(d, p); return; }
            for (int c = 0; c < D; ++c) if (!used[c]) { used[c] = 1; cards[k] = c; rec(k + 1); used[c] = 0; }
        };
        rec(0);
    }

private:
    void fill(Deal& d, const int* cards) const {
        int pub = cards[n_] / 2;
        d.board[0] = (int8_t)cards[n_];
        for (int p = 0; p < n_; ++p) {
            int r = cards[p] / 2;
            d.hole[p][0] = (int8_t)cards[p]; d.hole[p][1] = -1;
            d.bucket[p][0] = (int16_t)r;
            d.bucket[p][1] = (int16_t)(r * R_ + pub);
            d.value[p] = r == pub ? (uint32_t)(1000 + r) : (uint32_t)r;
        }
        d.river_ready = true;
    }
    int n_, R_;
    Tree tree_;
};

// ---------------------------------------------------------------------------
// No-limit hold'em with the card abstraction.
// ---------------------------------------------------------------------------
class NLHEGame {
public:
    NLHEGame(const NLConfig& cfg, std::shared_ptr<const Abstraction> abs) : cfg_(cfg), abs_(std::move(abs)) {
        if (!abs_) throw std::runtime_error("nlhe needs an abstraction");
        for (int s = 1; s < 4; ++s)
            if (cfg_.nb[s] != abs_->nb[s])
                throw std::runtime_error("bucket counts in the config do not match the abstraction file");
        cfg_.nb[0] = 169;
        tree_ = NLBuilder(cfg_).build();
    }
    const Tree& tree() const { return tree_; }
    const NLConfig& config() const { return cfg_; }
    const Abstraction& abstraction() const { return *abs_; }
    int n() const { return cfg_.n; }
    bool enumerable() const { return false; }
    void enumerate(const std::function<void(Deal&, double)>&) const {
        throw std::runtime_error("nlhe is not enumerable");
    }

    void sample_deal(Rng& rng, Deal& d) const {
        int deck[52];
        for (int c = 0; c < 52; ++c) deck[c] = c;
        int need = 2 * cfg_.n + 5;
        for (int k = 0; k < need; ++k) { int j = k + (int)rng.below((uint32_t)(52 - k)); std::swap(deck[k], deck[j]); }
        set_deal(d, deck);
    }

    // cards: hole[0..2n) then board[5]
    void set_deal(Deal& d, const int* cards) const {
        const Abstraction& A = *abs_;
        int n = cfg_.n;
        for (int k = 0; k < 5; ++k) d.board[k] = (int8_t)cards[2 * n + k];
        int bd[5];
        for (int k = 0; k < 5; ++k) bd[k] = cards[2 * n + k];
        uint64_t bm = 0;
        for (int k = 0; k < 5; ++k) bm |= sbit(bd[k]);
        Canon cf = canonical_board(bd, 3), ct = canonical_board(bd, 4);
        int fi = A.flop_index(cf.key), ti = A.turn_index(ct.key);
        for (int p = 0; p < n; ++p) {
            int a = cards[2 * p], b = cards[2 * p + 1];
            d.hole[p][0] = (int8_t)a; d.hole[p][1] = (int8_t)b;
            d.value[p] = eval_mask(bm | sbit(a) | sbit(b));
            d.bucket[p][0] = (int16_t)preflop_class(a, b);
            d.bucket[p][1] = (int16_t)A.flop_bucket_at(fi, holding_index(cf.map_card(a), cf.map_card(b)));
            d.bucket[p][2] = (int16_t)A.turn_bucket_at(ti, holding_index(ct.map_card(a), ct.map_card(b)));
            d.bucket[p][3] = -1;
        }
        d.river_ready = false;
    }

    int bucket(Deal& d, int p, int s) const {
        if (s == 3 && !d.river_ready) river(d);
        return d.bucket[p][s];
    }
    double utility(Deal& d, const Node& nd, int p) const { return terminal_utility(tree_, d, nd, p); }

private:
    void river(Deal& d) const {
        int bd[5], hole[MAXP][2];
        for (int k = 0; k < 5; ++k) bd[k] = d.board[k];
        for (int p = 0; p < cfg_.n; ++p) { hole[p][0] = d.hole[p][0]; hole[p][1] = d.hole[p][1]; }
        double hs[MAXP];
        hs_players(bd, cfg_.n, hole, hs);
        for (int p = 0; p < cfg_.n; ++p) d.bucket[p][3] = (int16_t)abs_->river_bucket(bd, hs[p]);
        d.river_ready = true;
    }
    NLConfig cfg_;
    std::shared_ptr<const Abstraction> abs_;
    Tree tree_;
};

// ---------------------------------------------------------------------------
// Range deal sampler: draw each seat's holding from its weighted range and
// restart the WHOLE deal on any card collision, which gives the law
// prod_i min(w_i, 1) * 1[disjoint] / Z. broken=1 is the audit's negative
// control: it redraws only the colliding seat, which does NOT produce that law.
// ---------------------------------------------------------------------------
inline void sample_ranges(int nseats, const std::vector<std::vector<std::pair<int, int>>>& combos,
                          const std::vector<std::vector<double>>& w, uint64_t dead_board, Rng& rng,
                          int* out_idx, bool broken) {
    for (;;) {
        uint64_t used = dead_board;
        bool ok = true;
        for (int i = 0; i < nseats; ++i) {
            const auto& C = combos[(size_t)i];
            int k;
            for (int tries = 0;; ++tries) {
                for (;;) {
                    k = (int)rng.below((uint32_t)C.size());
                    double wk = w[(size_t)i][(size_t)k];
                    if (wk >= 1.0 || rng.uniform() <= wk) break;
                }
                uint64_t m = (1ULL << C[(size_t)k].first) | (1ULL << C[(size_t)k].second);
                if (m & used) {
                    // broken: redraw this seat only (wrong law); give up after
                    // many tries, since the seat may have no disjoint combo left
                    if (broken && tries < 1000) continue;
                    ok = false;
                }
                break;
            }
            if (!ok) break;
            used |= (1ULL << C[(size_t)k].first) | (1ULL << C[(size_t)k].second);
            out_idx[i] = k;
        }
        if (ok) return;
    }
}

}  // namespace donk
