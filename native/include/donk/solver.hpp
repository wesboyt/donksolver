// solver.hpp -- tabular Monte Carlo CFR over an explicit abstract tree.
//
//   traversal   cs  : chance sampling, full width for everyone, prune sigma<=eps
//               es  : external sampling (opponents' actions sampled), which is
//                     what makes 6-max trees affordable per iteration
//   weighting   linear     : regrets AND average weighted by t (Linear CFR,
//                            Brown & Sandholm 2019) -- the default
//               linear_avg : uniform regrets, t-weighted average (no proven
//                            bound carries over; kept to compare)
//               vanilla    : uniform both
//   average     weighted : own reach only, opponents full width -> consistent
//               sampled  : own action sampled from sigma (Pluribus's update);
//                          same expectation, far cheaper at 6-max
//               Both run with probability p_avg per iteration.
//   root EV     one seat per deal, from the first traversal, so the samples
//               are independent across deals.
//
// Threads share the tables. Writes take a striped spinlock; reads are
// unlocked. hogwild=1 drops the write locks.
#pragma once
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "games.hpp"

namespace donk {

enum { ALGO_ES = 0, ALGO_CS = 1 };
enum { W_LINEAR = 0, W_LINEAR_AVG = 1, W_VANILLA = 2 };
enum { AVG_WEIGHTED = 0, AVG_SAMPLED = 1, AVG_OFF = 2 };

struct SolverConfig {
    int algo = ALGO_ES;
    int weighting = W_LINEAR;
    bool rm_plus = false;
    double prune_eps = 1e-7;
    int avg_mode = AVG_SAMPLED;
    double p_avg = 0.05;
    int avg_max_street = 3;
    bool hogwild = false;
    uint64_t seed = 1;
};

struct SolveStats {
    uint64_t iterations = 0;
    double ev_sum[MAXP] = {0};
    double ev_sq[MAXP] = {0};
    uint64_t ev_n[MAXP] = {0};
    uint64_t avg_passes = 0;
    double seconds = 0;
};

// Type-erased surface the C API talks to.
class AnySolver {
public:
    virtual ~AnySolver() = default;
    virtual const Tree& tree() const = 0;
    virtual void run(uint64_t iters, int threads) = 0;
    virtual const SolveStats& stats() const = 0;
    virtual void reset_ev() = 0;
    virtual void reset_average() = 0;
    virtual int strategy(uint32_t node, int bucket, bool avg, double* out) const = 0;
    virtual bool enumerable() const = 0;
    virtual void exact(double* ev, double* br) const = 0;
    virtual void debug_regret(int player, int mode, uint64_t n, bool broken, uint64_t seed, double* out) = 0;
    virtual void shadow_enable(bool on, bool broken) = 0;
    virtual double shadow_compare(double min_mass, double* worst_mass) const = 0;
    virtual void save_state(const std::string& path) const = 0;
    virtual void load_state(const std::string& path) = 0;
    virtual size_t bytes() const = 0;
    virtual const NLHEGame* nlhe() const = 0;
};

template <class G, class Real>
class Solver final : public AnySolver {
public:
    Solver(std::shared_ptr<const G> g, const SolverConfig& cfg)
        : g_(std::move(g)), t_(g_->tree()), cfg_(cfg) {
        R_.reset(new Real[t_.entries]());
        S_.reset(new Real[t_.entries]());
        locks_.reset(new std::atomic_flag[NLOCK]);
        for (size_t i = 0; i < NLOCK; ++i) locks_[i].clear();
    }

    const Tree& tree() const override { return t_; }
    const SolveStats& stats() const override { return stats_; }
    bool enumerable() const override { return g_->enumerable(); }
    size_t bytes() const override { return (size_t)t_.entries * 2 * sizeof(Real); }
    const NLHEGame* nlhe() const override { return nlhe_ptr(g_.get()); }

    void reset_ev() override {
        for (int i = 0; i < MAXP; ++i) { stats_.ev_sum[i] = 0; stats_.ev_sq[i] = 0; stats_.ev_n[i] = 0; }
    }
    void reset_average() override { std::fill(S_.get(), S_.get() + t_.entries, (Real)0); }

    void run(uint64_t iters, int threads) override {
        if (shadow_on_ && threads != 1) throw std::runtime_error("shadow average needs threads=1");
        threads = std::max(1, threads);
        uint64_t target = iter_.load() + iters;
        std::vector<Local> locals((size_t)threads);
        auto t0 = std::chrono::steady_clock::now();
        auto work = [&](int tid) {
            Rng rng(cfg_.seed * 0x9E3779B97F4A7C15ULL + (iter_.load() + 1) * 7919ULL + (uint64_t)tid * 104729ULL);
            Deal d;
            Local& L = locals[(size_t)tid];
            int order[MAXP];
            int n = t_.n;
            for (;;) {
                uint64_t t = iter_.fetch_add(1) + 1;
                if (t > target) break;
                double wr = cfg_.weighting == W_LINEAR ? (double)t : 1.0;
                double wa = cfg_.weighting == W_VANILLA ? 1.0 : (double)t;
                g_->sample_deal(rng, d);
                for (int i = 0; i < n; ++i) order[i] = i;
                for (int i = n - 1; i > 0; --i) std::swap(order[i], order[rng.below((uint32_t)(i + 1))]);
                for (int k = 0; k < n; ++k) {
                    int i = order[k];
                    double v = cfg_.algo == ALGO_ES ? es(0, d, i, rng, wr) : cs(0, d, i, 1.0, wr, nullptr, 1.0, false);
                    if (k == 0) { L.ev[i] += v; L.sq[i] += v * v; L.n[i]++; }
                }
                if (cfg_.avg_mode != AVG_OFF && rng.uniform() < cfg_.p_avg) {
                    for (int i = 0; i < n; ++i) {
                        if (cfg_.avg_mode == AVG_WEIGHTED) avg_w(0, d, i, 1.0, wa, S_.get(), true, broken_avg_);
                        else avg_s(0, d, i, wa, rng);
                    }
                    L.passes++;
                }
                if (shadow_on_) shadow_step(wa);
                L.iters++;
            }
        };
        std::vector<std::thread> th;
        for (int k = 0; k < threads; ++k) th.emplace_back(work, k);
        for (auto& x : th) x.join();
        if (iter_.load() > target) iter_.store(target);
        for (auto& L : locals) {
            for (int i = 0; i < MAXP; ++i) { stats_.ev_sum[i] += L.ev[i]; stats_.ev_sq[i] += L.sq[i]; stats_.ev_n[i] += L.n[i]; }
            stats_.avg_passes += L.passes;
        }
        stats_.iterations = iter_.load();
        stats_.seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    }

    int strategy(uint32_t nid, int b, bool avg, double* out) const override {
        const Node& nd = t_.nodes[nid];
        if (nd.actor < 0) return 0;
        const Real* row = &R_[nd.row + (uint64_t)b * nd.nA];
        if (avg) {
            const Real* s = &S_[nd.row + (uint64_t)b * nd.nA];
            double tot = 0;
            for (int a = 0; a < nd.nA; ++a) tot += (double)s[a];
            if (tot > 0) { for (int a = 0; a < nd.nA; ++a) out[a] = (double)s[a] / tot; return nd.nA; }
        }
        rm(row, nd.nA, out);
        return nd.nA;
    }

    // ---- exact analysis (enumerable games) -----------------------------------
    // ev[i] = value of the average profile; br[i] = best-response value of seat i
    // against the others' average strategies. Deviation gain = br - ev.
    void exact(double* ev, double* br) const override {
        std::vector<Deal> deals;
        std::vector<double> prob;
        g_->enumerate([&](Deal& d, double p) { deals.push_back(d); prob.push_back(p); });
        for (int i = 0; i < t_.n; ++i) {
            std::vector<double> reach(prob);
            std::vector<double> v;
            ev[i] = sum(walk(0, i, deals, reach, false, v));
            std::vector<double> reach2(prob);
            br[i] = sum(walk(0, i, deals, reach2, true, v));
        }
    }

    // Unbiasedness audit hook: expected (mode 0, exact) or sampled-mean (mode 1)
    // chance-sampled regret increment for `player` under the CURRENT profile,
    // written to out[entries]. broken=1 drops the opponent reach factor q.
    void debug_regret(int player, int mode, uint64_t n, bool broken, uint64_t seed, double* out) override {
        std::fill(out, out + t_.entries, 0.0);
        std::vector<double> sink(t_.entries, 0.0);
        if (mode == 0) {
            g_->enumerate([&](Deal& d, double p) { cs(0, d, player, 1.0, 1.0, sink.data(), p, broken); });
        } else if (mode == 2) {
            // exact second moment E_deal[increment^2], for exact standard errors
            std::vector<double> tmp(t_.entries, 0.0);
            g_->enumerate([&](Deal& d, double p) {
                std::fill(tmp.begin(), tmp.end(), 0.0);
                cs(0, d, player, 1.0, 1.0, tmp.data(), 1.0, broken);
                for (uint64_t e = 0; e < t_.entries; ++e) sink[e] += p * tmp[e] * tmp[e];
            });
        } else {
            Rng rng(seed);
            Deal d;
            double s = 1.0 / (double)n;
            for (uint64_t k = 0; k < n; ++k) { g_->sample_deal(rng, d); cs(0, d, player, 1.0, 1.0, sink.data(), s, broken); }
        }
        std::copy(sink.begin(), sink.end(), out);
    }

    // Average audit hook: a shadow average accumulating the EXACT expected
    // average-pass increment every iteration (single thread, enumerable games).
    // broken=1 makes the REAL average pass weight by opponents' reach too.
    void shadow_enable(bool on, bool broken) override {
        shadow_on_ = on;
        broken_avg_ = broken;
        if (on) shadow_.assign(t_.entries, 0.0);
    }
    double shadow_compare(double min_mass, double* worst_mass) const override {
        double worst = 0;
        *worst_mass = 0;
        for (const Node& nd : t_.nodes) {
            if (nd.actor < 0) continue;
            for (int b = 0; b < t_.nb[nd.street]; ++b) {
                uint64_t base = nd.row + (uint64_t)b * nd.nA;
                double ts = 0, tx = 0;
                for (int a = 0; a < nd.nA; ++a) { ts += (double)S_[base + a]; tx += shadow_[base + a]; }
                if (ts <= 0 || tx <= 0) continue;
                double mass = tx;
                if (mass < min_mass) continue;
                for (int a = 0; a < nd.nA; ++a) {
                    double e = std::fabs((double)S_[base + a] / ts - shadow_[base + a] / tx);
                    if (e > worst) { worst = e; *worst_mass = mass; }
                }
            }
        }
        return worst;
    }

    // ---- checkpoint -----------------------------------------------------------
    void save_state(const std::string& path) const override {
        FILE* f = std::fopen(path.c_str(), "wb");
        if (!f) throw std::runtime_error("cannot write " + path);
        uint64_t hdr[5] = {0x3154534B4E4F44ULL /*DONKST1*/, t_.entries, sizeof(Real), iter_.load(), tree_hash()};
        std::fwrite(hdr, 8, 5, f);
        std::fwrite(&stats_, sizeof(stats_), 1, f);
        std::fwrite(R_.get(), sizeof(Real), t_.entries, f);
        std::fwrite(S_.get(), sizeof(Real), t_.entries, f);
        bool ok = std::ferror(f) == 0;
        std::fclose(f);
        if (!ok) throw std::runtime_error("write failed: " + path);
    }
    void load_state(const std::string& path) override {
        FILE* f = std::fopen(path.c_str(), "rb");
        if (!f) throw std::runtime_error("cannot open " + path);
        uint64_t hdr[5];
        if (std::fread(hdr, 8, 5, f) != 5 || hdr[0] != 0x3154534B4E4F44ULL) { std::fclose(f); throw std::runtime_error("not a solver state file"); }
        if (hdr[1] != t_.entries || hdr[2] != sizeof(Real) || hdr[4] != tree_hash()) { std::fclose(f); throw std::runtime_error("state file does not match this tree/precision"); }
        bool ok = std::fread(&stats_, sizeof(stats_), 1, f) == 1 &&
                  std::fread(R_.get(), sizeof(Real), t_.entries, f) == t_.entries &&
                  std::fread(S_.get(), sizeof(Real), t_.entries, f) == t_.entries;
        std::fclose(f);
        if (!ok) throw std::runtime_error("truncated state file");
        iter_.store(hdr[3]);
    }

    uint64_t tree_hash() const {
        uint64_t h = fnv1a((const uint8_t*)t_.nodes.data(), t_.nodes.size() * sizeof(Node));
        h = fnv1a((const uint8_t*)t_.child.data(), t_.child.size() * sizeof(uint32_t), h);
        return fnv1a((const uint8_t*)&t_.entries, 8, h);
    }

    const Real* avg_table() const { return S_.get(); }
    const Real* regret_table() const { return R_.get(); }

private:
    static constexpr size_t NLOCK = 1 << 16;
    struct Local {
        double ev[MAXP] = {0}, sq[MAXP] = {0};
        uint64_t n[MAXP] = {0};
        uint64_t passes = 0, iters = 0;
    };

    static const NLHEGame* nlhe_ptr(const NLHEGame* g) { return g; }
    static const NLHEGame* nlhe_ptr(const void*) { return nullptr; }

    static void rm(const Real* r, int nA, double* out) {
        double tot = 0;
        for (int a = 0; a < nA; ++a) { double v = (double)r[a]; out[a] = v > 0 ? v : 0; tot += out[a]; }
        if (tot > 0) { for (int a = 0; a < nA; ++a) out[a] /= tot; }
        else { double u = 1.0 / nA; for (int a = 0; a < nA; ++a) out[a] = u; }
    }

    static int sample(const double* p, int nA, Rng& rng) {
        double u = rng.uniform(), c = 0;
        for (int a = 0; a < nA - 1; ++a) { c += p[a]; if (u < c) return a; }
        return nA - 1;
    }

    void lock(uint64_t base) {
        if (cfg_.hogwild) return;
        auto& l = locks_[(base * 0x9E3779B97F4A7C15ULL) >> 48];
        while (l.test_and_set(std::memory_order_acquire)) {}
    }
    void unlock(uint64_t base) {
        if (cfg_.hogwild) return;
        locks_[(base * 0x9E3779B97F4A7C15ULL) >> 48].clear(std::memory_order_release);
    }

    void add_regret(uint64_t base, int nA, const double* inc) {
        lock(base);
        Real* r = &R_[base];
        if (cfg_.rm_plus) {
            for (int a = 0; a < nA; ++a) { Real v = r[a] + (Real)inc[a]; r[a] = v > 0 ? v : (Real)0; }
        } else {
            for (int a = 0; a < nA; ++a) r[a] += (Real)inc[a];
        }
        unlock(base);
    }

    // External sampling. Returns the traverser's sampled value.
    double es(uint32_t nid, Deal& d, int i, Rng& rng, double wr) {
        const Node& nd = t_.nodes[nid];
        if (nd.actor < 0) return g_->utility(d, nd, i);
        int j = nd.actor;
        int b = g_->bucket(d, j, nd.street);
        uint64_t base = nd.row + (uint64_t)b * nd.nA;
        double sig[16];
        rm(&R_[base], nd.nA, sig);
        if (j != i) return es(t_.child[nd.child0 + sample(sig, nd.nA, rng)], d, i, rng, wr);
        double v[16], V = 0, inc[16];
        for (int a = 0; a < nd.nA; ++a) {
            v[a] = t_.act[nd.child0 + a].kind == A_FOLD ? -(double)nd.actor_commit
                                                         : es(t_.child[nd.child0 + a], d, i, rng, wr);
            V += sig[a] * v[a];
        }
        for (int a = 0; a < nd.nA; ++a) inc[a] = wr * (v[a] - V);
        add_regret(base, nd.nA, inc);
        return V;
    }

    // Chance sampling. sink != nullptr: write increments*scale
    // into sink instead of the table (audit). broken: drop q at opponent nodes.
    double cs(uint32_t nid, Deal& d, int i, double q, double wr, double* sink, double scale, bool broken) {
        const Node& nd = t_.nodes[nid];
        if (nd.actor < 0) return q * g_->utility(d, nd, i);
        int j = nd.actor;
        int b = g_->bucket(d, j, nd.street);
        uint64_t base = nd.row + (uint64_t)b * nd.nA;
        double sig[16];
        rm(&R_[base], nd.nA, sig);
        if (j != i) {
            double V = 0;
            for (int a = 0; a < nd.nA; ++a) {
                if (sig[a] <= cfg_.prune_eps && !sink) continue;
                double qa = broken ? q : q * sig[a];
                double va = cs(t_.child[nd.child0 + a], d, i, qa, wr, sink, scale, broken);
                V += broken ? sig[a] * va : va;
            }
            return V;
        }
        double Q[16], V = 0, inc[16];
        for (int a = 0; a < nd.nA; ++a) {
            Q[a] = t_.act[nd.child0 + a].kind == A_FOLD ? -q * (double)nd.actor_commit
                                                         : cs(t_.child[nd.child0 + a], d, i, q, wr, sink, scale, broken);
            V += sig[a] * Q[a];
        }
        if (sink) {
            for (int a = 0; a < nd.nA; ++a) sink[base + a] += scale * (Q[a] - V);
        } else {
            for (int a = 0; a < nd.nA; ++a) inc[a] = wr * (Q[a] - V);
            add_regret(base, nd.nA, inc);
        }
        return V;
    }

    // Average pass: own reach omega, opponents full width.
    // `locked` false for the single-threaded shadow. `broken` also multiplies
    // by the opponents' action probabilities (inconsistent; audit control).
    template <class T>
    void avg_w(uint32_t nid, Deal& d, int i, double omega, double wa, T* S, bool locked, bool broken) {
        const Node& nd = t_.nodes[nid];
        if (nd.actor < 0 || nd.street > cfg_.avg_max_street || omega < 1e-7) return;
        int j = nd.actor;
        int b = g_->bucket(d, j, nd.street);
        uint64_t base = nd.row + (uint64_t)b * nd.nA;
        double sig[16];
        rm(&R_[base], nd.nA, sig);
        if (j == i) {
            if (locked) lock(base);
            for (int a = 0; a < nd.nA; ++a) S[base + a] += (T)(wa * omega * sig[a]);
            if (locked) unlock(base);
            for (int a = 0; a < nd.nA; ++a)
                if (sig[a] > 0 && t_.act[nd.child0 + a].kind != A_FOLD)
                    avg_w(t_.child[nd.child0 + a], d, i, omega * sig[a], wa, S, locked, broken);
        } else {
            for (int a = 0; a < nd.nA; ++a)
                avg_w(t_.child[nd.child0 + a], d, i, broken ? omega * sig[a] : omega, wa, S, locked, broken);
        }
    }

    // Sampled own action, opponents full width (Pluribus UPDATE-STRATEGY).
    void avg_s(uint32_t nid, Deal& d, int i, double wa, Rng& rng) {
        const Node& nd = t_.nodes[nid];
        if (nd.actor < 0 || nd.street > cfg_.avg_max_street) return;
        int j = nd.actor;
        int b = g_->bucket(d, j, nd.street);
        uint64_t base = nd.row + (uint64_t)b * nd.nA;
        if (j == i) {
            double sig[16];
            rm(&R_[base], nd.nA, sig);
            int a = sample(sig, nd.nA, rng);
            lock(base);
            S_[base + a] += (Real)wa;
            unlock(base);
            if (t_.act[nd.child0 + a].kind != A_FOLD) avg_s(t_.child[nd.child0 + a], d, i, wa, rng);
        } else {
            for (int a = 0; a < nd.nA; ++a) avg_s(t_.child[nd.child0 + a], d, i, wa, rng);
        }
    }

    void shadow_step(double wa) {
        // The real pass already ran (or not) this iteration; the shadow adds the
        // exact expected increment, independent of the Bernoulli draw.
        g_->enumerate([&](Deal& d, double p) {
            for (int i = 0; i < t_.n; ++i) avg_w(0, d, i, 1.0, wa * p, shadow_.data(), false, false);
        });
    }

    static double sum(const std::vector<double>& v) { double s = 0; for (double x : v) s += x; return s; }

    // value vector over deals for player i (chance and opponent reach folded in)
    const std::vector<double>& walk(uint32_t nid, int i, std::vector<Deal>& deals, std::vector<double>& reach,
                                    bool best, std::vector<double>& out) const {
        size_t D = deals.size();
        const Node& nd = t_.nodes[nid];
        out.assign(D, 0.0);
        if (nd.actor < 0) {
            for (size_t k = 0; k < D; ++k)
                if (reach[k] != 0) out[k] = reach[k] * g_->utility(deals[k], nd, i);
            return out;
        }
        int j = nd.actor;
        double sig[16];
        if (j != i) {
            std::vector<double> cr(D), cv;
            for (int a = 0; a < nd.nA; ++a) {
                for (size_t k = 0; k < D; ++k) {
                    if (reach[k] == 0) { cr[k] = 0; continue; }
                    int b = g_->bucket(deals[k], j, nd.street);
                    strategy(nid, b, true, sig);
                    cr[k] = reach[k] * sig[a];
                }
                walk(t_.child[nd.child0 + a], i, deals, cr, best, cv);
                for (size_t k = 0; k < D; ++k) out[k] += cv[k];
            }
            return out;
        }
        std::vector<std::vector<double>> vals((size_t)nd.nA);
        for (int a = 0; a < nd.nA; ++a) {
            std::vector<double> r2(reach);
            walk(t_.child[nd.child0 + a], i, deals, r2, best, vals[(size_t)a]);
        }
        int B = t_.nb[nd.street];
        if (best) {
            std::vector<double> tot((size_t)B * nd.nA, 0.0);
            for (size_t k = 0; k < D; ++k) {
                int b = g_->bucket(deals[k], i, nd.street);
                for (int a = 0; a < nd.nA; ++a) tot[(size_t)b * nd.nA + a] += vals[(size_t)a][k];
            }
            std::vector<int> arg((size_t)B, 0);
            for (int b = 0; b < B; ++b)
                for (int a = 1; a < nd.nA; ++a)
                    if (tot[(size_t)b * nd.nA + a] > tot[(size_t)b * nd.nA + arg[(size_t)b]]) arg[(size_t)b] = a;
            for (size_t k = 0; k < D; ++k) out[k] = vals[(size_t)arg[(size_t)g_->bucket(deals[k], i, nd.street)]][k];
        } else {
            for (size_t k = 0; k < D; ++k) {
                int b = g_->bucket(deals[k], i, nd.street);
                strategy(nid, b, true, sig);
                for (int a = 0; a < nd.nA; ++a) out[k] += sig[a] * vals[(size_t)a][k];
            }
        }
        return out;
    }

    std::shared_ptr<const G> g_;
    const Tree& t_;
    SolverConfig cfg_;
    std::unique_ptr<Real[]> R_, S_;
    std::unique_ptr<std::atomic_flag[]> locks_;
    std::atomic<uint64_t> iter_{0};
    SolveStats stats_;
    bool shadow_on_ = false, broken_avg_ = false;
    std::vector<double> shadow_;
};

}  // namespace donk
