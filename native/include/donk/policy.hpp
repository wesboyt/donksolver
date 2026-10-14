// policy.hpp -- frozen policies and the batch action server.
//
// A policy file is the abstract tree plus the normalised average strategy of
// one solve (one stack class). A Bundle holds several classes and one card
// abstraction and answers "what does the solver do here" for REAL game
// states, which may be off the abstract tree:
//
//  1. class      nearest stack depth (log scale) to the actor's effective stack
//  2. replay     walk the real action history through the class tree; each real
//                raise is mapped to a neighbouring abstract size with the
//                pseudo-harmonic mapping (Ganzfried & Sandholm, IJCAI 2013);
//                the mapping's coin is a hash of information the ACTING seat
//                has (stacks, public history, its own cards), so a hand's
//                interpretation is a fixed function of the state
//  3. act        bucket the actor's cards, read the strategy row, sample with
//                the caller's uniform, and convert the abstract action back to
//                chips in the real state (a raise committing at least
//                commit_frac of the maximum becomes the all-in)
//
// Every row reports a status so the caller can fall back to another policy
// where the abstraction has nothing to say, and log how often that happens.
#pragma once
#include <cmath>
#include <cstdio>
#include <memory>
#include <string>
#include <vector>

#include "abstraction.hpp"
#include "tree.hpp"

namespace donk {

enum { ST_OK = 0, ST_OFFTREE = 1, ST_NO_NODE = 2, ST_DESYNC = 3, ST_BAD_INPUT = 4 };
enum { K_FOLD = 0, K_CHECK = 1, K_CALL = 2, K_RAISE = 3, K_ALLIN = 4 };

struct PolicyFile {
    int n = 0;
    int64_t stack = 0, sb = 0, bb = 0;
    int nb[4] = {0, 0, 0, 0};
    uint64_t abs_hash = 0;
    uint64_t iterations = 0;
    std::string config;
    std::vector<Node> nodes;
    std::vector<uint32_t> child;
    std::vector<Action> act;
    std::vector<float> prob;
    uint64_t version = 0;  // content hash: the frozen-policy stamp

    template <class StrategyFn>
    static void write(const std::string& path, const Tree& t, int64_t stack, int64_t sb, int64_t bb,
                      uint64_t abs_hash, uint64_t iterations, const std::string& config, StrategyFn strat) {
        std::vector<float> prob(t.entries, 0.0f);
        double buf[16];
        for (uint32_t id = 0; id < t.nodes.size(); ++id) {
            const Node& nd = t.nodes[id];
            if (nd.actor < 0) continue;
            for (int b = 0; b < t.nb[nd.street]; ++b) {
                strat(id, b, buf);
                for (int a = 0; a < nd.nA; ++a) prob[nd.row + (uint64_t)b * nd.nA + a] = (float)buf[a];
            }
        }
        FILE* f = std::fopen(path.c_str(), "wb");
        if (!f) throw std::runtime_error("cannot write " + path);
        auto W = [&](const void* p, size_t k) { if (k && std::fwrite(p, 1, k, f) != k) throw std::runtime_error("write failed"); };
        W("DONKPOL1", 8);
        int32_t n = t.n;
        W(&n, 4); W(&stack, 8); W(&sb, 8); W(&bb, 8); W(t.nb, sizeof(t.nb));
        W(&abs_hash, 8); W(&iterations, 8);
        uint64_t cl = config.size();
        W(&cl, 8); W(config.data(), cl);
        uint64_t nn = t.nodes.size(), nc = t.child.size(), ne = t.entries;
        W(&nn, 8); W(t.nodes.data(), nn * sizeof(Node));
        W(&nc, 8); W(t.child.data(), nc * sizeof(uint32_t)); W(t.act.data(), nc * sizeof(Action));
        W(&ne, 8); W(prob.data(), ne * sizeof(float));
        std::fclose(f);
    }

    void load(const std::string& path) {
        FILE* f = std::fopen(path.c_str(), "rb");
        if (!f) throw std::runtime_error("cannot open " + path);
        auto R = [&](void* p, size_t k) { if (k && std::fread(p, 1, k, f) != k) { std::fclose(f); throw std::runtime_error("truncated policy file " + path); } };
        char magic[8];
        R(magic, 8);
        if (std::memcmp(magic, "DONKPOL1", 8)) { std::fclose(f); throw std::runtime_error("not a donksolver policy: " + path); }
        int32_t n32;
        R(&n32, 4); n = n32;
        R(&stack, 8); R(&sb, 8); R(&bb, 8); R(nb, sizeof(nb));
        R(&abs_hash, 8); R(&iterations, 8);
        uint64_t cl;
        R(&cl, 8);
        if (cl > (1u << 24)) { std::fclose(f); throw std::runtime_error("corrupt policy file"); }
        config.resize(cl); R(&config[0], cl);
        uint64_t nn, nc, ne;
        R(&nn, 8); nodes.resize(nn); R(nodes.data(), nn * sizeof(Node));
        R(&nc, 8); child.resize(nc); act.resize(nc);
        R(child.data(), nc * sizeof(uint32_t)); R(act.data(), nc * sizeof(Action));
        R(&ne, 8); prob.resize(ne); R(prob.data(), ne * sizeof(float));
        std::fclose(f);
        uint64_t h = fnv1a((const uint8_t*)nodes.data(), nodes.size() * sizeof(Node));
        h = fnv1a((const uint8_t*)child.data(), child.size() * 4, h);
        h = fnv1a((const uint8_t*)prob.data(), prob.size() * 4, h);
        version = fnv1a((const uint8_t*)config.data(), config.size(), h);
    }
};

struct ActRow {
    int n;
    const int64_t* stacks;  // starting stacks, n
    int64_t sb, bb;
    int hole[2];
    int board[5];
    int nboard;
    const int32_t* hseat;
    const int32_t* hfold;
    const int64_t* hchips;  // ADDITIONAL chips as recorded in u_hand (0 = check)
    int hlen;
    double u;
};

struct ActResult {
    int kind = K_FOLD;
    int64_t chips = 0;
    double prob = 0;
    int status = ST_BAD_INPUT;
    int cls = -1;
    int64_t node = -1;
    int bucket = -1;
    int offtree = 0;
};

class Bundle {
public:
    std::shared_ptr<Abstraction> abs;
    std::vector<PolicyFile> pols;
    double commit_frac = 0.9;

    void add(const std::string& path) {
        PolicyFile p;
        p.load(path);
        if (abs && p.abs_hash != abs->hash)
            throw std::runtime_error("policy " + path + " was solved with a different abstraction");
        if (!pols.empty() && (p.n != pols[0].n))
            throw std::runtime_error("all policies in a bundle must have the same player count");
        pols.push_back(std::move(p));
    }

    int choose_class(const ActRow& r, int actor) const {
        int64_t other = 0;
        for (int j = 0; j < r.n; ++j) if (j != actor) other = std::max(other, r.stacks[j]);
        double eff = (double)std::min(r.stacks[actor], other) / (double)r.bb;
        int best = 0;
        double bd = 1e300;
        for (size_t k = 0; k < pols.size(); ++k) {
            double s = (double)pols[k].stack / (double)pols[k].bb;
            double dist = std::fabs(std::log(std::max(eff, 1e-9) / s));
            if (dist < bd) { bd = dist; best = (int)k; }
        }
        return best;
    }

    static uint64_t mix(uint64_t h, uint64_t v) {
        h ^= v + 0x9E3779B97F4A7C15ULL + (h << 6) + (h >> 2);
        uint64_t x = h;
        return Rng::splitmix(x);
    }

    ActResult act(const ActRow& r) const {
        ActResult out;
        if (pols.empty() || !abs || r.n != pols[0].n || r.n > MAXP) return out;
        BetState rs = BetState::start(r.n, r.stacks, r.sb, r.bb);
        // replay first to find the actor; class choice needs it
        for (int k = 0; k < r.hlen; ++k) {
            if (rs.done || rs.actor() != r.hseat[k]) return out;
            apply_real(rs, r.hfold[k], r.hchips[k]);
        }
        if (rs.done) return out;
        int actor = rs.actor();
        int cls = choose_class(r, actor);
        const PolicyFile& P = pols[(size_t)cls];
        out.cls = cls;

        uint64_t h = 0x243F6A8885A308D3ULL;
        for (int i = 0; i < r.n; ++i) h = mix(h, (uint64_t)r.stacks[i]);
        h = mix(h, (uint64_t)actor * 131 + (uint64_t)holding_index(r.hole[0], r.hole[1]));

        BetState s = BetState::start(r.n, r.stacks, r.sb, r.bb);
        uint32_t node = 0;
        int offtree = 0;
        for (int k = 0; k <= r.hlen; ++k) {
            // Abstract seats that cannot act in the real hand (all-in there
            // because stacks differ) are advanced along their call edge.
            while (P.nodes[node].actor >= 0 && P.nodes[node].actor != (k < r.hlen ? r.hseat[k] : actor)) {
                const Node& nd = P.nodes[node];
                int sa = nd.actor;
                if (s.live(sa)) { out.status = ST_DESYNC; out.node = node; return out; }
                node = P.child[nd.child0 + call_index(P, nd)];
                ++offtree;
            }
            if (k == r.hlen) break;
            const Node& nd = P.nodes[node];
            if (nd.actor < 0) { out.status = ST_NO_NODE; out.node = node; out.offtree = offtree; return out; }
            h = mix(h, (uint64_t)r.hseat[k] * 7 + (uint64_t)r.hfold[k] + ((uint64_t)r.hchips[k] << 4));
            int ci = map_action(P, nd, s, r.hseat[k], r.hfold[k], r.hchips[k], h, offtree);
            apply_real(s, r.hfold[k], r.hchips[k]);
            node = P.child[nd.child0 + ci];
        }
        const Node& nd = P.nodes[node];
        out.node = node;
        out.offtree = offtree;
        if (nd.actor < 0) { out.status = ST_NO_NODE; return out; }
        if (nd.actor != actor || nd.street != s.street) { out.status = ST_DESYNC; return out; }
        int street = s.street;
        int need = street == 0 ? 0 : street + 2;
        if (r.nboard < need) { out.status = ST_BAD_INPUT; return out; }
        int b = abs->bucket(street, r.hole[0], r.hole[1], r.board);
        out.bucket = b;
        const float* p = &P.prob[nd.row + (uint64_t)b * nd.nA];
        double c = 0;
        int a = nd.nA - 1;
        double tot = 0;
        for (int j = 0; j < nd.nA; ++j) tot += p[j];
        if (tot <= 0) tot = 1;
        for (int j = 0; j < nd.nA; ++j) { c += p[j] / tot; if (r.u < c) { a = j; break; } }
        out.prob = p[a] / tot;
        to_real(s, actor, P.act[nd.child0 + a], out);
        out.status = offtree ? ST_OFFTREE : ST_OK;
        return out;
    }

private:
    static void apply_real(BetState& s, int fold, int64_t chips) {
        int i = s.actor();
        if (fold) { s.fold(); return; }
        if (chips <= s.to_call(i)) s.check_call();
        else s.bet_raise(chips);
    }

    static int call_index(const PolicyFile& P, const Node& nd) {
        for (int a = 0; a < nd.nA; ++a) if (P.act[nd.child0 + a].kind == A_CALL) return a;
        return 0;
    }

    static int map_action(const PolicyFile& P, const Node& nd, const BetState& s, int seat, int fold, int64_t chips,
                          uint64_t h, int& offtree) {
        int fold_i = -1, call_i = -1;
        int rz[16], nr = 0, allin_i = -1;
        for (int a = 0; a < nd.nA; ++a) {
            uint8_t k = P.act[nd.child0 + a].kind;
            if (k == A_FOLD) fold_i = a;
            else if (k == A_CALL) call_i = a;
            else { rz[nr++] = a; if (k == A_ALLIN) allin_i = a; }
        }
        int64_t coc = s.to_call(seat);
        if (fold) {
            if (fold_i >= 0) return fold_i;
            ++offtree;
            return call_i;
        }
        if (chips <= coc) return call_i;
        if (nr == 0) { ++offtree; return call_i; }
        bool real_allin = chips >= s.stacks[seat];
        if (real_allin && allin_i >= 0) return allin_i;
        int64_t mb = s.max_bet();
        double x = (double)(s.bets[seat] + chips - mb) / (double)(s.pot + coc);
        double f0 = P.act[nd.child0 + rz[0]].frac, fl = P.act[nd.child0 + rz[nr - 1]].frac;
        if (x <= f0) { if (x < f0 * 0.999) ++offtree; return rz[0]; }
        if (x >= fl) { if (x > fl * 1.001) ++offtree; return rz[nr - 1]; }
        for (int j = 0; j + 1 < nr; ++j) {
            double A = P.act[nd.child0 + rz[j]].frac, B = P.act[nd.child0 + rz[j + 1]].frac;
            if (x >= A && x <= B) {
                if (std::fabs(x - A) <= 1e-6 * (1 + A)) return rz[j];
                if (std::fabs(x - B) <= 1e-6 * (1 + B)) return rz[j + 1];
                double pA = ((B - x) * (1.0 + A)) / ((B - A) * (1.0 + x));
                uint64_t hh = h;
                double u = (double)(Rng::splitmix(hh) >> 11) * (1.0 / 9007199254740992.0);
                ++offtree;
                return u < pA ? rz[j] : rz[j + 1];
            }
        }
        return rz[nr - 1];
    }

    void to_real(const BetState& s, int i, const Action& a, ActResult& out) const {
        int64_t coc = s.to_call(i);
        bool cr = s.can_raise(i);
        auto call = [&]() {
            if (coc == 0) { out.kind = K_CHECK; out.chips = 0; }
            else if (coc >= s.stacks[i]) { out.kind = K_ALLIN; out.chips = coc; }
            else { out.kind = K_CALL; out.chips = coc; }
        };
        if (a.kind == A_FOLD) {
            if (coc == 0) { out.kind = K_CHECK; out.chips = 0; }
            else { out.kind = K_FOLD; out.chips = 0; }
            return;
        }
        if (a.kind == A_CALL || !cr) { call(); return; }
        int64_t mn, mx;
        s.raise_bounds(i, mn, mx);
        if (a.kind == A_ALLIN) { out.kind = K_ALLIN; out.chips = mx; return; }
        double to = (double)s.max_bet() + (double)a.frac * (double)(s.pot + coc);
        int64_t chips = (int64_t)std::floor(to - (double)s.bets[i] + 1e-9);
        if (chips < mn) chips = mn;
        if (chips > mx) chips = mx;
        if (mx > 0 && (double)chips >= commit_frac * (double)mx) { out.kind = K_ALLIN; out.chips = mx; return; }
        out.kind = K_RAISE;
        out.chips = chips;
    }
};

}  // namespace donk
