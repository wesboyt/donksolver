// capi.cpp -- the C ABI. Everything the Python package and the CLI need goes
// through here, so the library can be driven from any language and shipped as
// one shared object with no Python headers involved (works unchanged on
// free-threaded CPython).
//
// Errors: functions return 0/NULL on failure and ds_last_error() says why
// (thread-local).
#include <cstdio>
#include <cstdlib>
#include <map>
#include <sstream>

#include "donk/policy.hpp"
#include "donk/solver.hpp"

#if defined(_WIN32)
#define DONK_API extern "C" __declspec(dllexport)
#else
#define DONK_API extern "C" __attribute__((visibility("default")))
#endif

#ifndef DONK_VERSION
#define DONK_VERSION "0.0.0"
#endif

using namespace donk;

static thread_local std::string g_err;
static void set_err(const std::string& e) { g_err = e; }

#define GUARD_BEGIN try {
#define GUARD_END(ret) } catch (const std::exception& e) { set_err(e.what()); return ret; } catch (...) { set_err("unknown error"); return ret; }

// ---- config text: "key=value" lines ------------------------------------------
using Cfg = std::map<std::string, std::string>;

static Cfg parse_cfg(const char* text) {
    Cfg c;
    std::istringstream in(text ? text : "");
    std::string line;
    while (std::getline(in, line)) {
        size_t hash = line.find('#');
        if (hash != std::string::npos) line = line.substr(0, hash);
        size_t eq = line.find('=');
        if (eq == std::string::npos) continue;
        auto trim = [](std::string s) {
            size_t a = s.find_first_not_of(" \t\r"), b = s.find_last_not_of(" \t\r");
            return a == std::string::npos ? std::string() : s.substr(a, b - a + 1);
        };
        c[trim(line.substr(0, eq))] = trim(line.substr(eq + 1));
    }
    return c;
}
static std::string get(const Cfg& c, const std::string& k, const std::string& d) {
    auto it = c.find(k);
    return it == c.end() ? d : it->second;
}
static double getd(const Cfg& c, const std::string& k, double d) {
    auto it = c.find(k);
    return it == c.end() ? d : std::strtod(it->second.c_str(), nullptr);
}
static int64_t geti(const Cfg& c, const std::string& k, int64_t d) {
    auto it = c.find(k);
    return it == c.end() ? d : std::strtoll(it->second.c_str(), nullptr, 10);
}
static std::vector<std::string> split(const std::string& s, char sep) {
    std::vector<std::string> out;
    std::string cur;
    for (char ch : s) {
        if (ch == sep) { out.push_back(cur); cur.clear(); } else cur += ch;
    }
    if (!cur.empty() || !out.empty()) out.push_back(cur);
    return out;
}
static void get_int4(const Cfg& c, const std::string& k, int* out) {
    auto it = c.find(k);
    if (it == c.end()) return;
    auto v = split(it->second, ',');
    if (v.size() != 4) throw std::runtime_error(k + " needs 4 comma-separated values");
    for (int i = 0; i < 4; ++i) out[i] = std::atoi(v[(size_t)i].c_str());
}
// "0.6,1.0|0.8|1.0" -> levels
static std::vector<std::vector<double>> get_menu(const Cfg& c, const std::string& k) {
    std::vector<std::vector<double>> m;
    auto it = c.find(k);
    if (it == c.end() || it->second.empty()) return m;
    for (auto& lvl : split(it->second, '|')) {
        std::vector<double> fr;
        for (auto& x : split(lvl, ',')) if (!x.empty()) fr.push_back(std::strtod(x.c_str(), nullptr));
        std::sort(fr.begin(), fr.end());
        m.push_back(fr);
    }
    return m;
}

static NLConfig nl_config(const Cfg& c) {
    NLConfig n;
    n.n = (int)geti(c, "players", 6);
    if (n.n < 2 || n.n > MAXP) throw std::runtime_error("players must be 2..6");
    n.stack = geti(c, "stack", 20000);
    n.sb = geti(c, "sb", 100);
    n.bb = geti(c, "bb", 200);
    n.rake.pct = getd(c, "rake_pct", 0.0);
    n.rake.cap = geti(c, "rake_cap", 0);
    n.rake.no_flop_no_drop = geti(c, "no_flop_no_drop", 1) != 0;
    for (int s = 0; s < 4; ++s) {
        n.menus[s] = get_menu(c, "menu." + std::to_string(s));
        n.mw_menus[s] = get_menu(c, "mw_menu." + std::to_string(s));
    }
    get_int4(c, "max_raises", n.max_raises);
    get_int4(c, "mw_max_raises", n.mw_max_raises);
    n.kappa_bet = getd(c, "kappa_bet", 0.9);
    std::string ar = get(c, "allin", "always");
    n.allin_rule = ar == "always" ? 0 : ar == "threshold" ? 1 : ar == "never" ? 2 : -1;
    if (n.allin_rule < 0) throw std::runtime_error("allin must be always|threshold|never");
    n.kappa_ai = getd(c, "kappa_ai", 1.5);
    n.postflop_max_players = (int)geti(c, "postflop_max_players", 2);
    get_int4(c, "buckets", n.nb);
    n.max_nodes = (uint64_t)geti(c, "max_nodes", 200000000LL);
    return n;
}

static SolverConfig solver_config(const Cfg& c) {
    SolverConfig s;
    std::string a = get(c, "algo", "es");
    s.algo = a == "es" ? ALGO_ES : a == "cs" ? ALGO_CS : -1;
    std::string w = get(c, "weighting", "linear");
    s.weighting = w == "linear" ? W_LINEAR : w == "linear_avg" ? W_LINEAR_AVG : w == "vanilla" ? W_VANILLA : -1;
    std::string m = get(c, "avg_mode", "sampled");
    s.avg_mode = m == "weighted" ? AVG_WEIGHTED : m == "sampled" ? AVG_SAMPLED : m == "off" ? AVG_OFF : -1;
    if (s.algo < 0 || s.weighting < 0 || s.avg_mode < 0) throw std::runtime_error("bad algo/weighting/avg_mode");
    s.rm_plus = geti(c, "rm_plus", 0) != 0;
    s.prune_eps = getd(c, "prune_eps", 1e-7);
    s.p_avg = getd(c, "p_avg", 0.05);
    s.avg_max_street = (int)geti(c, "avg_max_street", 3);
    s.hogwild = geti(c, "hogwild", 0) != 0;
    s.seed = (uint64_t)geti(c, "seed", 1);
    return s;
}

struct SolverHandle {
    std::string config;
    std::unique_ptr<AnySolver> s;
    std::shared_ptr<const Abstraction> abs;
    int64_t stack = 0, sb = 0, bb = 0;
};

template <class G>
static std::unique_ptr<AnySolver> make_solver(std::shared_ptr<const G> g, const SolverConfig& sc, bool dbl) {
    if (dbl) return std::unique_ptr<AnySolver>(new Solver<G, double>(g, sc));
    return std::unique_ptr<AnySolver>(new Solver<G, float>(g, sc));
}

static std::string tree_info(const Tree& t, const Cfg& c) {
    std::ostringstream o;
    o << "players=" << t.n << "\n";
    o << "nodes=" << t.nodes.size() << "\n";
    o << "terminals=" << t.terms.size() << "\n";
    o << "checkdown_terminals=" << t.checkdown_terms << "\n";
    o << "entries=" << t.entries << "\n";
    for (int s = 0; s < 4; ++s) {
        o << "decision_nodes." << s << "=" << t.decision_nodes[s] << "\n";
        o << "rows." << s << "=" << t.rows_per_street[s] << "\n";
        o << "buckets." << s << "=" << t.nb[s] << "\n";
    }
    std::string prec = get(c, "precision", "double");
    size_t w = prec == "float" ? 4 : 8;
    o << "table_bytes=" << (t.entries * 2 * w) << "\n";
    return o.str();
}

// size a tree without allocating tables
static Tree build_tree(const Cfg& c) {
    std::string game = get(c, "game", "nlhe");
    int n = (int)geti(c, "players", 2);
    if (game == "kuhn") return KuhnGame(n).tree();
    if (game == "leduc") return LeducGame(n).tree();
    if (game == "nlhe") {
        NLConfig nc = nl_config(c);
        return NLBuilder(nc).build();
    }
    throw std::runtime_error("unknown game " + game);
}

// ============================================================================
DONK_API const char* ds_version() { return DONK_VERSION; }
DONK_API const char* ds_last_error() { return g_err.c_str(); }

DONK_API uint32_t ds_eval(const int32_t* cards, int n) {
    int c[7];
    for (int i = 0; i < n && i < 7; ++i) c[i] = cards[i];
    return eval_cards(c, n);
}
DONK_API uint32_t ds_eval_reference(const int32_t* cards, int n) {
    int c[7];
    for (int i = 0; i < n && i < 7; ++i) c[i] = cards[i];
    return eval_reference(c, n);
}
DONK_API void ds_eval_batch(const int32_t* cards, int64_t rows, int ncards, uint32_t* out) {
    for (int64_t r = 0; r < rows; ++r) {
        uint64_t m = 0;
        for (int k = 0; k < ncards; ++k) m |= sbit(cards[r * ncards + k]);
        out[r] = eval_mask(m);
    }
}

DONK_API void ds_settle(int n, const int64_t* committed, const uint8_t* folded, const uint32_t* value, int showdown,
                        int saw_flop, double rake_pct, int64_t rake_cap, int no_flop_no_drop, int integer_chips,
                        double* out) {
    bool f[MAXP];
    for (int i = 0; i < n; ++i) f[i] = folded[i] != 0;
    RakeRule r;
    r.pct = rake_pct; r.cap = rake_cap; r.no_flop_no_drop = no_flop_no_drop != 0;
    settle(n, committed, f, value, showdown != 0, saw_flop != 0, r, integer_chips != 0, out);
}

// Replay a real action history under the betting rules. Returns the number
// of actions applied (hlen on success, -1 if an action came from a seat that
// was not to act). out_state: [done, showdown, street, actor, pot, to_call,
// min_bet, max_bet, can_raise] and committed[n] after it.
DONK_API int ds_replay(int n, const int64_t* stacks, int64_t sb, int64_t bb, const int32_t* hseat,
                       const int32_t* hfold, const int64_t* hchips, int hlen, int64_t* out_state,
                       int64_t* out_committed) {
    BetState s = BetState::start(n, stacks, sb, bb);
    int k = 0;
    for (; k < hlen; ++k) {
        if (s.done || s.actor() != hseat[k]) return -1;
        int i = s.actor();
        if (hfold[k]) s.fold();
        else if (hchips[k] <= s.to_call(i)) s.check_call();
        else s.bet_raise(hchips[k]);
    }
    int i = s.actor();
    int64_t mn = 0, mx = 0;
    bool cr = !s.done && i >= 0 && s.can_raise(i);
    if (cr) s.raise_bounds(i, mn, mx);
    out_state[0] = s.done; out_state[1] = s.showdown; out_state[2] = s.street; out_state[3] = s.done ? -1 : i;
    out_state[4] = s.pot; out_state[5] = (!s.done && i >= 0) ? s.to_call(i) : 0;
    out_state[6] = mn; out_state[7] = mx; out_state[8] = cr;
    for (int j = 0; j < n; ++j) out_committed[j] = s.committed[j];
    return k;
}

// ---- abstraction ---------------------------------------------------------------
DONK_API void* ds_abs_build(int nbf, int nbt, int nbr, int threads, uint64_t river_boards, uint64_t seed,
                            void (*progress)(const char*, double)) {
    GUARD_BEGIN
    auto* a = new Abstraction();
    std::function<void(const char*, double)> pf;
    if (progress) pf = progress;
    a->build(nbf, nbt, nbr, threads, river_boards, seed, pf);
    return a;
    GUARD_END(nullptr)
}
DONK_API void* ds_abs_load(const char* path) {
    GUARD_BEGIN
    auto* a = new Abstraction();
    a->load(path);
    return a;
    GUARD_END(nullptr)
}
DONK_API int ds_abs_save(void* h, const char* path) {
    GUARD_BEGIN
    static_cast<Abstraction*>(h)->save(path);
    return 1;
    GUARD_END(0)
}
DONK_API void ds_abs_free(void* h) { delete static_cast<Abstraction*>(h); }
DONK_API uint64_t ds_abs_hash(void* h) { return static_cast<Abstraction*>(h)->hash; }
DONK_API int ds_abs_buckets(void* h, int32_t* out4) {
    auto* a = static_cast<Abstraction*>(h);
    for (int i = 0; i < 4; ++i) out4[i] = a->nb[i];
    return 1;
}
DONK_API int ds_abs_bucket(void* h, int street, const int32_t* hole, const int32_t* board) {
    GUARD_BEGIN
    int bd[5] = {0, 0, 0, 0, 0};
    int nbd = street == 0 ? 0 : street + 2;
    for (int i = 0; i < nbd; ++i) bd[i] = board[i];
    return static_cast<Abstraction*>(h)->bucket(street, hole[0], hole[1], bd);
    GUARD_END(-1)
}
DONK_API int ds_abs_debug_values(void* h, int n, const int32_t* board, float* vals, uint8_t* tab) {
    GUARD_BEGIN
    int bd[5] = {0};
    for (int i = 0; i < n; ++i) bd[i] = board[i];
    return static_cast<Abstraction*>(h)->debug_values(n, bd, vals, tab);
    GUARD_END(-2)
}
DONK_API double ds_hand_strength(const int32_t* hole, const int32_t* board5) {
    int bd[5], ho[1][2] = {{hole[0], hole[1]}};
    for (int i = 0; i < 5; ++i) bd[i] = board5[i];
    double hs;
    hs_players(bd, 1, ho, &hs);
    return hs;
}

// ---- solver --------------------------------------------------------------------
// Shared abstraction cache so several solvers (stack classes) reuse one load.
static std::mutex g_abs_mu;
static std::map<std::string, std::weak_ptr<const Abstraction>> g_abs_cache;
static std::shared_ptr<const Abstraction> load_abs_cached(const std::string& path) {
    std::lock_guard<std::mutex> lk(g_abs_mu);
    auto it = g_abs_cache.find(path);
    if (it != g_abs_cache.end())
        if (auto sp = it->second.lock()) return sp;
    auto a = std::make_shared<Abstraction>();
    a->load(path);
    std::shared_ptr<const Abstraction> c = a;
    g_abs_cache[path] = c;
    return c;
}

DONK_API int ds_tree_info(const char* config, char* buf, int cap) {
    GUARD_BEGIN
    Cfg c = parse_cfg(config);
    Tree t = build_tree(c);
    std::string s = tree_info(t, c);
    if ((int)s.size() + 1 > cap) throw std::runtime_error("buffer too small");
    std::memcpy(buf, s.c_str(), s.size() + 1);
    return (int)s.size();
    GUARD_END(-1)
}

DONK_API void* ds_solver_create(const char* config) {
    GUARD_BEGIN
    Cfg c = parse_cfg(config);
    auto h = std::unique_ptr<SolverHandle>(new SolverHandle());
    h->config = config;
    SolverConfig sc = solver_config(c);
    bool dbl = get(c, "precision", "double") != "float";
    std::string game = get(c, "game", "nlhe");
    int n = (int)geti(c, "players", 2);
    double limit_gb = getd(c, "memory_limit_gb", 12.0);
    auto check_mem = [&](const Tree& t) {
        double gb = (double)t.entries * 2 * (dbl ? 8 : 4) / 1073741824.0;
        if (gb > limit_gb) {
            std::ostringstream o;
            o << "tables need " << gb << " GB > memory_limit_gb=" << limit_gb;
            throw std::runtime_error(o.str());
        }
    };
    if (game == "kuhn") {
        auto g = std::make_shared<const KuhnGame>(n);
        check_mem(g->tree());
        h->s = make_solver(g, sc, dbl);
    } else if (game == "leduc") {
        auto g = std::make_shared<const LeducGame>(n);
        check_mem(g->tree());
        h->s = make_solver(g, sc, dbl);
    } else if (game == "nlhe") {
        NLConfig nc = nl_config(c);
        std::string ap = get(c, "abstraction", "");
        if (ap.empty()) throw std::runtime_error("nlhe needs abstraction=<path>");
        h->abs = load_abs_cached(ap);
        auto g = std::make_shared<const NLHEGame>(nc, h->abs);
        check_mem(g->tree());
        h->s = make_solver(g, sc, dbl);
        h->stack = nc.stack; h->sb = nc.sb; h->bb = nc.bb;
    } else {
        throw std::runtime_error("unknown game " + game);
    }
    return h.release();
    GUARD_END(nullptr)
}
DONK_API void ds_solver_free(void* h) { delete static_cast<SolverHandle*>(h); }

DONK_API int ds_solver_info(void* hp, char* buf, int cap) {
    GUARD_BEGIN
    auto* h = static_cast<SolverHandle*>(hp);
    std::string s = tree_info(h->s->tree(), parse_cfg(h->config.c_str()));
    const SolveStats& st = h->s->stats();
    std::ostringstream o;
    o << s << "iterations=" << st.iterations << "\nseconds=" << st.seconds << "\navg_passes=" << st.avg_passes << "\n";
    s = o.str();
    if ((int)s.size() + 1 > cap) throw std::runtime_error("buffer too small");
    std::memcpy(buf, s.c_str(), s.size() + 1);
    return (int)s.size();
    GUARD_END(-1)
}

DONK_API int ds_solver_run(void* hp, uint64_t iters, int threads) {
    GUARD_BEGIN
    static_cast<SolverHandle*>(hp)->s->run(iters, threads);
    return 1;
    GUARD_END(0)
}

DONK_API int ds_solver_root_ev(void* hp, double* sum, double* sq, uint64_t* n) {
    auto* h = static_cast<SolverHandle*>(hp);
    const SolveStats& st = h->s->stats();
    for (int i = 0; i < h->s->tree().n; ++i) { sum[i] = st.ev_sum[i]; sq[i] = st.ev_sq[i]; n[i] = st.ev_n[i]; }
    return h->s->tree().n;
}
DONK_API void ds_solver_reset_ev(void* hp) { static_cast<SolverHandle*>(hp)->s->reset_ev(); }
DONK_API void ds_solver_reset_average(void* hp) { static_cast<SolverHandle*>(hp)->s->reset_average(); }
DONK_API uint64_t ds_solver_entries(void* hp) { return static_cast<SolverHandle*>(hp)->s->tree().entries; }
DONK_API uint64_t ds_solver_iterations(void* hp) { return static_cast<SolverHandle*>(hp)->s->stats().iterations; }

DONK_API int ds_solver_exact(void* hp, double* ev, double* br) {
    GUARD_BEGIN
    auto* h = static_cast<SolverHandle*>(hp);
    if (!h->s->enumerable()) throw std::runtime_error("exact analysis needs an enumerable game");
    h->s->exact(ev, br);
    return h->s->tree().n;
    GUARD_END(0)
}

// node info: [actor, street, nA, term, child0, in_hand]; children/kinds/fracs/chips sized nA
DONK_API int ds_solver_node(void* hp, uint32_t node, int32_t* info, uint32_t* children, uint8_t* kinds, float* fracs,
                            int64_t* chips) {
    GUARD_BEGIN
    const Tree& t = static_cast<SolverHandle*>(hp)->s->tree();
    if (node >= t.nodes.size()) throw std::runtime_error("node out of range");
    const Node& nd = t.nodes[node];
    info[0] = nd.actor; info[1] = nd.street; info[2] = nd.nA; info[3] = (int32_t)nd.term; info[4] = (int32_t)nd.child0;
    info[5] = nd.in_hand;
    for (int a = 0; a < nd.nA; ++a) {
        if (children) children[a] = t.child[nd.child0 + a];
        if (kinds) kinds[a] = t.act[nd.child0 + a].kind;
        if (fracs) fracs[a] = t.act[nd.child0 + a].frac;
        if (chips) chips[a] = t.act[nd.child0 + a].chips;
    }
    return nd.nA;
    GUARD_END(-1)
}
DONK_API int ds_solver_row(void* hp, uint32_t node, int bucket, int32_t* out /*[row, nA]*/) {
    const Tree& t = static_cast<SolverHandle*>(hp)->s->tree();
    const Node& nd = t.nodes[node];
    out[0] = (int32_t)(nd.row + (uint64_t)bucket * nd.nA);
    out[1] = nd.nA;
    return 1;
}

DONK_API int ds_solver_strategy(void* hp, uint32_t node, int bucket, int avg, double* out) {
    GUARD_BEGIN
    return static_cast<SolverHandle*>(hp)->s->strategy(node, bucket, avg != 0, out);
    GUARD_END(-1)
}

DONK_API int ds_solver_debug_regret(void* hp, int player, int mode, uint64_t n, int broken, uint64_t seed, double* out) {
    GUARD_BEGIN
    auto* h = static_cast<SolverHandle*>(hp);
    if (mode == 0 && !h->s->enumerable()) throw std::runtime_error("exact mode needs an enumerable game");
    h->s->debug_regret(player, mode, n, broken != 0, seed, out);
    return 1;
    GUARD_END(0)
}
DONK_API int ds_solver_shadow(void* hp, int on, int broken) {
    GUARD_BEGIN
    auto* h = static_cast<SolverHandle*>(hp);
    if (on && !h->s->enumerable()) throw std::runtime_error("shadow average needs an enumerable game");
    h->s->shadow_enable(on != 0, broken != 0);
    return 1;
    GUARD_END(0)
}
DONK_API double ds_solver_shadow_compare(void* hp, double min_mass, double* worst_mass) {
    return static_cast<SolverHandle*>(hp)->s->shadow_compare(min_mass, worst_mass);
}

DONK_API int ds_solver_save_state(void* hp, const char* path) {
    GUARD_BEGIN
    static_cast<SolverHandle*>(hp)->s->save_state(path);
    return 1;
    GUARD_END(0)
}
DONK_API int ds_solver_load_state(void* hp, const char* path) {
    GUARD_BEGIN
    static_cast<SolverHandle*>(hp)->s->load_state(path);
    return 1;
    GUARD_END(0)
}

DONK_API int ds_solver_save_policy(void* hp, const char* path) {
    GUARD_BEGIN
    auto* h = static_cast<SolverHandle*>(hp);
    if (!h->s->nlhe()) throw std::runtime_error("policies are exported for nlhe only");
    AnySolver* s = h->s.get();
    PolicyFile::write(path, s->tree(), h->stack, h->sb, h->bb, h->abs->hash, s->stats().iterations, h->config,
                      [&](uint32_t node, int b, double* out) { s->strategy(node, b, true, out); });
    return 1;
    GUARD_END(0)
}

// ---- range deal sampler ----------------------------------------------------------
// combos: per seat, ncombos[i] pairs of card ids (flattened); weights likewise.
DONK_API int ds_sample_ranges(int nseats, const int32_t* ncombos, const int32_t* combos, const double* weights,
                              const int32_t* board, int nboard, uint64_t ndeals, uint64_t seed, int broken,
                              int32_t* out) {
    GUARD_BEGIN
    std::vector<std::vector<std::pair<int, int>>> C((size_t)nseats);
    std::vector<std::vector<double>> W((size_t)nseats);
    size_t off = 0;
    for (int i = 0; i < nseats; ++i) {
        for (int k = 0; k < ncombos[i]; ++k) {
            C[(size_t)i].push_back({combos[2 * (off + (size_t)k)], combos[2 * (off + (size_t)k) + 1]});
            W[(size_t)i].push_back(weights[off + (size_t)k]);
        }
        off += (size_t)ncombos[i];
    }
    uint64_t dead = 0;
    for (int k = 0; k < nboard; ++k) dead |= 1ULL << board[k];
    Rng rng(seed);
    int idx[MAXP];
    for (uint64_t d = 0; d < ndeals; ++d) {
        sample_ranges(nseats, C, W, dead, rng, idx, broken != 0);
        for (int i = 0; i < nseats; ++i) out[d * (uint64_t)nseats + (uint64_t)i] = idx[i];
    }
    return 1;
    GUARD_END(0)
}

// ---- serving ------------------------------------------------------------------------
DONK_API void* ds_bundle_load(const char* abstraction_path, const char* policy_paths_nl, double commit_frac) {
    GUARD_BEGIN
    auto b = std::unique_ptr<Bundle>(new Bundle());
    auto a = std::make_shared<Abstraction>();
    a->load(abstraction_path);
    b->abs = a;
    b->commit_frac = commit_frac;
    for (auto& p : split(policy_paths_nl ? policy_paths_nl : "", '\n'))
        if (!p.empty()) b->add(p);
    if (b->pols.empty()) throw std::runtime_error("bundle has no policies");
    return b.release();
    GUARD_END(nullptr)
}
DONK_API void ds_bundle_free(void* h) { delete static_cast<Bundle*>(h); }

// per class: [players, stack, sb, bb, iterations, version, nodes, entries]
DONK_API int ds_bundle_info(void* h, int cls, int64_t* out) {
    auto* b = static_cast<Bundle*>(h);
    if (cls < 0 || cls >= (int)b->pols.size()) return (int)b->pols.size();
    const PolicyFile& p = b->pols[(size_t)cls];
    out[0] = p.n; out[1] = p.stack; out[2] = p.sb; out[3] = p.bb; out[4] = (int64_t)p.iterations;
    out[5] = (int64_t)p.version; out[6] = (int64_t)p.nodes.size(); out[7] = (int64_t)p.prob.size();
    return (int)b->pols.size();
}

// Row layout: meta[r*4] = {n, nboard, hist_begin, hist_len}; stacks[r*6];
// blinds[r*2]; cards[r*7] = hole0, hole1, board0..4; u[r].
// Outputs: kind, chips, prob, status, cls, node, bucket, offtree.
DONK_API int ds_bundle_act(void* h, int64_t rows, const int32_t* meta, const int64_t* stacks, const int64_t* blinds,
                           const int32_t* cards, const int32_t* hseat, const int32_t* hfold, const int64_t* hchips,
                           const double* u, int32_t* o_kind, int64_t* o_chips, double* o_prob, int32_t* o_status,
                           int32_t* o_cls, int64_t* o_node, int32_t* o_bucket, int32_t* o_offtree) {
    GUARD_BEGIN
    auto* b = static_cast<Bundle*>(h);
    for (int64_t r = 0; r < rows; ++r) {
        ActRow row;
        row.n = meta[r * 4];
        row.nboard = meta[r * 4 + 1];
        int hb = meta[r * 4 + 2];
        row.hlen = meta[r * 4 + 3];
        row.stacks = &stacks[r * 6];
        row.sb = blinds[r * 2];
        row.bb = blinds[r * 2 + 1];
        row.hole[0] = cards[r * 7];
        row.hole[1] = cards[r * 7 + 1];
        for (int k = 0; k < 5; ++k) row.board[k] = cards[r * 7 + 2 + k];
        row.hseat = hseat + hb;
        row.hfold = hfold + hb;
        row.hchips = hchips + hb;
        row.u = u[r];
        ActResult res;
        try {
            res = b->act(row);
        } catch (const std::exception&) {
            res.status = ST_BAD_INPUT;
        }
        o_kind[r] = res.kind; o_chips[r] = res.chips; o_prob[r] = res.prob; o_status[r] = res.status;
        o_cls[r] = res.cls; o_node[r] = res.node; o_bucket[r] = res.bucket; o_offtree[r] = res.offtree;
    }
    return 1;
    GUARD_END(0)
}

// strategy row of a bundle class at (node, bucket); returns nA, kinds/fracs too
DONK_API int ds_bundle_strategy(void* h, int cls, int64_t node, int bucket, float* probs, uint8_t* kinds, float* fracs) {
    GUARD_BEGIN
    auto* b = static_cast<Bundle*>(h);
    const PolicyFile& p = b->pols.at((size_t)cls);
    const Node& nd = p.nodes.at((size_t)node);
    if (nd.actor < 0) return 0;
    for (int a = 0; a < nd.nA; ++a) {
        probs[a] = p.prob[nd.row + (uint64_t)bucket * nd.nA + a];
        kinds[a] = p.act[nd.child0 + a].kind;
        fracs[a] = p.act[nd.child0 + a].frac;
    }
    return nd.nA;
    GUARD_END(-1)
}
