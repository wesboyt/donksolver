"""Solve configurations and presets.

Everything the native core reads is a flat `key=value` text; these dataclasses
produce it, and the same text is embedded in every policy file so a frozen
policy always says exactly how it was made.
"""
from dataclasses import dataclass, field, asdict
from typing import List, Optional

from . import rake as _rake


@dataclass
class Game:
    """The abstract no-limit game: bet-size menus on top of the exact betting rules.

    menus[street][level] are fractions of the POT AFTER CALLING; the last level
    repeats. mw_menus apply when 3+ players are in the hand (empty = menus).
    """
    players: int = 6
    stack_bb: float = 100.0
    sb: int = 100
    bb: int = 200
    rake_pct: float = 0.05
    rake_cap: Optional[int] = None          # None: the schedule in rake.py for the dealt stake
    rake_stake_bb: Optional[int] = None     # with rake_cap None: charge this stake's cap, in big blinds
    no_flop_no_drop: bool = True
    menus: List[List[List[float]]] = field(default_factory=lambda: [[[0.6]], [[0.5]], [[0.66]], [[0.75]]])
    mw_menus: List[List[List[float]]] = field(default_factory=lambda: [[], [], [], []])
    max_raises: List[int] = field(default_factory=lambda: [3, 2, 2, 2])
    mw_max_raises: List[int] = field(default_factory=lambda: [3, 1, 1, 1])
    kappa_bet: float = 0.9          # a bet committing >= this share of the stack is left to the all-in
    allin: str = "always"           # always | threshold | never
    kappa_ai: float = 1.5
    postflop_max_players: int = 2   # more see the flop -> check-down leaf (not served)
    buckets: List[int] = field(default_factory=lambda: [169, 50, 50, 50])

    def text(self):
        cap = self.rake_cap
        if cap is None:
            cap = _rake.rake_cap(self.bb, self.players, self.rake_stake_bb)
        lines = [
            "game=nlhe",
            "players=%d" % self.players,
            "stack=%d" % int(round(self.stack_bb * self.bb)),
            "sb=%d" % self.sb, "bb=%d" % self.bb,
            "rake_pct=%r" % self.rake_pct, "rake_cap=%d" % cap,
            "no_flop_no_drop=%d" % int(self.no_flop_no_drop),
            "max_raises=%s" % ",".join(map(str, self.max_raises)),
            "mw_max_raises=%s" % ",".join(map(str, self.mw_max_raises)),
            "kappa_bet=%r" % self.kappa_bet, "allin=%s" % self.allin, "kappa_ai=%r" % self.kappa_ai,
            "postflop_max_players=%d" % self.postflop_max_players,
            "buckets=%s" % ",".join(map(str, self.buckets)),
        ]
        for s in range(4):
            lines.append("menu.%d=%s" % (s, _menu(self.menus[s])))
            if self.mw_menus[s]:
                lines.append("mw_menu.%d=%s" % (s, _menu(self.mw_menus[s])))
        return "\n".join(lines) + "\n"


@dataclass
class Solve:
    algo: str = "es"              # es (external sampling) | cs (chance sampling)
    weighting: str = "linear"     # linear | linear_avg | vanilla
    rm_plus: bool = False
    prune_eps: float = 1e-7
    avg_mode: str = "sampled"     # sampled | weighted | off
    p_avg: float = 0.02
    avg_max_street: int = 3
    precision: str = "double"     # double | float (float loses increments past ~1e7 visits/row)
    hogwild: bool = False
    seed: int = 1
    memory_limit_gb: float = 12.0

    def text(self):
        d = asdict(self)
        d["rm_plus"] = int(self.rm_plus)
        d["hogwild"] = int(self.hogwild)
        return "".join("%s=%s\n" % (k, v) for k, v in d.items())


def _menu(levels):
    return "|".join(",".join(repr(float(f)) for f in lvl) for lvl in levels)


def toy(game: str, players: int = 2, **solve):
    """Config text for the validation games (kuhn | leduc)."""
    return "game=%s\nplayers=%d\n" % (game, players) + Solve(**solve).text()


# ---------------------------------------------------------------------------
# Presets. Sizes are pot-after-call fractions: preflop 0.6 at the open is a
# 2.5bb raise; 1.0 facing it is a ~3.6x three-bet.
# ---------------------------------------------------------------------------
def preset(name: str, stack_bb: float = 100.0, abstraction_buckets=(50, 50, 50)) -> Game:
    b = [169] + list(abstraction_buckets)
    if name == "hu":
        return Game(players=2, stack_bb=stack_bb, buckets=b,
                    menus=[[[0.6, 1.2], [1.0], [0.8]], [[0.33, 0.75], [1.0]], [[0.5, 1.0], [1.0]], [[0.5, 1.0], [1.0]]],
                    max_raises=[4, 3, 3, 3])
    if name == "6max-lite":
        # The budget blueprint: one open size, one 3-bet size, all-in;
        # heads-up flops played with one bet size and one raise.
        return Game(players=6, stack_bb=stack_bb, buckets=b,
                    menus=[[[0.6], [1.0], [0.9]], [[0.5]], [[0.66]], [[0.75]]],
                    max_raises=[3, 2, 2, 2], mw_max_raises=[3, 1, 1, 1],
                    postflop_max_players=2)
    if name == "6max":
        return Game(players=6, stack_bb=stack_bb, buckets=b,
                    menus=[[[0.6], [0.9], [0.8]], [[0.33, 0.75], [0.8]], [[0.5, 1.0], [0.8]], [[0.5, 1.0], [0.8]]],
                    max_raises=[4, 2, 2, 2], mw_max_raises=[3, 1, 1, 1],
                    postflop_max_players=2)
    if name == "tiny":
        # For tests: heads-up, short stack, one size everywhere.
        return Game(players=2, stack_bb=stack_bb, buckets=b,
                    menus=[[[1.0]], [[0.75]], [[0.75]], [[0.75]]], max_raises=[2, 1, 1, 1])
    raise ValueError("unknown preset %r" % name)
