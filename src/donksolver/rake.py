"""The default cash-game rake schedule.

A share of the pot (5%), capped by a table that depends on the stake and on how
many players were dealt in, with no flop no drop. `Game.rake_cap` overrides the
cap and `Game.rake_pct = 0` disables rake. The table is the one pokerfast
ships; tests/audit_payoff.py fails if the two disagree.
"""

# (big blind in chips) -> (2-player cap, 3-4 player cap, 5+ player cap)
_RAKE_TABLE = (
    (2, 25, 50, 75),
    (10, 75, 100, 150),
    (20, 75, 125, 200),
    (50, 100, 250, 300),
    (100, 100, 250, 350),
    (200, 100, 250, 350),
    (500, 125, 300, 400),
    (1000, 150, 300, 450),
    (2000, 150, 300, 550),
)
RAKE_PCT = 0.05


def rake_cap(big_blind, n_players_dealt, stake_ref_bb=None):
    """Cap in chips at blind `big_blind`.

    Stakes between two rows snap down to the lower row. With `stake_ref_bb`,
    the cap of the stake whose big blind is `stake_ref_bb` is converted to big
    blinds and charged in the chip units actually dealt.
    """
    bb = max(int(big_blind), 1)
    lookup = stake_ref_bb if stake_ref_bb else bb
    row = _RAKE_TABLE[0]
    for cand in _RAKE_TABLE:
        if cand[0] <= lookup:
            row = cand
        else:
            break
    cap = row[1] if n_players_dealt <= 2 else row[2] if n_players_dealt <= 4 else row[3]
    if stake_ref_bb:
        cap = int(round(cap * (bb / float(stake_ref_bb))))
    return cap
