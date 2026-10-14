"""Card ids shared with the native core: id = rank*4 + suit, rank '23456789TJQKA', suit 'cdhs'."""
RANKS = "23456789TJQKA"
SUITS = "cdhs"
DECK = tuple(r + s for r in RANKS for s in SUITS)
_ID = {c: i for i, c in enumerate(DECK)}


def card_id(card: str) -> int:
    return _ID[card[0].upper() + card[1].lower()]


def card_str(cid: int) -> str:
    return DECK[cid]


def parse_cards(s: str):
    """'AhKd' or 'Ah Kd' or ['Ah','Kd'] -> [ids]."""
    if not isinstance(s, str):
        return [card_id(c) for c in s]
    s = s.replace(" ", "")
    return [card_id(s[i:i + 2]) for i in range(0, len(s), 2)]
