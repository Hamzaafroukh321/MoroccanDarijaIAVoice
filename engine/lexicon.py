# EVERY LIST BELOW NEEDS HUMAN REVIEW BY A NATIVE DARIJA SPEAKER.
# The coding agent must not add, remove, or "correct" entries.

NEEDS_HUMAN_REVIEW = True

# Speaker is hesitating — they have NOT finished. Extend the wait.
HESITATION_MARKERS = [
    "اااه", "eeeh", "eh", "ahh",
    "زعما", "zaama", "z3ma",
    "يعني", "y3ni", "yaani",
    "mmm", "emm", "hmm",
    "euh", "ben", "alors",
    "[[TODO_DARIJA: add more hesitation fillers]]",
]

# Sentence ends on a connector — more is coming. Extend the wait.
CONTINUATION_MARKERS = [
    "و", "ou", "w", "wa",
    "زيد", "zid",
    "مع", "m3a", "m3ak",
    "ديال", "dyal", "d",
    "et", "avec", "plus",
    "[[TODO_DARIJA: add more trailing connectors]]",
]

# Hints passed to the router LLM so it recognises overwrites.
CORRECTION_MARKERS = [
    "لا لا", "la la", "lala",
    "ماشي", "machi",
    "بدل", "bdel",
    "سمح ليا", "sme7 liya", "smehli",
    "non", "pardon",
    "[[TODO_DARIJA: add more correction phrases]]",
]

AFFIRM_MARKERS = [
    "ايه", "iyeh", "ah", "wah",
    "واخا", "wakha",
    "صافي", "safi",
    "oui", "ok", "d'accord",
    "[[TODO_DARIJA: add more yes words]]",
]

NEGATE_MARKERS = [
    "لا", "la",
    "ماشي", "machi",
    "non",
    "[[TODO_DARIJA: add more no words]]",
]
