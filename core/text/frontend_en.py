"""English text front-end (normalisation + grapheme-to-phoneme).

Added for this assignment. The reference repo's `preprocess_english` looked
words up in the LibriSpeech lexicon and fell back to g2p_en; it did not expand
times/dates, read AM/PM as the word "am", and always took the *first* lexicon
entry, so heteronyms such as "conduct" (noun vs verb) were read the same way.

Pipeline:
  1. normalise()      times, dates, ordinals, numbers, acronyms -> words
  2. repair_splits()  re-join a word broken by a stray space ("rhyth m")
                       when the joined form is a dictionary word and the
                       left fragment is not
  3. to_phones()      POS-aware heteronym choice (g2p_en homograph table),
                       then LibriSpeech lexicon, then g2p_en neural fallback.
Punctuation becomes the short-pause token "sp", matching MFA output.
"""
import re

import inflect

_inflect = inflect.engine()

MONTHS = [
    "january", "february", "march", "april", "may", "june", "july",
    "august", "september", "october", "november", "december",
]
LETTERS = {
    "A": "EY1", "B": "B IY1", "C": "S IY1", "D": "D IY1", "E": "IY1",
    "F": "EH1 F", "G": "JH IY1", "H": "EY1 CH", "I": "AY1", "J": "JH EY1",
    "K": "K EY1", "L": "EH1 L", "M": "EH1 M", "N": "EH1 N", "O": "OW1",
    "P": "P IY1", "Q": "K Y UW1", "R": "AA1 R", "S": "EH1 S", "T": "T IY1",
    "U": "Y UW1", "V": "V IY1", "W": "D AH1 B AH0 L Y UW0", "X": "EH1 K S",
    "Y": "W AY1", "Z": "Z IY1",
}


def _num(n):
    return _inflect.number_to_words(n, andword="").replace("-", " ").replace(",", "")


def _year(n):
    n = int(n)
    if 2000 <= n <= 2009:
        return _num(n)
    if 1100 <= n <= 2099 and n % 100 != 0:
        return _num(n // 100) + " " + _num(n % 100)
    if 1100 <= n <= 2099:
        return _num(n // 100) + " hundred"
    return _num(n)


def _ordinal(n):
    w = _inflect.number_to_words(_inflect.ordinal(int(n)), andword="")
    return w.replace("-", " ").replace(",", "")


def _time(m):
    h, mnt, ampm = int(m.group(1)), int(m.group(2)), m.group(3)
    words = _num(h)
    if mnt == 0:
        words += " o'clock" if not ampm else ""
    elif mnt < 10:
        words += " oh " + _num(mnt)
    else:
        words += " " + _num(mnt)
    if ampm:
        words += " " + ampm.replace(".", "").upper()
    return words


_month_re = "|".join(MONTHS)


def normalise(text):
    t = text
    t = t.replace("’", "'").replace("‘", "'")
    t = re.sub(r"\b(\d{1,2}):(\d{2})\s*([AaPp]\.?[Mm]\.?)?", _time, t)
    # "12 November 2026" / "November 12, 2026"
    t = re.sub(
        r"\b(\d{1,2})(?:st|nd|rd|th)?\s+(" + _month_re + r")\b,?\s*(\d{4})?",
        lambda m: _ordinal(m.group(1)) + " " + m.group(2)
        + ((" " + _year(m.group(3))) if m.group(3) else ""),
        t, flags=re.I,
    )
    t = re.sub(
        r"\b(" + _month_re + r")\s+(\d{1,2})(?:st|nd|rd|th)?\b,?\s*(\d{4})?",
        lambda m: m.group(1) + " " + _ordinal(m.group(2))
        + ((" " + _year(m.group(3))) if m.group(3) else ""),
        t, flags=re.I,
    )
    t = re.sub(r"\b(\d+)(st|nd|rd|th)\b", lambda m: _ordinal(m.group(1)), t)
    t = re.sub(r"\b(1[1-9]\d\d|20\d\d)\b", lambda m: _year(m.group(1)), t)
    t = re.sub(r"(\d),(\d{3})", r"\1\2", t)
    t = re.sub(r"\b\d+\.\d+\b",
               lambda m: _num(m.group(0).split(".")[0]) + " point "
               + " ".join(_num(c) for c in m.group(0).split(".")[1]), t)
    t = re.sub(r"\d+", lambda m: _num(m.group(0)), t)
    t = t.replace("&", " and ").replace("%", " percent")
    return re.sub(r"\s+", " ", t).strip()


def read_lexicon(path):
    lex = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            lex.setdefault(parts[0].lower(), []).append(parts[1:])
    return lex


_word_re = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)*|[^\sA-Za-z]")


def tokenize(text):
    return _word_re.findall(text)


def repair_splits(tokens, lexicon):
    """Join token pairs like ['rhyth', 'm'] -> ['rhythm'].

    Only fires when the left piece is NOT a word but the concatenation is.
    Returns (tokens, list_of_repairs)."""
    out, repairs, i = [], [], 0
    while i < len(tokens):
        a = tokens[i]
        if i + 1 < len(tokens):
            b = tokens[i + 1]
            joined = (a + b).lower()
            if (a.isalpha() and b.isalpha() and a.lower() not in lexicon
                    and joined in lexicon):
                out.append(a + b)
                repairs.append((a, b, a + b))
                i += 2
                continue
        out.append(a)
        i += 1
    return out, repairs


class EnglishFrontend:
    def __init__(self, lexicon_path, repair=True):
        from g2p_en import G2p

        self.lexicon = read_lexicon(lexicon_path)
        self.g2p = G2p()
        self.repair = repair

    def word_phones(self, word, pos):
        w = word.lower()
        homo = self.g2p.homograph2features
        if w in homo:
            pron1, pron2, pos1 = homo[w]
            return pron1 if pos.startswith(pos1) else pron2, "homograph"
        if word.isupper() and len(word) <= 4 and w not in ("a", "i"):
            # acronyms such as AM, PM, TTS
            if all(c in LETTERS for c in word):
                return " ".join(LETTERS[c] for c in word).split(), "acronym"
        if w in self.lexicon:
            return self.lexicon[w][0], "lexicon"
        if w in self.g2p.cmu:
            return self.g2p.cmu[w][0], "cmudict"
        return [p for p in self.g2p.predict(w)], "g2p"

    def __call__(self, text, verbose=False):
        import nltk

        norm = normalise(text)
        toks = tokenize(norm)
        repairs = []
        if self.repair:
            toks, repairs = repair_splits(toks, self.lexicon)
        tagged = nltk.pos_tag(toks)
        phones, trace = [], []
        for tok, pos in tagged:
            if re.fullmatch(r"[A-Za-z']+", tok):
                ph, src = self.word_phones(tok, pos)
                phones += ph
                trace.append((tok, pos, src, " ".join(ph)))
            elif tok in ",;:.!?-()\"":
                if phones and phones[-1] != "sp":
                    phones.append("sp")
        while phones and phones[-1] == "sp":
            phones.pop()
        info = {"normalised": norm, "tokens": toks, "repairs": repairs, "trace": trace}
        if verbose:
            print("normalised:", norm)
            for r in repairs:
                print("repaired split:", r)
            for t in trace:
                print("  %-30s %-5s %-9s %s" % t)
        return phones, info
