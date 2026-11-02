"""Telugu text front-end: normalisation + rule-based grapheme-to-phoneme.

Added for this assignment; the reference repo has no Telugu support and MFA
ships no Telugu acoustic model or dictionary. Telugu script is close to
phonemic, so a rule-based G2P over aksharas is reliable:

  * consonant + vowel sign  -> C V
  * consonant + virama (్)  -> C        (conjuncts / geminates: k k, n d ...)
  * bare consonant          -> C a      (inherent vowel; Telugu keeps the
                                          word-final 'a', unlike Hindi)
  * anusvara (ం)            -> nasal homorganic with the next stop
                               (ng / ny / nn / n / m), 'm' word-finally
  * visarga (ః) -> h,  chandrabindu (ఁ) -> dropped,  ృ -> r u
The same G2P builds the MFA pronunciation dictionary for training and the
phone sequence at inference, so the two always agree.
"""
import re
import unicodedata

VOWELS = {
    "అ": "a", "ఆ": "aa", "ఇ": "i", "ఈ": "ii", "ఉ": "u", "ఊ": "uu",
    "ఋ": "r u", "ౠ": "r uu", "ఌ": "l u", "ౡ": "l uu",
    "ఎ": "e", "ఏ": "ee", "ఐ": "ai", "ఒ": "o", "ఓ": "oo", "ఔ": "au",
}
MATRAS = {
    "ా": "aa", "ి": "i", "ీ": "ii", "ు": "u", "ూ": "uu", "ృ": "r u",
    "ౄ": "r uu", "ె": "e", "ే": "ee", "ై": "ai", "ొ": "o", "ో": "oo",
    "ౌ": "au", "ౢ": "l u", "ౣ": "l uu",
}
CONSONANTS = {
    "క": "k", "ఖ": "kh", "గ": "g", "ఘ": "gh", "ఙ": "ng",
    "చ": "c", "ఛ": "ch", "జ": "j", "ఝ": "jh", "ఞ": "ny",
    "ట": "tt", "ఠ": "tth", "డ": "dd", "ఢ": "ddh", "ణ": "nn",
    "త": "t", "థ": "th", "ద": "d", "ధ": "dh", "న": "n",
    "ప": "p", "ఫ": "ph", "బ": "b", "భ": "bh", "మ": "m",
    "య": "y", "ర": "r", "ఱ": "r", "ల": "l", "ళ": "ll", "ఴ": "zh",
    "వ": "v", "శ": "sh", "ష": "ss", "స": "s", "హ": "h",
    "ౘ": "ts", "ౙ": "dz", "ౚ": "r",
}
VIRAMA, ANUSVARA, VISARGA, CANDRABINDU = "్", "ం", "ః", "ఁ"
ARDHA = "ఀ"

# place of articulation for anusvara assimilation
_NASAL_FOR = {}
for c in "కఖగఘఙ":
    _NASAL_FOR[c] = "ng"
for c in "చఛజఝఞ":
    _NASAL_FOR[c] = "ny"
for c in "టఠడఢణ":
    _NASAL_FOR[c] = "nn"
for c in "తథదధన":
    _NASAL_FOR[c] = "n"
for c in "పఫబభమ":
    _NASAL_FOR[c] = "m"

PHONES = sorted(
    set(" ".join(list(VOWELS.values()) + list(MATRAS.values())
                 + list(CONSONANTS.values()) + ["ng", "ny", "nn", "n", "m", "h"]).split())
)

# ---------------------------------------------------------------- numbers
_ONES = ["సున్నా", "ఒకటి", "రెండు", "మూడు", "నాలుగు", "ఐదు", "ఆరు", "ఏడు",
         "ఎనిమిది", "తొమ్మిది", "పది", "పదకొండు", "పన్నెండు", "పదమూడు",
         "పద్నాలుగు", "పదిహేను", "పదహారు", "పదిహేడు", "పద్దెనిమిది",
         "పందొమ్మిది"]
_TENS = {2: "ఇరవై", 3: "ముప్పై", 4: "నలభై", 5: "యాభై", 6: "అరవై",
         7: "డెబ్బై", 8: "ఎనభై", 9: "తొంభై"}


def _below_100(n):
    if n < 20:
        return _ONES[n]
    t, o = divmod(n, 10)
    return _TENS[t] + ("" if o == 0 else " " + _ONES[o])


def _below_1000(n):
    if n < 100:
        return _below_100(n)
    h, r = divmod(n, 100)
    if r == 0:
        return "వంద" if h == 1 else _ONES[h] + " వందలు"
    head = "నూట" if h == 1 else _ONES[h] + " వందల"
    return head + " " + _below_100(r)


def num_to_words(n):
    n = int(n)
    if n < 1000:
        return _below_1000(n)
    for unit, sing, plur_final, plur_join in (
        (10 ** 7, "కోటి", "కోట్లు", "కోట్ల"),
        (10 ** 5, "లక్ష", "లక్షలు", "లక్షల"),
        (1000, "వెయ్యి", "వేలు", "వేల"),
    ):
        if n >= unit:
            q, r = divmod(n, unit)
            if q == 1:
                head = sing
            else:
                head = num_to_words(q) + " " + (plur_final if r == 0 else plur_join)
            return head if r == 0 else head + " " + num_to_words(r)
    return _below_1000(n)


def ordinal(n):
    """12 -> పన్నెండవ (used for '12వ')."""
    w = num_to_words(n)
    if w.endswith("ై"):
        return w[:-1] + "య్యవ"
    if w[-1] in "ిు":
        return w[:-1] + "వ"
    return w + "వ"


def normalise(text):
    t = unicodedata.normalize("NFC", text)
    t = t.replace("‌", "").replace("‍", "")
    # time 10:45 -> పది నలభై ఐదు
    t = re.sub(r"(\d{1,2}):(\d{2})",
               lambda m: num_to_words(m.group(1))
               + ("" if int(m.group(2)) == 0 else " " + num_to_words(m.group(2))), t)
    t = re.sub(r"(\d+)\s*వ(?![ా-్])", lambda m: ordinal(m.group(1)) + " ", t)
    t = re.sub(r"(\d),(\d)", r"\1\2", t)
    t = re.sub(r"\d+", lambda m: " " + num_to_words(m.group(0)) + " ", t)
    t = re.sub(r"[\"“”‘’'`]", "", t)
    return re.sub(r"\s+", " ", t).strip()


# ------------------------------------------------------------------- G2P
def word_to_phones(word):
    out = []
    chars = list(word)
    i = 0
    while i < len(chars):
        ch = chars[i]
        nxt = chars[i + 1] if i + 1 < len(chars) else ""
        if ch in CONSONANTS:
            out += CONSONANTS[ch].split()
            if nxt == VIRAMA:
                i += 2
                continue
            if nxt in MATRAS:
                out += MATRAS[nxt].split()
                i += 2
                continue
            out.append("a")
            i += 1
            continue
        if ch in VOWELS:
            out += VOWELS[ch].split()
        elif ch in MATRAS:  # stray matra
            out += MATRAS[ch].split()
        elif ch == ANUSVARA or ch == ARDHA:
            out.append(_NASAL_FOR.get(nxt, "m"))
        elif ch == VISARGA:
            out.append("h")
        # candrabindu, virama on its own, other marks: silent
        i += 1
    return out


_tel_word = re.compile(r"[ఀ-౿]+")


def tokenize(text):
    return re.findall(r"[ఀ-౿]+|[A-Za-z]+|[^\sఀ-౿A-Za-z]", text)


def repair_splits(tokens, vocab):
    """Re-join a word broken by a stray space, e.g. 'అడ విలో' -> 'అడవిలో'.

    Fires only when the joined form occurs in the training-transcript vocabulary
    and the left fragment does not."""
    if not vocab:
        return tokens, []
    out, repairs, i = [], [], 0
    while i < len(tokens):
        a = tokens[i]
        if i + 1 < len(tokens):
            b = tokens[i + 1]
            if (_tel_word.fullmatch(a) and _tel_word.fullmatch(b)
                    and a not in vocab and (a + b) in vocab):
                out.append(a + b)
                repairs.append((a, b, a + b))
                i += 2
                continue
        out.append(a)
        i += 1
    return out, repairs


class TeluguFrontend:
    def __init__(self, vocab_path=None, repair=True):
        self.vocab = set()
        if vocab_path:
            try:
                with open(vocab_path, encoding="utf-8") as f:
                    self.vocab = {l.split()[0] for l in f if l.strip()}
            except FileNotFoundError:
                pass
        self.repair = repair

    def __call__(self, text, verbose=False):
        norm = normalise(text)
        toks = tokenize(norm)
        repairs = []
        if self.repair:
            toks, repairs = repair_splits(toks, self.vocab)
        phones, trace = [], []
        for tok in toks:
            if _tel_word.fullmatch(tok):
                ph = word_to_phones(tok)
                phones += ph
                trace.append((tok, "in-vocab" if tok in self.vocab else "OOV", " ".join(ph)))
            elif tok in ",;:.!?-()।|":
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
                print("  %-28s %-8s %s" % t)
        return phones, info
