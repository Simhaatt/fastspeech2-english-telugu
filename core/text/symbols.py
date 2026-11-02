""" from https://github.com/keithito/tacotron """

"""
Defines the set of symbols used in text input to the model.

The default is a set of ASCII characters that works well for English or text that has been run through Unidecode. For other data, you can modify _characters. See TRAINING_DATA.md for details. """

from text import cmudict, pinyin

_pad = "_"
_punctuation = "!'(),.:;? "
_special = "-"
_letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
_silences = ["@sp", "@spn", "@sil"]

# Prepend "@" to ARPAbet symbols to ensure uniqueness (some are the same as uppercase letters):
_arpabet = ["@" + s for s in cmudict.valid_symbols]
_pinyin = ["@" + s for s in pinyin.valid_symbols]

# Export all symbols:
symbols = (
    [_pad]
    + list(_special)
    + list(_punctuation)
    + list(_letters)
    + _arpabet
    + _pinyin
    + _silences
)

# Added for this assignment: Telugu phones from text/frontend_te.py.
# They are APPENDED (and only when not already present, e.g. "@b" also exists
# in the pinyin set) so the indices of every original symbol are unchanged.
# That keeps the public LJSpeech checkpoint loadable for the English
# fine-tuning ablation; a separate Telugu model simply ignores unused rows.
from text.frontend_te import PHONES as _telugu_phones

_existing = set(symbols)
symbols += ["@" + p for p in _telugu_phones if "@" + p not in _existing]
