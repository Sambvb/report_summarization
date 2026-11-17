"""
preprocessing.py
----------------
Text cleaning helpers used by BOTH models.

    clean_text()       -> removes extra spaces / strange characters
    split_sentences()  -> breaks a long report into individual sentences
    chunk_text()       -> cuts a long report into pieces small enough for BART
    tokenize()         -> breaks a sentence into lowercase words (for the BiLSTM)
    Vocabulary         -> converts words into numbers (neural networks only understand numbers)
"""

import json
import re
from collections import Counter

PAD_TOKEN = "<pad>"   # filler used to make all sentences the same length
UNK_TOKEN = "<unk>"   # used for words the model has never seen
NUM_TOKEN = "<num>"   # every number (5, 12.4, 1,200) becomes this one token


# ---------------------------------------------------------------------------
# 1. Cleaning & splitting
# ---------------------------------------------------------------------------
def clean_text(text: str) -> str:
    """Normalise quotes/dashes and collapse multiple spaces & new lines."""
    text = text.replace("’", "'").replace("‘", "'")
    text = text.replace("“", '"').replace("”", '"')
    text = text.replace("–", "-").replace("—", "-")
    text = re.sub(r"\s+", " ", text)          # many spaces/newlines -> one space
    return text.strip()


def split_sentences(text: str) -> list[str]:
    """
    Split text into sentences: a new sentence starts after . ! or ?
    followed by a space and a capital letter / number / quote.
    Very short pieces (< 4 words, e.g. headings) are dropped.
    """
    text = clean_text(text)
    parts = re.split(r'(?<=[.!?])\s+(?=[A-Z0-9"(])', text)
    return [p.strip() for p in parts if len(p.split()) >= 4]


def chunk_text(text: str, max_words: int) -> list[str]:
    """
    BART can only read about one page at a time, so we group whole
    sentences together until a chunk reaches `max_words` words.
    """
    chunks, current, count = [], [], 0
    for sentence in split_sentences(text):
        n = len(sentence.split())
        if current and count + n > max_words:
            chunks.append(" ".join(current))
            current, count = [], 0
        current.append(sentence)
        count += n
    if current:
        chunks.append(" ".join(current))
    return chunks


# ---------------------------------------------------------------------------
# 2. Tokenizing (sentence -> list of words) for the BiLSTM
# ---------------------------------------------------------------------------
def tokenize(sentence: str) -> list[str]:
    """
    "Profit rose 12.5% to EUR 3.2 mn." ->
    ['profit', 'rose', '<num>', '%', 'to', 'eur', '<num>', 'mn']
    """
    tokens = re.findall(r"[a-z]+(?:'[a-z]+)?|\d[\d,.]*|%", sentence.lower())
    return [NUM_TOKEN if t[0].isdigit() else t for t in tokens]


# ---------------------------------------------------------------------------
# 3. Vocabulary (word <-> number)
# ---------------------------------------------------------------------------
class Vocabulary:
    """
    Gives every known word an ID number, e.g. {'<pad>':0, '<unk>':1, 'profit':2, ...}
    The BiLSTM receives these numbers instead of words.
    """

    def __init__(self, word_to_id: dict[str, int] | None = None):
        self.word_to_id = word_to_id or {PAD_TOKEN: 0, UNK_TOKEN: 1}

    def __len__(self):
        return len(self.word_to_id)

    @classmethod
    def build(cls, sentences: list[str], min_freq: int) -> "Vocabulary":
        """Create the vocabulary from the TRAINING sentences only."""
        counts = Counter(word for s in sentences for word in tokenize(s))
        vocab = cls()
        for word, freq in counts.most_common():
            if freq >= min_freq:
                vocab.word_to_id[word] = len(vocab.word_to_id)
        return vocab

    def encode(self, sentence: str, max_len: int) -> list[int]:
        """Sentence -> fixed-length list of IDs (cut if too long, padded with 0 if too short)."""
        ids = [self.word_to_id.get(w, 1) for w in tokenize(sentence)][:max_len]
        return ids + [0] * (max_len - len(ids))

    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.word_to_id, f)

    @classmethod
    def load(cls, path) -> "Vocabulary":
        with open(path, encoding="utf-8") as f:
            return cls(json.load(f))
