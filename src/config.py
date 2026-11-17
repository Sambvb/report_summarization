"""
config.py
---------
ONE place for all settings (file paths, model names, training numbers).
If you want to change something (e.g. number of epochs), change it here
and every other file will automatically use the new value.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Folder paths (all relative to the project root folder)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent   # "report summarization/"

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"                          # downloaded training dataset
SAMPLE_REPORTS_DIR = DATA_DIR / "sample_reports"         # example reports for the demo

MODELS_DIR = PROJECT_ROOT / "models"                     # trained BiLSTM is saved here
BILSTM_WEIGHTS_PATH = MODELS_DIR / "bilstm_sentiment.pt" # learned weights of the BiLSTM
VOCAB_PATH = MODELS_DIR / "vocab.json"                   # word -> number dictionary

# ---------------------------------------------------------------------------
# Dataset: Financial PhraseBank (Malo et al., 2014)
# ~4,800 sentences from financial news, each labelled by finance experts as
# positive / negative / neutral.
# ---------------------------------------------------------------------------
DATASET_URL = (
    "https://huggingface.co/datasets/takala/financial_phrasebank/"
    "resolve/main/data/FinancialPhraseBank-v1.0.zip"
)
# "75Agree" = keep sentences where at least 75% of the experts agreed on the label
DATASET_FILE = "Sentences_75Agree.txt"

LABELS = ["negative", "neutral", "positive"]             # index 0, 1, 2

# ---------------------------------------------------------------------------
# BiLSTM settings (our own model, trained from scratch)
# ---------------------------------------------------------------------------
MAX_SEQ_LEN = 50        # every sentence is cut / padded to 50 words
MIN_WORD_FREQ = 2       # words seen less than 2 times are treated as "unknown"
EMBEDDING_DIM = 100     # each word becomes a vector of 100 numbers
HIDDEN_DIM = 64         # LSTM memory size (per direction)
NUM_LAYERS = 1          # stacked LSTM layers
DROPOUT = 0.5           # randomly switch off 50% of neurons while training (prevents over-fitting)

BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 1e-3
RANDOM_SEED = 42

# ---------------------------------------------------------------------------
# BART settings (pre-trained summarization model, downloaded from HuggingFace)
# ---------------------------------------------------------------------------
# DistilBART = a smaller, faster version of BART, already trained on
# CNN/DailyMail news articles to write summaries. No training needed.
SUMMARIZER_MODEL = "sshleifer/distilbart-cnn-12-6"

CHUNK_WORDS = 300         # BART can read ~1024 tokens at once, so long reports are cut into pieces
                          # (smaller pieces = more bullet points, so fewer important facts are lost)
SUMMARY_MAX_TOKENS = 120  # max length of each piece's summary
SUMMARY_MIN_TOKENS = 30   # min length of each piece's summary

# ---------------------------------------------------------------------------
# Finance keyword rule (applied on top of the BiLSTM)
# ---------------------------------------------------------------------------
# The BiLSTM learned that "rose / increased" usually means good news, so it can
# miss obviously bad sentences in a different style. If a sentence contains one
# of these strongly negative finance words, it is marked "negative".
NEGATIVE_FINANCE_WORDS = [
    "net loss", "losses", "(loss)", "impairment", "write-down", "write-off", "downgrade",
    "downgraded", "suspend", "suspended", "breach", "breaching", "waiver",
    "default", "disappointing", "uncertainty", "cancelled", "withdrawn",
    "deteriorat", "challenging", "negative", "squeezed", "below investment grade",
    "headcount reduction", "layoff", "job cuts", "headwind", "unused capacity",
]
# Costs or debt going UP is bad news (even though "rose" sounds positive),
# and costs or debt going DOWN is good news.
COST_WORDS = ["expense", "cost", "debt", "leverage", "borrowing", "interest"]
UP_WORDS = ["rose", "increased", "higher", "risen", "grew", " up "]
DOWN_WORDS = ["declined", "decline", "reduced", "reduce", "lower", "fell", "decreased"]
# words that flip the meaning ("NO indication of impairment", "grew at a SLOWER rate")
NEGATION_WORDS = [" not ", " no ", "without", "slower"]

# Forward-looking statements = PROMISES about the future ("we expect", "will improve").
# Companies always sound optimistic about the future, so when calculating the
# overall verdict these sentences count only half as much as actual results.
FORWARD_LOOKING_WORDS = [" will ", " expect", " aim", " plan ", " plans ", " intend",
                         " forecast", " anticipate", " commitment", " goal"]
FORWARD_LOOKING_WEIGHT = 0.5
