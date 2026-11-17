"""
src = "source code". All the reusable logic of the project lives here.

    config.py         -> settings and file paths
    preprocessing.py  -> cleaning, sentence splitting, chunking, vocabulary
    bilstm_model.py   -> BiLSTM neural network definition (PyTorch)
    sentiment.py      -> uses the trained BiLSTM to find good / bad news
    summarizer.py     -> uses pre-trained BART to write the summary
    jargon.py         -> explains financial words in simple language
    pipeline.py       -> joins everything: text in -> simple summary out
"""

import os
from pathlib import Path

# Store the downloaded BART model INSIDE the project (models/hf_cache/) instead
# of the default C:\Users\<you>\.cache folder. This keeps the project
# self-contained and avoids filling up the C: drive.
# (This runs before any HuggingFace library is imported.)
os.environ.setdefault("HF_HOME", str(Path(__file__).resolve().parent.parent / "models" / "hf_cache"))
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
