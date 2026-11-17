"""
summarizer.py
-------------
Writes the short summary using BART (a pre-trained Transformer model).

    * We do NOT train BART. "sshleifer/distilbart-cnn-12-6" was already trained by
      HuggingFace on ~300,000 news articles + their human-written summaries.
    * It is "abstractive": it writes NEW sentences, not just copy-paste.
    * Limitation: BART reads at most ~1024 tokens (~1 page). So a long report is
      cut into chunks, each chunk is summarised, and the results become bullet points.
    * The first run downloads the model (~1.2 GB) into the HuggingFace cache.
"""

import re

from src import config   # imported FIRST: src/__init__.py sets where BART is stored
from src.preprocessing import chunk_text, split_sentences

import torch  # noqa: E402
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer  # noqa: E402


class ReportSummarizer:
    def __init__(self):
        # Tokenizer: text <-> token IDs.   Model: the BART encoder-decoder network.
        # Runs on the CPU, so it works on any laptop.
        self.tokenizer = AutoTokenizer.from_pretrained(config.SUMMARIZER_MODEL)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(config.SUMMARIZER_MODEL)
        self.model.eval()

    @torch.no_grad()
    def _summarize_chunk(self, chunk: str, max_len: int, min_len: int) -> str:
        # 1) encoder reads the chunk (cut at BART's 1024-token limit)
        inputs = self.tokenizer(chunk, return_tensors="pt", truncation=True, max_length=1024)
        # 2) decoder writes the summary word by word (beam search = tries
        #    several candidate sentences and keeps the best one)
        output_ids = self.model.generate(
            **inputs,
            max_length=max_len,
            min_length=min_len,
            num_beams=4,
            no_repeat_ngram_size=3,   # never repeat the same 3 words -> no repetition
            length_penalty=2.0,
            early_stopping=True,
            do_sample=False,          # always give the same answer for the same input
        )
        # 3) token IDs -> readable text
        return self.tokenizer.decode(output_ids[0], skip_special_tokens=True)

    def summarize(self, text: str) -> list[str]:
        """Long report -> list of short bullet-point sentences."""
        chunks = chunk_text(text, config.CHUNK_WORDS)
        if not chunks:
            return []

        bullets = []
        for chunk in chunks:
            words = len(chunk.split())
            # a very small chunk doesn't need summarising, and the summary
            # must never be longer than the input
            max_len = min(config.SUMMARY_MAX_TOKENS, max(20, int(words * 0.8)))
            min_len = min(config.SUMMARY_MIN_TOKENS, max_len - 5)

            summary = self._summarize_chunk(chunk, max_len, min_len)

            # BART sometimes writes "growth ." -> remove the space before punctuation
            summary = re.sub(r"\s+([.,;:!?])", r"\1", summary)
            bullets.extend(split_sentences(summary))

        # tidy up: remove duplicates and half-finished sentences, keep the order
        seen, unique = set(), []
        for b in bullets:
            b = b.strip()
            if b.split()[-1].lower().strip('",.') in {"says", "said", "according", "and", "the"}:
                continue                         # BART stopped in the middle of a sentence
            if b.startswith('"'):
                continue                         # a quote without context isn't useful to a layman
            if b[-1] not in '.!?"':
                b += "."
            if b.lower() not in seen:
                seen.add(b.lower())
                unique.append(b[0].upper() + b[1:])
        return unique
