"""
sentiment.py
------------
Uses the TRAINED BiLSTM (from models/) to:
    1. label every sentence of a report as negative / neutral / positive
    2. give an overall verdict for the whole report
    3. pick the strongest "good news" and "concern" sentences
"""

import torch

from src import config
from src.bilstm_model import BiLSTMClassifier
from src.preprocessing import Vocabulary, split_sentences


class SentimentAnalyzer:
    def __init__(self):
        if not config.BILSTM_WEIGHTS_PATH.exists():
            raise FileNotFoundError(
                "BiLSTM model not found. Train it first:  python scripts/train_bilstm.py"
            )
        # Load the word->number dictionary and the trained weights
        self.vocab = Vocabulary.load(config.VOCAB_PATH)
        self.model = BiLSTMClassifier(
            vocab_size=len(self.vocab),
            embedding_dim=config.EMBEDDING_DIM,
            hidden_dim=config.HIDDEN_DIM,
            num_classes=len(config.LABELS),
            num_layers=config.NUM_LAYERS,
            dropout=config.DROPOUT,
        )
        self.model.load_state_dict(torch.load(config.BILSTM_WEIGHTS_PATH, map_location="cpu"))
        self.model.eval()   # switch off dropout -> prediction mode

    @torch.no_grad()
    def predict(self, sentences: list[str]) -> list[dict]:
        """Return label + confidence for each sentence."""
        if not sentences:
            return []
        ids = torch.tensor([self.vocab.encode(s, config.MAX_SEQ_LEN) for s in sentences])
        probs = torch.softmax(self.model(ids), dim=1)        # scores -> probabilities (sum = 1)
        results = []
        for sentence, p in zip(sentences, probs):
            idx = int(p.argmax())
            label, confidence, source = config.LABELS[idx], float(p[idx]), "BiLSTM"

            # Finance keyword rule (see config.py) - corrects common BiLSTM mistakes
            lower = f" {sentence.lower()} "
            def has(words):
                return any(w in lower for w in words)

            if not has(config.NEGATION_WORDS):
                is_cost = has(config.COST_WORDS)
                if label != "negative" and (has(config.NEGATIVE_FINANCE_WORDS)
                                            or (is_cost and has(config.UP_WORDS))):
                    label, confidence, source = "negative", 0.90, "BiLSTM + keyword rule"
                elif label != "positive" and is_cost and has(config.DOWN_WORDS) \
                        and not has(config.UP_WORDS):
                    label, confidence, source = "positive", 0.90, "BiLSTM + keyword rule"

            results.append({
                "sentence": sentence,
                "label": label,
                "confidence": confidence,
                "source": source,
            })
        return results

    def analyze_report(self, text: str, top_k: int = 3) -> dict:
        """Sentence-level predictions + overall verdict for a full report."""
        predictions = self.predict(split_sentences(text))

        counts = {label: 0 for label in config.LABELS}
        weighted = {"positive": 0.0, "negative": 0.0}
        for p in predictions:
            counts[p["label"]] += 1
            # promises about the future count less than actual results
            lower = f" {p['sentence'].lower()} "
            p["forward_looking"] = any(w in lower for w in config.FORWARD_LOOKING_WORDS)
            if p["label"] in weighted:
                weighted[p["label"]] += config.FORWARD_LOOKING_WEIGHT if p["forward_looking"] else 1.0

        # Overall score (-1 .. +1): among sentences that carry an opinion (ignore neutral facts),
        # how much more positive than negative is the report?
        opinionated = max(weighted["positive"] + weighted["negative"], 1)
        score = (weighted["positive"] - weighted["negative"]) / opinionated
        if score > 0.20:
            verdict = "Positive - overall the news is GOOD for the company"
        elif score < -0.20:
            verdict = "Negative - overall the news is BAD / worrying for the company"
        else:
            verdict = "Mixed / Neutral - good and bad news are roughly balanced"

        def strongest(label):
            chosen = [p for p in predictions if p["label"] == label]
            return sorted(chosen, key=lambda p: p["confidence"], reverse=True)[:top_k]

        return {
            "verdict": verdict,
            "score": round(score, 2),
            "counts": counts,
            "good_news": strongest("positive"),
            "concerns": strongest("negative"),
            "sentences": predictions,
        }
