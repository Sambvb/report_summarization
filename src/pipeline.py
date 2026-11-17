"""
pipeline.py
-----------
The MAIN entry point. Joins all the pieces together:

    report text
        |-- BART summarizer   -> bullet-point summary
        |-- Jargon simplifier -> plain words in the summary + glossary
        |-- BiLSTM            -> overall verdict, good news, concerns
        v
    one result dictionary  (used by the notebook AND the Streamlit app)
"""

import time

from src.jargon import find_jargon, simplify_summary
from src.preprocessing import clean_text
from src.sentiment import SentimentAnalyzer
from src.summarizer import ReportSummarizer


class FinancialReportPipeline:
    def __init__(self):
        # Loading models takes a few seconds, so we do it ONCE here
        self.sentiment = SentimentAnalyzer()     # our trained BiLSTM
        self.summarizer = ReportSummarizer()     # pre-trained BART

    def run(self, report_text: str) -> dict:
        start = time.time()
        text = clean_text(report_text)

        summary = simplify_summary(self.summarizer.summarize(text))
        sentiment = self.sentiment.analyze_report(text)
        glossary = find_jargon(text)

        return {
            "summary": summary,                      # list of simple bullet points
            "verdict": sentiment["verdict"],         # overall good / bad / mixed
            "score": sentiment["score"],             # -1 (very bad) .. +1 (very good)
            "counts": sentiment["counts"],           # how many pos / neu / neg sentences
            "good_news": sentiment["good_news"],     # strongest positive sentences
            "concerns": sentiment["concerns"],       # strongest negative sentences
            "sentences": sentiment["sentences"],     # every sentence with its label
            "glossary": glossary,                    # difficult word -> simple meaning
            "stats": {
                "original_words": len(text.split()),
                "summary_words": sum(len(b.split()) for b in summary),
                "seconds": round(time.time() - start, 1),
            },
        }


def print_result(result: dict) -> None:
    """Pretty print the result in the terminal / notebook."""
    line = "=" * 70
    print(line, "\nOVERALL VERDICT:", result["verdict"], f"(score {result['score']:+})")
    print(line, "\nSUMMARY IN SIMPLE POINTS:")
    for i, point in enumerate(result["summary"], 1):
        print(f"  {i}. {point}")

    print(line, "\nGOOD NEWS (found by BiLSTM):")
    for item in result["good_news"] or [{"sentence": "None found", "confidence": 0}]:
        print(f"  + {item['sentence']}")
    print("\nCONCERNS (found by BiLSTM):")
    for item in result["concerns"] or [{"sentence": "None found", "confidence": 0}]:
        print(f"  - {item['sentence']}")

    print(line, "\nDIFFICULT WORDS EXPLAINED:")
    for term, meaning in result["glossary"].items():
        print(f"  * {term}: {meaning}")

    s = result["stats"]
    print(line, f"\n{s['original_words']} words -> {s['summary_words']} words "
                f"in {s['seconds']} seconds")
