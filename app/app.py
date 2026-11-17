"""
app.py  -  The web demo (Streamlit)
-----------------------------------
Start it with:      streamlit run app/app.py
A browser tab opens at http://localhost:8501

The user can:
    * paste a report, OR upload a .txt file, OR pick a sample report
    * click "Summarize"
    * see: overall verdict, simple bullet points, good news / concerns,
           difficult words explained, and how each sentence was classified
"""

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# allow "from src import ..." because this file lives inside app/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config                                   # noqa: E402
from src.pipeline import FinancialReportPipeline         # noqa: E402

st.set_page_config(page_title="Financial Report Summarizer", page_icon="📊", layout="wide")


# Load both models ONCE and keep them in memory (otherwise every click would reload them)
@st.cache_resource(show_spinner="Loading BiLSTM + BART models (first time can take a minute)...")
def load_pipeline():
    return FinancialReportPipeline()


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("📊 Financial Report Summarizer")
st.caption("Paste a long, complex financial report and get a short, simple summary "
           "that anyone can understand.  BiLSTM (sentiment) + BART (summary) + jargon explainer")

# ---------------------------------------------------------------------------
# Input: sample / upload / paste
# ---------------------------------------------------------------------------
samples = {p.stem.replace("_", " "): p for p in sorted(config.SAMPLE_REPORTS_DIR.glob("*.txt"))}

col_a, col_b = st.columns(2)
with col_a:
    choice = st.selectbox("Load a sample report (optional)", ["-- none --", *samples])
with col_b:
    uploaded = st.file_uploader("...or upload a .txt file", type=["txt"])

default_text = ""
if uploaded is not None:
    default_text = uploaded.read().decode("utf-8", errors="ignore")
elif choice != "-- none --":
    default_text = samples[choice].read_text(encoding="utf-8")

report = st.text_area("Report text", value=default_text, height=280,
                      placeholder="Paste the financial report here...")

if st.button("✨ Summarize", type="primary", width="stretch"):
    if len(report.split()) < 30:
        st.warning("Please paste a longer report (at least 30 words).")
        st.stop()

    try:
        pipe = load_pipeline()
    except FileNotFoundError as e:
        st.error(str(e))
        st.stop()

    with st.spinner("Reading the report and writing a simple summary..."):
        result = pipe.run(report)

    # -----------------------------------------------------------------------
    # Output
    # -----------------------------------------------------------------------
    st.divider()

    # 1) Overall verdict
    verdict = result["verdict"]
    if verdict.startswith("Positive"):
        st.success(f"📈 **Overall verdict:** {verdict}")
    elif verdict.startswith("Negative"):
        st.error(f"📉 **Overall verdict:** {verdict}")
    else:
        st.info(f"⚖️ **Overall verdict:** {verdict}")

    s = result["stats"]
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Original length", f"{s['original_words']} words")
    m2.metric("Summary length", f"{s['summary_words']} words")
    m3.metric("Reading time saved",
              f"~{max(1, round((s['original_words'] - s['summary_words']) / 200))} min")
    m4.metric("Processed in", f"{s['seconds']} s")

    # 2) Summary
    st.subheader("📝 Summary in simple points")
    for point in result["summary"]:
        st.markdown(f"- {point}")

    # 3) Good news vs concerns
    st.subheader("🔍 What's good and what's worrying (BiLSTM)")
    g, c = st.columns(2)
    with g:
        st.markdown("**✅ Good news**")
        for item in result["good_news"] or [{"sentence": "Nothing clearly positive found."}]:
            st.markdown(f"- {item['sentence']}")
    with c:
        st.markdown("**⚠️ Concerns**")
        for item in result["concerns"] or [{"sentence": "Nothing clearly negative found."}]:
            st.markdown(f"- {item['sentence']}")

    # 4) Sentiment chart
    counts = result["counts"]
    st.bar_chart(pd.DataFrame({"sentences": [counts["positive"], counts["neutral"],
                                             counts["negative"]]},
                              index=["positive", "neutral", "negative"]))

    # 5) Glossary
    if result["glossary"]:
        st.subheader("📖 Difficult words explained")
        st.table(pd.DataFrame(list(result["glossary"].items()),
                              columns=["Word in report", "What it means"]))

    # 6) Details: every sentence and its label
    with st.expander("See how the BiLSTM labelled every sentence"):
        st.dataframe(pd.DataFrame([
            {"label": p["label"], "confidence": round(p["confidence"], 2),
             "decided by": p["source"], "sentence": p["sentence"]}
            for p in result["sentences"]
        ]), width="stretch")
