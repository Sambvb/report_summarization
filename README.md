# 📊 Financial Report Summarizer

Turns a long, jargon-filled financial report (2+ pages) into a **short, simple summary that anyone can understand**.

**Input:** paste report text
**Output:**
1. 📝 **Simple bullet-point summary**, written by **BART** (pre-trained Transformer)
2. 📈 **Overall verdict** (good / bad / mixed news) plus the top **good news** and **concerns**, from our **BiLSTM** (trained by us)
3. 📖 **Difficult words explained** (EBITDA, impairment, liquidity...) in plain English

---

## 📁 Folder structure (what each folder means)

```
report summarization/
│
├── data/                         ← ALL DATA
│   ├── raw/                        downloaded training dataset (Financial PhraseBank), auto-created
│   └── sample_reports/             example reports for the demo: 2 fictional + 2 REAL press releases (NVIDIA, Intel)
│
├── models/                       ← TRAINED MODEL FILES
│   ├── bilstm_sentiment.pt         learned weights of our BiLSTM (created by the training script)
│   ├── vocab.json                  word → number dictionary used by the BiLSTM
│   └── hf_cache/                   downloaded pre-trained BART model (~1.2 GB, auto-downloaded)
│
├── src/                          ← SOURCE CODE: all the project logic
│   ├── config.py                   settings & paths (change epochs, model names here)
│   ├── preprocessing.py            cleaning, sentence splitting, chunking, vocabulary
│   ├── bilstm_model.py             BiLSTM neural network architecture (PyTorch)
│   ├── sentiment.py                uses trained BiLSTM → verdict, good news, concerns
│   ├── summarizer.py               uses pre-trained BART → bullet-point summary
│   ├── jargon.py                   financial dictionary → plain-English explanations
│   └── pipeline.py                 MAIN: joins everything (text in → result out)
│
├── scripts/                      ← ONE-TIME SCRIPTS
│   └── train_bilstm.py             downloads data, trains, tests & saves the BiLSTM
│
├── notebooks/                    ← JUPYTER NOTEBOOK (step-by-step demo)
│   └── demo.ipynb                  input cell → output cell, runs inside VS Code
│
├── app/                          ← WEB DEMO
│   └── app.py                      Streamlit web page (paste → click → summary)
│
├── requirements.txt              ← libraries to install
└── README.md                     ← this file
```

---

## ▶️ How to run

Open a terminal in the project folder:

```bash
# 1. Activate the virtual environment (already created in .venv)
.venv\Scripts\activate

# 2. (Only if setting up on a new computer) install libraries
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

# 3. Train the BiLSTM (ONE time only, ~2-5 min)
python scripts/train_bilstm.py

# 4a. Demo as a web page
streamlit run app/app.py

# 4b. ...or open notebooks/demo.ipynb in VS Code
#     → top-right "Select Kernel" → "Python (report-summarization)" (or ".venv") → Run All
```

> The first run downloads the BART model (~1.2 GB). After that everything works offline.

---

## 🧠 How it works

```
report text ──► clean & split into sentences / chunks
                     │
                     ├──► BART summarizer (per chunk) ──► jargon simplifier ──► bullet points
                     │
                     ├──► BiLSTM (per sentence) ──► positive / neutral / negative
                     │                              ──► overall verdict + good news + concerns
                     │
                     └──► jargon finder ──► "difficult words explained" glossary
```

| Component | What it is | Trained by us? | Dataset |
|---|---|---|---|
| **BiLSTM** | Bidirectional LSTM (recurrent neural network) | ✅ Yes | Financial PhraseBank: 3,453 financial news sentences on which at least 75% of the expert labellers agreed |
| **Finance keyword rule** | Rule-based correction on top of the BiLSTM | — | Lists in `config.py` |
| **DistilBART** (`sshleifer/distilbart-cnn-12-6`) | Pre-trained Transformer encoder-decoder | ❌ No (transfer learning) | Already trained on CNN/DailyMail (~300k articles and summaries) |
| **Jargon dictionary** | Rule-based | — | ~70 hand-written financial terms |

### BiLSTM architecture
```
word IDs → Embedding(100) → BiLSTM(64 forward + 64 backward) → Dropout(0.3) → Linear → 3 classes
```
- **Embedding**: each word becomes a 100-number vector that the model learns.
- **Bidirectional**: reads the sentence left→right AND right→left, so each word sees context on both sides (e.g. "did **not** increase").
- **Packed sequences**: padding is skipped, so the LSTM only reads real words.
- **Class weights**: "neutral" is the majority class, so rarer classes get more weight. Otherwise the model would just predict "neutral".
- **Best-epoch saving**: the version with the highest validation accuracy is kept.

### BiLSTM results (held-out test set, 346 unseen sentences)
| | precision | recall | F1 |
|---|---|---|---|
| negative | 0.66 | 0.60 | 0.63 |
| neutral | 0.87 | 0.93 | 0.89 |
| positive | 0.81 | 0.71 | 0.75 |
| **accuracy** | | | **82.9%** |

The first version (66%-agreement data, dropout 0.3) reached **73.5%**. Switching to cleaner labels (75%-agreement) and stronger dropout (0.5) to reduce overfitting raised it to **82.9%**.

### Finance keyword rule (why it exists)
The BiLSTM learned from the data that "rose / increased" usually means good news. So it marked *"Interest expense rose 31%"* as **positive**, which is wrong. A small rule layer fixes these cases:
- strongly negative words (*net loss, impairment, downgraded, suspended, breach...*) → negative
- cost or debt **going up** → negative; cost or debt **going down** → positive
- the rule is skipped when the sentence has a negation (*"**no** indication of impairment"*)

In the app you can see, for each sentence, whether it was decided by the "BiLSTM" alone or by "BiLSTM + keyword rule".

### Forward-looking statements count half
Companies always sound optimistic about the future (*"these actions **will** strengthen our position"*). When the overall verdict is calculated, sentences that are **promises** (*will, expect, aim, plan, forecast...*) count **half** as much as **actual results**. Without this, Intel's Q2 2024 release (a loss, a suspended dividend and 15% job cuts) was wrongly rated "Positive" because of all its optimistic promises.

### Tested on real reports
| Report | Verdict |
|---|---|
| `03_REAL_nvidia_q3_fy2025_record_quarter.txt`: record revenue, +94% YoY | ✅ Positive (+0.90) |
| `04_REAL_intel_q2_2024_bad_quarter.txt`: loss, dividend suspended, 15% job cuts | ✅ Negative (−0.26) |
| `01_good_quarter_tech_company.txt` (fictional) | ✅ Positive (+0.59) |
| `02_bad_year_retail_company.txt` (fictional) | ✅ Negative (−0.66) |

---

## 🎤 Interview cheat-sheet

**Q: Why not use the BiLSTM for summarization itself?**
A seq2seq BiLSTM summarizer needs tens of thousands of report→summary pairs and days of GPU training, and the output is still often repetitive. So I used the BiLSTM where it is strong (**sentence classification**, which trains in minutes on a CPU) and a **pre-trained Transformer (BART)** for generating text. Using a pre-trained model like this is called **transfer learning**.

**Q: Why BART?**
It is an encoder-decoder Transformer built for **abstractive** summarization: it writes new sentences instead of copying them. DistilBART is a compressed version that is about 2× faster with nearly the same quality, so it runs on a normal laptop.

**Q: How do you handle long reports?**
BART reads at most ~1024 tokens (about 1 page). The report is split into chunks of ~450 words at sentence boundaries. Each chunk is summarised, and the results become the bullet points.

**Q: Extractive vs abstractive?**
Extractive picks existing sentences (like our BiLSTM "good news / concerns"). Abstractive writes new sentences (BART). This project uses **both**.

**Q: How did you evaluate?**
The BiLSTM is tested on a held-out 10% test set (accuracy, precision/recall/F1 and confusion matrix printed by `train_bilstm.py`). The split is 80/10/10 and stratified.

**Q: Why is there a keyword rule if you have a neural network?**
Error analysis showed the BiLSTM gets confused when a "positive-sounding" verb is attached to a cost ("expenses **rose**"), because the training data rarely contains this pattern. Combining a learned model with a few domain rules (a **hybrid** approach) is common in industry when the data is small.

**Q: Limitations / future work?**
- BART was trained on news, not finance. Fine-tuning on **ECTSum** (earnings-call summaries) would improve it.
- Replace the BiLSTM with **FinBERT** for higher sentiment accuracy.
- Support PDF upload and tables.
- The jargon dictionary is rule-based. An LLM could rewrite the text in fully plain language.
