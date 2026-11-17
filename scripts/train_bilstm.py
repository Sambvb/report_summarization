"""
train_bilstm.py
---------------
RUN ONCE before the demo:      python scripts/train_bilstm.py
(takes ~2-5 minutes on a normal laptop CPU)

Steps:
    1. Download the Financial PhraseBank dataset (if not already in data/raw/)
    2. Split it into 80% training / 10% validation / 10% test sentences
    3. Build the vocabulary (word -> number) from the training sentences
    4. Train the BiLSTM for a few epochs, keeping the best version (by validation accuracy)
    5. Test on sentences the model has NEVER seen and print accuracy + report
    6. Save the model to models/bilstm_sentiment.pt and models/vocab.json
"""

import io
import random
import sys
import zipfile
from pathlib import Path

import numpy as np
import requests
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

# allow "from src import ..." when running this file directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config                                   # noqa: E402
from src.bilstm_model import BiLSTMClassifier            # noqa: E402
from src.preprocessing import Vocabulary                 # noqa: E402


# ---------------------------------------------------------------------------
# Step 1: get the data
# ---------------------------------------------------------------------------
def download_dataset() -> Path:
    """Download + unzip Financial PhraseBank into data/raw/ (only the first time)."""
    target = config.RAW_DATA_DIR / config.DATASET_FILE
    if target.exists():
        print(f"Dataset already present: {target}")
        return target

    print("Downloading Financial PhraseBank dataset ...")
    config.RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    response = requests.get(config.DATASET_URL, timeout=60)
    response.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(response.content)) as z:
        for name in z.namelist():
            if name.endswith(config.DATASET_FILE) and "__MACOSX" not in name:
                target.write_bytes(z.read(name))
    print(f"Saved to {target}")
    return target


def load_dataset(path: Path):
    """Each line of the file looks like:   <sentence>@<label>"""
    sentences, labels = [], []
    with open(path, encoding="latin-1") as f:
        for line in f:
            if "@" not in line:
                continue
            sentence, label = line.strip().rsplit("@", 1)
            sentences.append(sentence)
            labels.append(config.LABELS.index(label))
    return sentences, labels


def to_tensors(sentences, labels, vocab):
    x = torch.tensor([vocab.encode(s, config.MAX_SEQ_LEN) for s in sentences])
    y = torch.tensor(labels)
    return TensorDataset(x, y)


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------
def evaluate(model, loader):
    """Return (accuracy, all_predictions, all_true_labels)."""
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for x, y in loader:
            preds.extend(model(x).argmax(dim=1).tolist())
            trues.extend(y.tolist())
    accuracy = float(np.mean(np.array(preds) == np.array(trues)))
    return accuracy, preds, trues


def main():
    # same random numbers every run -> reproducible results
    random.seed(config.RANDOM_SEED)
    np.random.seed(config.RANDOM_SEED)
    torch.manual_seed(config.RANDOM_SEED)

    # ---- Step 1 & 2: data ------------------------------------------------
    sentences, labels = load_dataset(download_dataset())
    print(f"\nTotal sentences: {len(sentences)}")
    for i, name in enumerate(config.LABELS):
        print(f"  {name:<9}: {labels.count(i)}")

    x_train, x_temp, y_train, y_temp = train_test_split(
        sentences, labels, test_size=0.2, stratify=labels, random_state=config.RANDOM_SEED)
    x_val, x_test, y_val, y_test = train_test_split(
        x_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=config.RANDOM_SEED)
    print(f"Train: {len(x_train)} | Validation: {len(x_val)} | Test: {len(x_test)}")

    # ---- Step 3: vocabulary ---------------------------------------------
    vocab = Vocabulary.build(x_train, config.MIN_WORD_FREQ)
    print(f"Vocabulary size: {len(vocab)} words")

    train_loader = DataLoader(to_tensors(x_train, y_train, vocab),
                              batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(to_tensors(x_val, y_val, vocab), batch_size=config.BATCH_SIZE)
    test_loader = DataLoader(to_tensors(x_test, y_test, vocab), batch_size=config.BATCH_SIZE)

    # ---- Step 4: model + training ---------------------------------------
    model = BiLSTMClassifier(
        vocab_size=len(vocab),
        embedding_dim=config.EMBEDDING_DIM,
        hidden_dim=config.HIDDEN_DIM,
        num_classes=len(config.LABELS),
        num_layers=config.NUM_LAYERS,
        dropout=config.DROPOUT,
    )

    # "neutral" sentences are the majority -> give rarer classes more weight
    # so the model doesn't just predict "neutral" for everything
    counts = np.bincount(y_train, minlength=len(config.LABELS))
    class_weights = torch.tensor(len(y_train) / (len(counts) * counts), dtype=torch.float)

    loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    print(f"\nTraining for {config.EPOCHS} epochs ...")
    best_val_acc, best_state = 0.0, None
    for epoch in range(1, config.EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for x, y in train_loader:
            optimizer.zero_grad()            # clear old gradients
            loss = loss_fn(model(x), y)      # how wrong are the predictions?
            loss.backward()                  # compute gradients (backpropagation)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)   # avoid exploding gradients
            optimizer.step()                 # update the weights
            total_loss += loss.item()

        train_acc, _, _ = evaluate(model, train_loader)
        val_acc, _, _ = evaluate(model, val_loader)
        marker = ""
        if val_acc > best_val_acc:           # keep the best model only
            best_val_acc = val_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            marker = "  <- best so far"
        print(f"Epoch {epoch:2d} | loss {total_loss / len(train_loader):.3f} | "
              f"train acc {train_acc:.3f} | val acc {val_acc:.3f}{marker}")

    # ---- Step 5: final test on unseen sentences -------------------------
    model.load_state_dict(best_state)
    test_acc, preds, trues = evaluate(model, test_loader)
    print(f"\nTEST ACCURACY (unseen sentences): {test_acc:.3f}\n")
    print(classification_report(trues, preds, target_names=config.LABELS, digits=3))
    print("Confusion matrix (rows = true, columns = predicted):")
    print("            " + "  ".join(f"{l[:3]:>4}" for l in config.LABELS))
    for name, row in zip(config.LABELS, confusion_matrix(trues, preds)):
        print(f"  {name:<9} " + "  ".join(f"{v:>4}" for v in row))

    # ---- Step 6: save ---------------------------------------------------
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, config.BILSTM_WEIGHTS_PATH)
    vocab.save(config.VOCAB_PATH)
    print(f"\nSaved model -> {config.BILSTM_WEIGHTS_PATH}")
    print(f"Saved vocab -> {config.VOCAB_PATH}")


if __name__ == "__main__":
    main()
