# =========================================================
# TAMIL POLITICAL SENTIMENT – FULL K-FOLD WORKFLOW WITH TQDM
# TRAIN: PS_train.csv
# DEV:   PS_dev.csv
# TEST:  PS_test_without_labels.csv
# XLM-R + IndicBERT
# Mean Pooling + Oversampling + Focal Loss + Early Stop
# SAVE BEST MODEL EACH FOLD + PREDICT TEST
# =========================================================

import os, re, unicodedata, torch, numpy as np, pandas as pd
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report
from sklearn.utils.class_weight import compute_class_weight
from sklearn.utils import resample
from tqdm import tqdm

# =========================
# SETUP
# =========================
os.environ["TOKENIZERS_PARALLELISM"] = "false"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

EPOCHS = 15
BATCH_SIZE = 16
MAX_LEN = 256
LR = 1.5e-5
PATIENCE = 3
N_FOLDS = 10
NUM_CLASSES = 7

SAVE_DIR = "saved_models_kfold"
os.makedirs(SAVE_DIR, exist_ok=True)

# =========================
# LOAD DATA
# =========================
train_df = pd.read_csv("PS_train.csv")
dev_df   = pd.read_csv("PS_dev.csv")
test_df  = pd.read_csv("PS_test_without_labels.csv")

# =========================
# LABEL MAPPING
# =========================
label2id = {
    "Substantiated": 0,
    "Sarcastic": 1,
    "Opinionated": 2,
    "Positive": 3,
    "Negative": 4,
    "Neutral": 5,
    "None of the above": 6
}
id2label = {v: k for k, v in label2id.items()}

train_df["label_id"] = train_df["labels"].map(label2id)
dev_df["label_id"]   = dev_df["labels"].map(label2id)

# =========================
# CLEAN TEXT
# =========================
def clean(text):
    text = str(text)
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

train_df["content"] = train_df["content"].apply(clean)
dev_df["content"]   = dev_df["content"].apply(clean)
test_df["content"]  = test_df["content"].apply(clean)

# =========================
# OVERSAMPLING TRAIN
# =========================
dfs = []
max_size = train_df["labels"].value_counts().max()
for label in train_df["labels"].unique():
    temp = train_df[train_df["labels"] == label]
    if len(temp) < max_size:
        temp = resample(temp, replace=True, n_samples=max_size, random_state=42)
    dfs.append(temp)
train_balanced = pd.concat(dfs).sample(frac=1, random_state=42).reset_index(drop=True)

print("\nTrain size before:", len(train_df))
print("Train size after oversample:", len(train_balanced))
print("Dev size:", len(dev_df))
print("Test size:", len(test_df))

# =========================
# DATASET CLASSES
# =========================
class TamilDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.texts = texts.tolist()
        self.labels = labels.tolist()
        self.tokenizer = tokenizer

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            padding="max_length",
            truncation=True,
            max_length=MAX_LEN,
            return_tensors="pt"
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long)
        }

    def __len__(self):
        return len(self.texts)

class TestDataset(Dataset):
    def __init__(self, texts, tokenizer):
        self.texts = texts
        self.tokenizer = tokenizer

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            padding="max_length",
            truncation=True,
            max_length=MAX_LEN,
            return_tensors="pt"
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0)
        }

    def __len__(self):
        return len(self.texts)

# =========================
# MODEL (MEAN POOLING)
# =========================
class TransformerClassifier(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(self.encoder.config.hidden_size, NUM_CLASSES)

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden = out.last_hidden_state
        mask = attention_mask.unsqueeze(-1).expand(hidden.size()).float()
        pooled = torch.sum(hidden * mask, dim=1) / torch.clamp(mask.sum(dim=1), min=1e-9)
        return self.fc(self.dropout(pooled))

# =========================
# FOCAL LOSS
# =========================
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, weight=None):
        super().__init__()
        self.gamma = gamma
        self.ce = nn.CrossEntropyLoss(weight=weight)

    def forward(self, logits, targets):
        ce_loss = self.ce(logits, targets)
        pt = torch.exp(-ce_loss)
        return ((1 - pt) ** self.gamma) * ce_loss

# =========================
# TRAIN K-FOLD AND PREDICT DEV
# =========================
def train_kfold_and_predict_dev(model_name):
    print(f"\n🚀 Training → {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    dev_ds = TamilDataset(dev_df["content"], dev_df["label_id"], tokenizer)
    dev_loader = DataLoader(dev_ds, batch_size=BATCH_SIZE, shuffle=False)

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    train_labels = train_balanced["label_id"].values
    dev_logits_all_folds = []

    for fold, (train_idx, val_idx) in enumerate(
        skf.split(train_balanced["content"], train_labels)
    ):
        print(f"\n📂 Fold {fold+1}/{N_FOLDS}")

        train_ds = TamilDataset(
            train_balanced.iloc[train_idx]["content"],
            train_balanced.iloc[train_idx]["label_id"],
            tokenizer
        )
        val_ds = TamilDataset(
            train_balanced.iloc[val_idx]["content"],
            train_balanced.iloc[val_idx]["label_id"],
            tokenizer
        )

        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

        weights = compute_class_weight(
            class_weight="balanced",
            classes=np.unique(train_labels),
            y=train_labels[train_idx]
        )
        weights = torch.tensor(weights, dtype=torch.float).to(device)

        model = TransformerClassifier(model_name).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            int(0.1 * len(train_loader) * EPOCHS),
            len(train_loader) * EPOCHS
        )
        criterion = FocalLoss(weight=weights)

        best_acc, patience, best_state = 0, 0, None

        # ========== TRAIN LOOP WITH PROGRESS BAR ==========
        for epoch in range(EPOCHS):
            model.train()
            loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} Train", leave=False)
            for batch in loop:
                optimizer.zero_grad()
                logits = model(batch["input_ids"].to(device), batch["attention_mask"].to(device))
                loss = criterion(logits, batch["labels"].to(device))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                loop.set_postfix(loss=loss.item())
            loop.close()

            # ========== VALIDATION LOOP ==========
            model.eval()
            val_logits = []
            loop_val = tqdm(val_loader, desc=f"Fold {fold+1} Validation", leave=False)
            with torch.no_grad():
                for batch in loop_val:
                    preds = model(batch["input_ids"].to(device), batch["attention_mask"].to(device))
                    val_logits.append(preds.cpu())
            loop_val.close()

            val_logits = torch.cat(val_logits)
            val_preds = torch.argmax(val_logits, 1).numpy()
            acc = accuracy_score(train_labels[val_idx], val_preds)
            print(f"Epoch {epoch+1} | Val Acc: {acc:.4f}")

            if acc > best_acc:
                best_acc = acc
                patience = 0
                best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            else:
                patience += 1
                if patience >= PATIENCE:
                    print("⏹ Early stopping")
                    break

        # ========== SAVE BEST MODEL ==========
        save_path = os.path.join(SAVE_DIR, f"{model_name.replace('/', '_')}_fold{fold+1}.pt")
        torch.save(best_state, save_path)
        print("✅ Saved:", save_path)

        # ========== DEV PREDICTION ==========
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
        model.eval()
        dev_logits_fold = []
        loop_dev = tqdm(dev_loader, desc=f"Dev Prediction Fold {fold+1}", leave=False)
        with torch.no_grad():
            for batch in loop_dev:
                preds = model(batch["input_ids"].to(device), batch["attention_mask"].to(device))
                dev_logits_fold.append(preds.cpu())
        loop_dev.close()
        dev_logits_all_folds.append(torch.cat(dev_logits_fold))

    dev_logits_avg = torch.stack(dev_logits_all_folds).mean(dim=0)
    return dev_logits_avg

# =========================
# TRAIN BOTH MODELS
# =========================
xlmr_dev_logits  = train_kfold_and_predict_dev("xlm-roberta-base")
indic_dev_logits = train_kfold_and_predict_dev("ai4bharat/IndicBERTv2-MLM-only")

# =========================
# WEIGHTED ENSEMBLE ON DEV
# =========================
dev_labels = dev_df["label_id"].values
pred_x = torch.argmax(xlmr_dev_logits, 1).numpy()
pred_i = torch.argmax(indic_dev_logits, 1).numpy()
acc_x = accuracy_score(dev_labels, pred_x)
acc_i = accuracy_score(dev_labels, pred_i)
w_x = acc_x / (acc_x + acc_i)
w_i = acc_i / (acc_x + acc_i)
final_dev_logits = (w_x * xlmr_dev_logits) + (w_i * indic_dev_logits)
final_dev_preds  = torch.argmax(final_dev_logits, 1).numpy()

print("\n📌 DEV RESULTS")
print(f"XLM-R Dev Acc:     {acc_x:.4f}")
print(f"IndicBERT Dev Acc: {acc_i:.4f}")
print(f"Ensemble Weights → w_x={w_x:.3f}, w_i={w_i:.3f}")
print("\n📊 FINAL ENSEMBLE DEV RESULT")
print("Accuracy:", accuracy_score(dev_labels, final_dev_preds))
print(classification_report(dev_labels, final_dev_preds, target_names=list(id2label.values())))

# =========================
# PREDICT TEST SET
# =========================
def predict_test(model_name, weight):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    test_ds = TestDataset(test_df["content"].tolist(), tokenizer)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)
    fold_logits_all = []

    for fold in range(1, N_FOLDS+1):
        model_path = os.path.join(SAVE_DIR, f"{model_name.replace('/', '_')}_fold{fold}.pt")
        model = TransformerClassifier(model_name).to(device)
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict({k: v.to(device) for k, v in state_dict.items()})
        model.eval()

        fold_logits = []
        loop_test = tqdm(test_loader, desc=f"Test Prediction Fold {fold}", leave=False)
        with torch.no_grad():
            for batch in loop_test:
                preds = model(batch["input_ids"].to(device), batch["attention_mask"].to(device))
                fold_logits.append(preds.cpu())
        loop_test.close()
        fold_logits_all.append(torch.cat(fold_logits))

    logits_avg = torch.stack(fold_logits_all).mean(dim=0)
    return weight * logits_avg

xlmr_test_logits  = predict_test("xlm-roberta-base", w_x)
indic_test_logits = predict_test("ai4bharat/IndicBERTv2-MLM-only", w_i)
final_test_logits = xlmr_test_logits + indic_test_logits
final_test_preds  = torch.argmax(final_test_logits, 1).numpy()

test_df["predicted_label"] = [id2label[i] for i in final_test_preds]
test_df[["content", "predicted_label"]].to_csv("submission3.csv", index=False)
print("\n✅ Test predictions saved to submission3.csv")
