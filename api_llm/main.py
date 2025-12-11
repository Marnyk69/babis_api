import os
import re
import numpy as np
import pandas as pd
import torch

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from datasets import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sentence_transformers import SentenceTransformer


# ---------------------------
# Konfigurace
# ---------------------------
MODEL_DIR = "robeczech_babis_freeze_last2"  # kde se uloží nebo odkud se načte model
HF_MODEL_NAME = "ufal/robeczech-base"
MAX_LENGTH = 96
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------
# FastAPI app
# ---------------------------
app = FastAPI(title="IsItAndrej")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # v produkci nastav konkrétní doménu
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic model pro request body
class PredictRequest(BaseModel):
    text: str

# ---------------------------
# Globální proměnné (nastavené při startu)
# ---------------------------
tokenizer: AutoTokenizer = None
model: AutoModelForSequenceClassification = None
# ---------------------------
# Utility: metriky
# ---------------------------
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    prec, rec, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary", zero_division=0
    )
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}


# ---------------------------
# Funkce pro trénink / přípravu dat (volitelné)
# ---------------------------
def prepare_and_train_if_needed():
    """
    Pokud v cwd existuje MODEL_DIR -> načteme model+tokenizer odtamtud.
    Jinak: pokud existuje politci_vyroky.csv, spustíme trénink (kopie tvého pipeline).
    Po tréninku uložíme model do MODEL_DIR.
    """
    global tokenizer, model

    # pokud existuje uložený model, načteme jej
    if os.path.isdir(MODEL_DIR) and os.path.exists(os.path.join(MODEL_DIR, "pytorch_model.bin")):
        tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
        model.to(DEVICE)
        return

    # jinak se pokusíme trénovat pokud máme CSV dataset
    csv_path = "politici_vyroky.csv"
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Model neexistuje (nebyl nalezen '{MODEL_DIR}') a dataset '{csv_path}' chybí. "
            "Nahraj dataset nebo připrav model v adresáři."
        )

    # === načtení tokenizeru a modelu z HF (základ) ===
    tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(HF_MODEL_NAME, num_labels=2)

    # === načtení a předzpracování dat ===
    df = pd.read_csv(csv_path)
    # očekáváme sloupce: ['label','babis_label','text'] (jak v původním kódu)
    df_temp = df[['label', 'babis_label', 'text']].copy()

    # split věty — tvůj pattern (počítá diakritiku)
    pattern = r'(?<=[.!?])\s+(?=[A-ZÁČĎÉĚÍŇÓŘŠŤÚŮÝŽ])'
    df_processed_sentences = df_temp.assign(
        text_sentence=df_temp['text'].astype(str).str.split(pattern)
    ).explode('text_sentence')
    df_processed_sentences['text_sentence'] = df_processed_sentences['text_sentence'].str.strip()
    df_processed_sentences = df_processed_sentences[df_processed_sentences['text_sentence'] != '']
    df_sentences = df_processed_sentences[['text_sentence', 'label', 'babis_label']].copy()

    # split trénink/val/test se stratifikací podle targetu
    train_df, temp_df = train_test_split(
        df_sentences,
        test_size=0.2,
        stratify=df_sentences['babis_label'],
        random_state=42,
    )
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.5,
        stratify=temp_df['babis_label'],
        random_state=42,
    )

    train_dataset = Dataset.from_pandas(train_df[['text_sentence', 'babis_label']].reset_index(drop=True))
    val_dataset = Dataset.from_pandas(val_df[['text_sentence', 'babis_label']].reset_index(drop=True))
    test_dataset = Dataset.from_pandas(test_df[['text_sentence', 'babis_label']].reset_index(drop=True))

    def tokenize_batch(batch):
        return tokenizer(
            batch["text_sentence"],
            padding="max_length",
            truncation=True,
            max_length=MAX_LENGTH,
        )

    train_tokenized = train_dataset.map(tokenize_batch, batched=True)
    val_tokenized = val_dataset.map(tokenize_batch, batched=True)
    test_tokenized = test_dataset.map(tokenize_batch, batched=True)

    train_tokenized = train_tokenized.rename_column("babis_label", "labels")
    val_tokenized = val_tokenized.rename_column("babis_label", "labels")
    test_tokenized = test_tokenized.rename_column("babis_label", "labels")

    train_tokenized.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    val_tokenized.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    test_tokenized.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    os.environ["WANDB_DISABLED"] = "true"

    # freeze většiny vrstev a odfreeze posledních dvou encoder vrstev (jak jsi měl)
    if hasattr(model, "roberta"):
        for param in model.roberta.parameters():
            param.requires_grad = False
        # poslední dvě vrstvy - bezpečně kontrolujeme existence
        try:
            for layer in model.roberta.encoder.layer[-2:]:
                for param in layer.parameters():
                    param.requires_grad = True
        except Exception:
            # pokud struktura není přesně stejná, ignoruj
            pass

    # classifier by měl být trénovatelný
    try:
        for param in model.classifier.parameters():
            param.requires_grad = True
    except Exception:
        pass

    training_args = TrainingArguments(
        output_dir=MODEL_DIR,
        learning_rate=3e-5,
        weight_decay=0.01,
        num_train_epochs=2,
        per_device_train_batch_size=16,  # zmenšeno defaultně, 64 může být příliš mnoho pro CPU/GPU
        per_device_eval_batch_size=32,
        logging_steps=100,
        save_strategy="epoch",
        evaluation_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized,
        eval_dataset=val_tokenized,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    # TRAIN
    trainer.train()

    # Uložíme finální model do MODEL_DIR pro pozdější načtení
    trainer.save_model(MODEL_DIR)
    tokenizer.save_pretrained(MODEL_DIR)

    # přesun modelu na device
    model.to(DEVICE)


# ---------------------------
# Predikční funkce
# ---------------------------
def predict_babis(text: str) -> Dict[str, Any]:
    """
    Vrací: { label_id, label_name, probability, probs_dict }
    """
    global tokenizer, model
    if tokenizer is None or model is None:
        raise RuntimeError("Model/tokenizer nejsou inicializovány")

    label_names = {0: "Non-Babiš", 1: "Babiš"}
    model.eval()
    model.to(DEVICE)

    # tokenizovat (batch velikost 1)
    inputs = tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH,
    )
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits  # shape (1, num_labels)
        probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]

    pred_id = int(np.argmax(probs))
    pred_prob = float(probs[pred_id])
    probs_dict = {label_names[i]: float(probs[i]) for i in range(len(probs))}

    return {
        "label_id": pred_id,
        "label_name": label_names[pred_id],
        "probability": pred_prob,
        "probs": probs_dict,
    }

from sentence_transformers import SentenceTransformer
import numpy as np
import pickle

embedder: SentenceTransformer = None
babis_sentences: list[str] = []
babis_embeddings: np.ndarray = None


def prepare_babis_embedding_database(df_sentences: pd.DataFrame, cache_path="babis_embeddings.pkl"):
    """
    Vytvoří embeddingy všech vět, kde babis_label == 1.
    Pokud existuje cache, načte ji.
    """
    global embedder, babis_sentences, babis_embeddings

    embedder = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

    # Pokud existuje cache → pouze načteme
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            data = pickle.load(f)
            babis_sentences = data["sentences"]
            babis_embeddings = data["embeddings"]
        print(f"Načteny embeddingy Babišových vět ({len(babis_sentences)} vět).")
        return

    # Jinak vytvoříme
    df_babis = df_sentences[df_sentences["babis_label"] == 1]
    babis_sentences = df_babis["text_sentence"].tolist()

    print(f"Vytvářím embeddingy pro {len(babis_sentences)} vět Andreje Babiše...")

    babis_embeddings = embedder.encode(babis_sentences, convert_to_numpy=True)

    # uložit cache
    with open(cache_path, "wb") as f:
        pickle.dump(
            {"sentences": babis_sentences, "embeddings": babis_embeddings},
            f
        )


def recommend_babis_sentence(input_text: str, top_k=1):
    """
    Vrátí nejbližší Babišovu větu podle embeddingové podobnosti.
    """
    global embedder, babis_sentences, babis_embeddings

    if embedder is None or babis_embeddings is None:
        return None

    query_emb = embedder.encode([input_text], convert_to_numpy=True)[0]

    similarities = babis_embeddings @ query_emb
    top_idx = similarities.argsort()[-top_k:][::-1]

    print("Babiš vět:", len(babis_sentences))
    print("Emb shape:", None if babis_embeddings is None else babis_embeddings.shape)

    return [(babis_sentences[i], float(similarities[i])) for i in top_idx]

# ---------------------------
# Startup event - inicializace (spustí se když běží aplikace)
# ---------------------------
@app.on_event("startup")
def startup_event():
    try:
        prepare_and_train_if_needed()
        print("Model a tokenizer připraveny.")
        # PO načtení / tréninku modelu potřebujeme mít k dispozici df_sentences
        try:
            df = pd.read_csv("politici_vyroky.csv")
            df_temp = df[['label', 'babis_label', 'text']].copy()

            pattern = r'(?<=[.!?])\s+(?=[A-ZÁČĎÉĚÍŇÓŘŠŤÚŮÝŽ])'
            df_processed = df_temp.assign(
                text_sentence=df_temp["text"].astype(str).str.split(pattern)
            ).explode("text_sentence")
            df_processed["text_sentence"] = df_processed["text_sentence"].str.strip()
            df_processed = df_processed[df_processed["text_sentence"] != ""]

            # vytvoří embedding databázi
            prepare_babis_embedding_database(df_processed)

        except Exception as e:
            print("Embedding databázi nelze vytvořit:", e)

    except Exception as e:
        # v případě chyb v startupu chceme aby to bylo zřejmé v logu, ale aplikace může stále běžet
        # pokud preferuješ strict behavior, můžeš tady re-raise
        print("Warning při inicializaci modelu:", str(e))


# ---------------------------
# Endpoint
# ---------------------------
from fastapi import Form

@app.post("/api/process")
def process(text: str = Form(...)):
    try:
        if not isinstance(text, str) or text.strip() == "":
            raise HTTPException(status_code=400, detail="Pole 'text' musí být neprázdný řetězec.")

        result = predict_babis(text)

        # doporučená věta
        recommendations = recommend_babis_sentence(text, top_k=1)
        suggested = recommendations[0][0] if recommendations else None

        return {
            "input": text,
            "prediction": {
                "label_id": result["label_id"],
                "label_name": result["label_name"],
                "probability": round(result["probability"], 4),
                "probabilities": {k: round(v, 4) for k, v in result["probs"].items()},
            },
            "suggested_babis_sentence": suggested
        }


    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
