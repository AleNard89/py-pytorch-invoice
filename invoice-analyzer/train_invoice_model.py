"""
Addestramento del modello LayoutLM (v1/v3) per estrazione dati da fatture.
Supporta apprendimento continuo/incrementale.
"""

import os
import sys
import json
import torch
import pandas as pd
import numpy as np
import traceback
import warnings
import datetime
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from transformers import TrainingArguments, Trainer
import pytesseract
from pdf2image import convert_from_path
from PIL import Image

from config import CONFIG, LABELS, id2label, label2id
from utils import normalize_box
from model import load_model_and_tokenizer, detect_model_version


class InvoiceDataset(Dataset):
    def __init__(self, pdf_paths, annotations_df, tokenizer_or_processor, version="v1",
                 poppler_path=None, tesseract_path=None, max_length=512):
        self.pdf_paths = pdf_paths
        self.annotations_df = annotations_df
        self.tokenizer_or_processor = tokenizer_or_processor
        self.version = version
        self.poppler_path = poppler_path or CONFIG["POPPLER_PATH"]
        self.tesseract_path = tesseract_path or CONFIG["TESSERACT_PATH"]
        self.max_length = max_length
        pytesseract.pytesseract.tesseract_cmd = self.tesseract_path
        self.examples = self._prepare_examples()

    def _prepare_examples(self):
        examples = []
        grouped_annotations = self.annotations_df.groupby(["image_id", "page_num"])

        for (image_id, page_num), group in grouped_annotations:
            try:
                pdf_path = self.pdf_paths[image_id] if image_id < len(self.pdf_paths) else self.pdf_paths[0]
                if image_id >= len(self.pdf_paths):
                    print(f"ATTENZIONE: image_id {image_id} fuori range. Uso il primo PDF.")

                print(f"Elaborazione PDF {pdf_path} per image_id {image_id}, pagina {page_num}")
                images = convert_from_path(
                    str(pdf_path), dpi=CONFIG["PDF_DPI"], poppler_path=self.poppler_path,
                    first_page=page_num, last_page=page_num, timeout=CONFIG["PDF_TIMEOUT"]
                )
                if not images:
                    print(f"Nessuna immagine estratta da {pdf_path} pagina {page_num}")
                    continue
                image = images[0]
                width, height = image.size

                words, word_level_boxes, word_labels = [], [], []
                for _, row in group.iterrows():
                    word = str(row["word"]).strip()
                    if not word:
                        continue
                    try:
                        x1 = max(0, int(row["x1"]))
                        y1 = max(0, int(row["y1"]))
                        x2 = min(width, int(row["x2"]))
                        y2 = min(height, int(row["y2"]))
                        if (x2 - x1) < 1 or (y2 - y1) < 1:
                            continue
                        box = [x1, y1, x2, y2]
                        normalized_box = normalize_box(box, width, height)
                        label = row["label"]
                        if label not in label2id:
                            label = "O"
                        words.append(word)
                        word_level_boxes.append(normalized_box)
                        word_labels.append(label2id[label])
                    except (ValueError, TypeError):
                        continue

                if not words:
                    continue

                if self.version == "v3":
                    tokenized_inputs = self._encode_v3(words, word_level_boxes, image)
                else:
                    tokenized_inputs = self._encode_v1(words, word_level_boxes)

                if tokenized_inputs is None:
                    continue

                labels = torch.ones(tokenized_inputs["input_ids"].shape, dtype=torch.long) * -100
                word_ids_for_labels = tokenized_inputs.word_ids()

                previous_word_idx = None
                for token_idx, current_word_idx in enumerate(word_ids_for_labels):
                    if current_word_idx is None:
                        continue
                    if current_word_idx != previous_word_idx:
                        if current_word_idx < len(word_labels):
                            labels[0, token_idx] = word_labels[current_word_idx]
                    else:
                        if current_word_idx < len(word_labels):
                            label_id = word_labels[current_word_idx]
                            if id2label[label_id].startswith("B-"):
                                i_label = "I-" + id2label[label_id][2:]
                                labels[0, token_idx] = label2id.get(i_label, label_id)
                            else:
                                labels[0, token_idx] = label_id
                    previous_word_idx = current_word_idx

                tokenized_inputs["labels"] = labels
                inputs = {k: v.squeeze(0) for k, v in tokenized_inputs.items()}
                examples.append(inputs)
                print(f"Esempio creato: {len(words)} parole ({self.version}).")

            except Exception as e_page:
                print(f"Errore elaborazione PDF {pdf_path} pagina {page_num}: {e_page}")
                traceback.print_exc()
                continue

        print(f"Totale esempi creati: {len(examples)}")
        if not examples:
            raise ValueError("Nessun esempio valido creato.")
        return examples

    def _encode_v3(self, words, boxes, image):
        """Encoding per LayoutLMv3: usa processor con immagine."""
        try:
            tokenized = self.tokenizer_or_processor(
                image, words, boxes=boxes,
                padding="max_length", truncation=True,
                max_length=self.max_length, return_tensors="pt",
            )
            return tokenized
        except Exception as e:
            print(f"Errore encoding v3: {e}")
            return None

    def _encode_v1(self, words, boxes):
        """Encoding per LayoutLM v1: tokenizer + bbox manuale."""
        try:
            tokenized = self.tokenizer_or_processor(
                words, boxes=boxes,
                padding="max_length", truncation=True,
                max_length=self.max_length, return_tensors="pt",
                is_split_into_words=True,
            )
        except TypeError:
            tokenized = self.tokenizer_or_processor(
                words, padding="max_length", truncation=True,
                max_length=self.max_length, return_tensors="pt",
                is_split_into_words=True,
            )
            aligned_boxes = []
            word_ids = tokenized.word_ids()
            tok = self.tokenizer_or_processor
            for i, tid in enumerate(tokenized["input_ids"][0]):
                if tid in (tok.cls_token_id, tok.sep_token_id, tok.pad_token_id):
                    aligned_boxes.append([0, 0, 0, 0])
                else:
                    widx = word_ids[i]
                    if widx is not None and widx < len(boxes):
                        aligned_boxes.append(boxes[widx])
                    else:
                        aligned_boxes.append([0, 0, 0, 0])
            if len(aligned_boxes) < self.max_length:
                aligned_boxes.extend([[0, 0, 0, 0]] * (self.max_length - len(aligned_boxes)))
            tokenized["bbox"] = torch.tensor([aligned_boxes[:self.max_length]], dtype=torch.long)
        return tokenized

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Addestramento modello LayoutLM per fatture")
    parser.add_argument("--annotations", required=True, help="Percorso al file Excel con annotazioni")
    parser.add_argument("--pdfs_dir", default="input_data", help="Directory contenente i file PDF")
    parser.add_argument("--model_name_or_path", default=CONFIG.get("BASE_MODEL_DIR", "models/layoutlmv3-base"),
                        help="Modello pre-addestrato o percorso locale")
    parser.add_argument("--output_dir", default=CONFIG.get("TRAINED_MODEL_DIR", "models/invoice_model_v3"),
                        help="Directory di output per il modello addestrato")
    parser.add_argument("--epochs", type=int, default=CONFIG["NUM_EPOCHS"], help="Numero di epoche")
    parser.add_argument("--batch_size", type=int, default=CONFIG["BATCH_SIZE"], help="Dimensione del batch")
    parser.add_argument("--learning_rate", type=float, default=CONFIG["LEARNING_RATE"], help="Learning rate")
    parser.add_argument("--max_seq_length", type=int, default=CONFIG["MAX_SEQ_LENGTH"], help="Lunghezza massima sequenza")
    parser.add_argument("--use_local_model", type=bool, default=CONFIG.get("USE_LOCAL_MODEL", True))
    parser.add_argument("--continue_from", default=None, help="Percorso al modello da cui continuare")

    args = parser.parse_args()
    start_time = datetime.datetime.now()
    print(f"Inizio esecuzione: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

    annotations_path = Path(args.annotations)
    if not annotations_path.exists():
        raise FileNotFoundError(f"File annotazioni non trovato: {annotations_path}")

    print(f"Caricamento annotazioni da {annotations_path}...")
    annotations_df = pd.read_excel(annotations_path)
    print(f"Caricate {len(annotations_df)} righe di annotazioni")

    required_columns = ["image_id", "page_num", "word", "x1", "y1", "x2", "y2", "label"]
    if not all(col in annotations_df.columns for col in required_columns):
        raise ValueError(f"Colonne mancanti. Richieste: {required_columns}")

    annotations_df = annotations_df.dropna(subset=required_columns)
    annotations_df["word"] = annotations_df["word"].astype(str).str.strip()
    annotations_df = annotations_df[annotations_df["word"] != ""]

    for col in ["x1", "y1", "x2", "y2", "image_id", "page_num"]:
        annotations_df[col] = pd.to_numeric(annotations_df[col], errors="coerce")
    annotations_df = annotations_df.dropna(subset=["x1", "y1", "x2", "y2", "image_id", "page_num"])
    annotations_df[["image_id", "page_num"]] = annotations_df[["image_id", "page_num"]].astype(int)

    unique_labels = annotations_df["label"].unique()
    for label in unique_labels:
        if label not in label2id:
            print(f"ATTENZIONE: Etichetta '{label}' non in LABELS. Trattata come 'O'.")
            annotations_df["label"] = annotations_df["label"].replace(label, "O")

    print("Distribuzione delle etichette:")
    for label, count in annotations_df["label"].value_counts().items():
        print(f"  - {label}: {count}")

    pdfs_dir = Path(args.pdfs_dir)
    pdf_files = sorted(pdfs_dir.glob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(f"Nessun file PDF trovato in {pdfs_dir}")
    for i, pdf_path in enumerate(pdf_files):
        print(f"  {i}: {pdf_path}")

    # Determina la sorgente del modello/tokenizer
    source = args.continue_from if args.continue_from else args.model_name_or_path
    model, tokenizer_or_processor, version = load_model_and_tokenizer(source, for_training=True)
    print(f"Modello {version} caricato da: {source}")

    # Train/validation split
    from sklearn.model_selection import train_test_split
    has_eval_data = False
    eval_dataset = None

    if len(annotations_df) >= 10:
        try:
            label_counts = annotations_df["label"].value_counts()
            can_stratify = all(count >= 2 for count in label_counts)
            train_annotations, eval_annotations = train_test_split(
                annotations_df, test_size=0.2, random_state=42,
                stratify=annotations_df["label"] if can_stratify else None,
            )
            has_eval_data = True
            print(f"Split: {len(train_annotations)} train, {len(eval_annotations)} eval")
        except Exception:
            train_annotations = annotations_df
    else:
        train_annotations = annotations_df

    print("Creazione dataset di addestramento...")
    train_dataset = InvoiceDataset(
        pdf_paths=pdf_files, annotations_df=train_annotations,
        tokenizer_or_processor=tokenizer_or_processor, version=version,
        max_length=args.max_seq_length,
    )
    print(f"Dataset training: {len(train_dataset)} esempi")

    if has_eval_data:
        eval_dataset = InvoiceDataset(
            pdf_paths=pdf_files, annotations_df=eval_annotations,
            tokenizer_or_processor=tokenizer_or_processor, version=version,
            max_length=args.max_seq_length,
        )
        print(f"Dataset validazione: {len(eval_dataset)} esempi")

    lr = args.learning_rate * 0.7 if args.continue_from else args.learning_rate

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=lr,
        logging_dir=f"{args.output_dir}/logs",
        logging_steps=10,
        save_steps=500,
        save_total_limit=2,
        fp16=torch.cuda.is_available(),
    )

    trainer_kwargs = {"model": model, "args": training_args, "train_dataset": train_dataset}
    if has_eval_data and eval_dataset:
        trainer_kwargs["eval_dataset"] = eval_dataset
    trainer = Trainer(**trainer_kwargs)

    print("\nInizio addestramento...")
    try:
        trainer.train()
        print("\n✅ Addestramento completato!")
    except Exception as e:
        print(f"\n❌ Errore addestramento: {e}")
        traceback.print_exc()

    print(f"\nSalvataggio modello in {args.output_dir}")
    try:
        trainer.save_model(args.output_dir)
        if version == "v3":
            tokenizer_or_processor.save_pretrained(args.output_dir)
        else:
            tokenizer_or_processor.save_pretrained(args.output_dir)

        labels_path = Path(args.output_dir) / "labels.json"
        with open(labels_path, "w") as f:
            json.dump(LABELS, f, indent=2)
        print(f"Etichette salvate in {labels_path}")

        readme_content = f"""# Modello Estrazione Dati Fatture (LayoutLM {version})

- Data: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- Modello base: {args.model_name_or_path}
- Versione: {version}
- Epoche: {args.epochs}
- Batch size: {args.batch_size}
- Learning rate: {lr}
- Annotazioni: {annotations_path}
- Esempi training: {len(train_dataset)}
- Esempi validazione: {len(eval_dataset) if eval_dataset else 0}
{f'- Continuato da: {args.continue_from}' if args.continue_from else ''}

Etichette: {json.dumps(LABELS, indent=2)}

Distribuzione: {annotations_df["label"].value_counts().to_dict()}
"""
        with open(Path(args.output_dir) / "README.md", "w") as f:
            f.write(readme_content)

        history_file = Path(args.output_dir) / "training_history.json"
        history_data = {
            "date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "version": version,
            "annotations_file": str(annotations_path),
            "examples_count": len(annotations_df),
            "label_distribution": annotations_df["label"].value_counts().to_dict(),
            "continued_from": str(args.continue_from) if args.continue_from else None,
            "epochs": args.epochs,
            "learning_rate": lr,
            "batch_size": args.batch_size,
        }
        existing_history = []
        if history_file.exists():
            try:
                with open(history_file) as h:
                    existing_history = json.load(h)
                    if not isinstance(existing_history, list):
                        existing_history = [existing_history]
            except Exception:
                pass
        existing_history.append(history_data)
        with open(history_file, "w") as h:
            json.dump(existing_history, h, indent=2)
        print(f"Storia addestramento aggiornata in {history_file}")

    except Exception as e:
        print(f"Errore nel salvataggio: {e}")
        traceback.print_exc()

    # Verifica
    try:
        print("Verifica modello salvato...")
        reloaded_model, reloaded_tok, reloaded_ver = load_model_and_tokenizer(args.output_dir)
        print(f"✓ Modello {reloaded_ver} ricaricato con successo da: {args.output_dir}")
    except Exception as e:
        print(f"✗ Errore verifica: {e}")

    end_time = datetime.datetime.now()
    print(f"Fine esecuzione: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Tempo totale: {end_time - start_time}")


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning, module="torchvision")
    warnings.filterwarnings("ignore", category=UserWarning, module="torch.utils.data.dataloader")
    main()
