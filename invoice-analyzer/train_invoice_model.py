"""
Addestramento del modello LayoutLM (v1) per estrazione dati da fatture
Versione compatibile con Windows che utilizza le librerie Hugging Face
Versione modificata con supporto per apprendimento continuo/incrementale
"""

import os
import sys
import json
import torch
import platform
import shutil
import pandas as pd
import numpy as np
import traceback
import warnings
import datetime
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from transformers import (
    LayoutLMForTokenClassification,
    LayoutLMTokenizer, 
    LayoutLMConfig,
    TrainingArguments,
    Trainer
)
import pytesseract
from pdf2image import convert_from_path
from PIL import Image

def _detect_path(binary_name, fallbacks=None):
    path = shutil.which(binary_name)
    if path:
        return path
    if fallbacks:
        for fb in fallbacks:
            if Path(fb).exists():
                return fb
    return binary_name

DEFAULT_CONFIG = {
    "POPPLER_PATH": str(Path(_detect_path("pdftoppm", ["/opt/homebrew/bin/pdftoppm", "/usr/bin/pdftoppm"])).parent),
    "TESSERACT_PATH": _detect_path("tesseract", ["/opt/homebrew/bin/tesseract", "/usr/bin/tesseract"]),
    "BASE_MODEL_DIR": "models/layoutlm-base-uncased",
    "TRAINED_MODEL_DIR": "models/invoice_model",
    "RESULTS_DIR": "output_data/results",
    "PDF_DPI": 300,
    "BATCH_SIZE": 2,
    "NUM_EPOCHS": 5,
    "LEARNING_RATE": 2e-5,
    "MAX_PAGES": 2,
    "MAX_SEQ_LENGTH": 512,
    "PDF_TIMEOUT": 120
}


######################################################################################################################################################

# Carica configurazione da file se esiste
def load_config():
    config = DEFAULT_CONFIG.copy()
    config_file = Path("invoice_config.json")
    
    if config_file.exists():
        try:
            with open(config_file, "r") as f:
                user_config = json.load(f)
                print("Configurazione predefinita:")
                for key, value in config.items():
                    print(f"  - {key}: {value}")
                
                if user_config:
                    print("\nValori da file di configurazione:")
                    for key, value in user_config.items():
                        print(f"  - {key}: {value}")
                    config.update(user_config)
                
            print(f"Configurazione caricata da {config_file}")
        except Exception as e:
            print(f"Errore nel caricamento della configurazione: {e}")
    
    for key in ["POPPLER_PATH", "TESSERACT_PATH"]:
        path = Path(config[key])
        if not path.exists():
            print(f"ATTENZIONE: Il percorso {key}={path} non esiste!")
    
    tesseract_path = Path(config["TESSERACT_PATH"])
    if tesseract_path.is_dir():
        tesseract_bin = "tesseract.exe" if platform.system() == "Windows" else "tesseract"
        config["TESSERACT_PATH"] = str(tesseract_path / tesseract_bin)
        print(f"Percorso Tesseract aggiornato a: {config['TESSERACT_PATH']}")
    
    return config

######################################################################################################################################################

CONFIG = load_config()

LABELS = [
    "O", "B-VENDOR", "I-VENDOR", "B-CUSTOMER", "I-CUSTOMER", "B-DATE", "I-DATE",
    "B-TOTAL", "I-TOTAL", "B-ITEM", "I-ITEM", "B-QUANTITY", "I-QUANTITY",
    "B-PRICE", "I-PRICE", "B-INVOICE_NUMBER", "I-INVOICE_NUMBER"
]
id2label = {i: label for i, label in enumerate(LABELS)}
label2id = {label: i for i, label in enumerate(LABELS)}


######################################################################################################################################################

def normalize_box(box, width, height):
    if width <= 0 or height <= 0:
        raise ValueError(f"Dimensioni dell'immagine non valide: {width}x{height}")
    x1 = min(max(0, box[0]), width - 1)
    y1 = min(max(0, box[1]), height - 1)
    x2 = min(max(x1 + 1, box[2]), width)
    y2 = min(max(y1 + 1, box[3]), height)
    normalized_box = [
        int(1000 * (x1 / width)),
        int(1000 * (y1 / height)),
        int(1000 * (x2 / width)),
        int(1000 * (y2 / height)),
    ]
    for i, coord in enumerate(normalized_box):
        if coord < 0 or coord > 1000:
            print(f"ATTENZIONE: Coordinata normalizzata fuori range: {coord}. Valore corretto.")
            normalized_box[i] = max(0, min(1000, coord))
    return normalized_box

######################################################################################################################################################

class InvoiceDataset(Dataset):
    def __init__(self, pdf_paths, annotations_df, tokenizer, poppler_path=None, tesseract_path=None, max_length=512):
        self.pdf_paths = pdf_paths
        self.annotations_df = annotations_df
        self.tokenizer = tokenizer
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
                    if not word: continue
                    try:
                        x1, y1, x2, y2 = max(0, int(row["x1"])), max(0, int(row["y1"])), min(width, int(row["x2"])), min(height, int(row["y2"]))
                        if (x2 - x1) < 1 or (y2 - y1) < 1: # Box non valide o troppo piccole
                            print(f"Box non valida o troppo piccola per '{word}': [{x1},{y1},{x2},{y2}]. Salto.")
                            continue
                        box = [x1, y1, x2, y2]
                        normalized_box = normalize_box(box, width, height)
                        label = row["label"]
                        if label not in label2id:
                            print(f"Etichetta non valida: {label}. Uso 'O'.")
                            label = "O"
                        words.append(word)
                        word_level_boxes.append(normalized_box)
                        word_labels.append(label2id[label])
                    except (ValueError, TypeError) as e:
                        print(f"Errore conversione coordinate per '{word}': {e}")
                        continue
                
                if not words:
                    print(f"Nessuna parola valida da {pdf_path} pagina {page_num}")
                    continue
                
                tokenized_inputs = None
                try:
                    print(f"Tentativo tokenizzazione con 'boxes' per {len(words)} parole.")
                    tokenized_inputs = self.tokenizer(
                        words,
                        boxes=word_level_boxes,
                        padding="max_length",
                        truncation=True,
                        max_length=self.max_length,
                        return_tensors="pt",
                        is_split_into_words=True
                    )
                    print("Tokenizzazione con 'boxes' riuscita.")
                except TypeError as e_tokenizer:
                    print(f"Errore tokenizzazione con 'boxes': {e_tokenizer}. Fallback a tokenizzazione manuale bbox.")
                    tokenized_inputs = self.tokenizer(
                        words,
                        padding="max_length",
                        truncation=True,
                        max_length=self.max_length,
                        return_tensors="pt",
                        is_split_into_words=True # Mantenuto per coerenza
                    )
                    # Fallback robusto per l'assegnazione delle bbox
                    # Questo blocco viene eseguito se l'argomento 'boxes' non è supportato
                    aligned_boxes = []
                    word_ids = tokenized_inputs.word_ids()
                    
                    for i, token_id in enumerate(tokenized_inputs["input_ids"][0]):
                        if token_id == self.tokenizer.cls_token_id or token_id == self.tokenizer.sep_token_id or token_id == self.tokenizer.pad_token_id:
                            aligned_boxes.append([0, 0, 0, 0]) # Box per token speciali
                        else:
                            word_idx = word_ids[i]
                            if word_idx is not None and word_idx < len(word_level_boxes):
                                aligned_boxes.append(word_level_boxes[word_idx])
                            else:
                                # Caso imprevisto, assegna una box di default
                                print(f"Attenzione: word_idx {word_idx} non valido per token {i}. Uso box di default.")
                                aligned_boxes.append([0,0,0,0])
                    
                    # Assicura che aligned_boxes abbia la lunghezza corretta (max_length)
                    if len(aligned_boxes) < self.max_length:
                        aligned_boxes.extend([[0,0,0,0]] * (self.max_length - len(aligned_boxes)))
                    aligned_boxes = aligned_boxes[:self.max_length]

                    tokenized_inputs["bbox"] = torch.tensor([aligned_boxes], dtype=torch.long)
                    print("Bbox aggiunte manualmente con fallback robusto.")

                # Creazione etichette token per token
                labels = torch.ones(tokenized_inputs["input_ids"].shape, dtype=torch.long) * -100 # Ignora per default
                word_ids_for_labels = tokenized_inputs.word_ids()

                previous_word_idx = None
                for token_idx, current_word_idx in enumerate(word_ids_for_labels):
                    if current_word_idx is None: # Token speciale
                        continue
                    if current_word_idx != previous_word_idx: # Inizio di una nuova parola
                        if current_word_idx < len(word_labels):
                             labels[0, token_idx] = word_labels[current_word_idx]
                        else:
                            print(f"Attenzione: current_word_idx {current_word_idx} fuori range per word_labels.")
                    else: # Stessa parola, token interno
                        if current_word_idx < len(word_labels):
                            label_id = word_labels[current_word_idx]
                            # Se l'etichetta originale era B-XXX, i token successivi diventano I-XXX
                            if id2label[label_id].startswith("B-"):
                                corresponding_i_label = "I-" + id2label[label_id][2:]
                                if corresponding_i_label in label2id:
                                    labels[0, token_idx] = label2id[corresponding_i_label]
                                else: # Se non esiste I-XXX, mantieni B-XXX (improbabile ma sicuro)
                                    labels[0, token_idx] = label_id 
                            else: # Se era già I-XXX o O, mantieni
                                labels[0, token_idx] = label_id
                        else:
                             print(f"Attenzione: current_word_idx {current_word_idx} fuori range per word_labels (token interno).")
                    previous_word_idx = current_word_idx
                
                tokenized_inputs["labels"] = labels
                inputs = {k: v.squeeze(0) for k, v in tokenized_inputs.items()}
                examples.append(inputs)
                print(f"Esempio creato: {len(words)} parole.")
            
            except Exception as e_page:
                print(f"Errore elaborazione PDF {pdf_path} pagina {page_num}: {e_page}")
                traceback.print_exc()
                continue
        
        print(f"Totale esempi creati: {len(examples)}")
        if not examples:
            raise ValueError("Nessun esempio valido creato. Controlla le annotazioni e i PDF.")
        return examples

    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        return self.examples[idx]

######################################################################################################################################################

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Addestramento modello LayoutLM per fatture")
    parser.add_argument("--annotations", required=True, help="Percorso al file Excel con annotazioni")
    parser.add_argument("--pdfs_dir", default="input_data", help="Directory contenente i file PDF")
    parser.add_argument("--model_name_or_path", default=CONFIG.get("BASE_MODEL_DIR", CONFIG.get("MODEL_DIR", "models/layoutlm-base-uncased")), help="Modello pre-addestrato o percorso locale")
    parser.add_argument("--output_dir", default=CONFIG.get("TRAINED_MODEL_DIR", CONFIG.get("OUTPUT_DIR", "models/invoice_model")), help="Directory di output per il modello addestrato")
    parser.add_argument("--epochs", type=int, default=CONFIG["NUM_EPOCHS"], help="Numero di epoche di addestramento")
    parser.add_argument("--batch_size", type=int, default=CONFIG["BATCH_SIZE"], help="Dimensione del batch")
    parser.add_argument("--learning_rate", type=float, default=CONFIG["LEARNING_RATE"], help="Tasso di apprendimento")
    parser.add_argument("--max_seq_length", type=int, default=CONFIG["MAX_SEQ_LENGTH"], help="Lunghezza massima sequenza")
    parser.add_argument("--use_local_model", type=bool, default=CONFIG.get("USE_LOCAL_MODEL", True), help="Usa modello locale")
    # NUOVO: Parametro per continuare l'addestramento da un modello esistente
    parser.add_argument("--continue_from", default=None, help="Percorso al modello da cui continuare l'addestramento")

    args = parser.parse_args()
    start_time = datetime.datetime.now()
    print(f"Inizio esecuzione: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

    # Carica annotazioni
    annotations_path = Path(args.annotations)
    if not annotations_path.exists():
        raise FileNotFoundError(f"File annotazioni non trovato: {annotations_path}")
    
    print(f"Caricamento annotazioni da {annotations_path}...")
    try:
        annotations_df = pd.read_excel(annotations_path)
        print(f"Caricate {len(annotations_df)} righe di annotazioni")
    except Exception as e:
        print(f"Errore nel caricamento del file Excel: {e}")
        sys.exit(1)

    # Pulisci e valida i dati
    print("Pulizia e validazione dei dati...")
    required_columns = ["image_id", "page_num", "word", "x1", "y1", "x2", "y2", "label"]
    if not all(col in annotations_df.columns for col in required_columns):
        raise ValueError(f"Colonne mancanti nel file di annotazioni. Richieste: {required_columns}")
    
    annotations_df = annotations_df.dropna(subset=required_columns)
    annotations_df["word"] = annotations_df["word"].astype(str).str.strip()
    annotations_df = annotations_df[annotations_df["word"] != ""]
    
    # Converti colonne numeriche e gestisci errori
    for col in ["x1", "y1", "x2", "y2", "image_id", "page_num"]:
        try:
            annotations_df[col] = pd.to_numeric(annotations_df[col], errors='coerce')
        except Exception as e:
            print(f"Errore nella conversione della colonna {col}: {e}")
            sys.exit(1)
    annotations_df = annotations_df.dropna(subset=["x1", "y1", "x2", "y2", "image_id", "page_num"])
    annotations_df[["image_id", "page_num"]] = annotations_df[["image_id", "page_num"]].astype(int)

    # Verifica etichette
    unique_labels = annotations_df["label"].unique()
    for label in unique_labels:
        if label not in label2id:
            print(f"ATTENZIONE: Etichetta '{label}' nelle annotazioni non è in LABELS. Sarà trattata come 'O'.")
            annotations_df["label"] = annotations_df["label"].replace(label, "O")
    
    print("Distribuzione delle etichette:")
    for label, count in annotations_df["label"].value_counts().items():
        print(f"  - {label}: {count} occorrenze")
    print(f"Numero di pagine uniche: {annotations_df['page_num'].nunique()}")
    print(f"Numero di image_id unici: {annotations_df['image_id'].nunique()}")

    # Ottieni percorsi PDF
    pdfs_dir = Path(args.pdfs_dir)
    if not pdfs_dir.is_dir():
        raise FileNotFoundError(f"Directory PDF non trovata: {pdfs_dir}")
    pdf_files = sorted([f for f in pdfs_dir.glob("*.pdf")])
    if not pdf_files:
        raise FileNotFoundError(f"Nessun file PDF trovato in {pdfs_dir}")
    print(f"Utilizzo di {len(pdf_files)} file PDF per l'addestramento:")
    for i, pdf_path in enumerate(pdf_files):
        print(f"  {i}: {pdf_path}")

    # Inizializza tokenizer
    # MODIFICATO: Determina la fonte del tokenizer (modello esistente o base)
    tokenizer_source = args.continue_from if args.continue_from else Path(args.model_name_or_path)
    print(f"Inizializzazione del tokenizer da: {tokenizer_source}")
    
    try:
        from transformers import LayoutLMTokenizerFast
        print(f"Caricamento del tokenizer locale da {tokenizer_source}...")
        print("Tentativo di caricamento del tokenizer fast...")
        tokenizer = LayoutLMTokenizerFast.from_pretrained(str(tokenizer_source), local_files_only=args.use_local_model)
        print("Tokenizer LayoutLM Fast caricato con successo")
    except (ImportError, Exception) as e_tok_fast:
        print(f"Impossibile caricare LayoutLMTokenizerFast: {e_tok_fast}. Fallback a LayoutLMTokenizer.")
        tokenizer = LayoutLMTokenizer.from_pretrained(str(tokenizer_source), local_files_only=args.use_local_model)
        print("Tokenizer LayoutLM (lento) caricato con successo")

    # NUOVO: Split training/validation
    from sklearn.model_selection import train_test_split
    
    # Dividi solo se abbiamo abbastanza dati (almeno 10 esempi)
    if len(annotations_df) >= 10:
        try:
            # Cerca di stratificare se possibile (ogni label deve avere almeno 2 esempi)
            label_counts = annotations_df["label"].value_counts()
            can_stratify = all(count >= 2 for count in label_counts)
            
            train_annotations, eval_annotations = train_test_split(
                annotations_df, 
                test_size=0.2, 
                random_state=42,
                stratify=annotations_df["label"] if can_stratify else None
            )
            print(f"Dati divisi in: {len(train_annotations)} esempi di training, {len(eval_annotations)} esempi di validazione")
            has_eval_data = True
        except Exception as e_split:
            print(f"Errore nella divisione dei dati: {e_split}. Usando tutti i dati per il training.")
            train_annotations = annotations_df
            eval_annotations = None
            has_eval_data = False
    else:
        print("Dataset troppo piccolo per la divisione. Usando tutti i dati per il training.")
        train_annotations = annotations_df
        eval_annotations = None
        has_eval_data = False

    # Crea dataset
    print("Creazione del dataset di addestramento...")
    train_dataset = InvoiceDataset(
        pdf_paths=pdf_files,
        annotations_df=train_annotations,
        tokenizer=tokenizer,
        max_length=args.max_seq_length
    )
    print(f"Dataset di training creato con successo: {len(train_dataset)} esempi")
    
    # Crea dataset di validazione se necessario
    eval_dataset = None
    if has_eval_data and eval_annotations is not None and len(eval_annotations) > 0:
        print("Creazione del dataset di validazione...")
        eval_dataset = InvoiceDataset(
            pdf_paths=pdf_files,
            annotations_df=eval_annotations,
            tokenizer=tokenizer,
            max_length=args.max_seq_length
        )
        print(f"Dataset di validazione creato con successo: {len(eval_dataset)} esempi")

    # MODIFICATO: Inizializza modello (da zero o continua l'addestramento)
    print(f"\nInizializzazione dell'addestramento...")
    
    if args.continue_from:
        continue_path = Path(args.continue_from)
        if not continue_path.exists():
            raise FileNotFoundError(f"Modello da continuare non trovato: {continue_path}")
            
        print(f"⭐ Continuando l'addestramento dal modello: {continue_path}")
        try:
            config = LayoutLMConfig.from_pretrained(str(continue_path), local_files_only=True)
            model = LayoutLMForTokenClassification.from_pretrained(str(continue_path), config=config, local_files_only=True)
            print(f"Modello caricato con successo da: {continue_path}")
            
            # Verifica che il modello esistente usi lo stesso schema di etichette
            # Questa verifica è importante per evitare problemi di incompatibilità
            model_id2label = model.config.id2label if hasattr(model.config, 'id2label') else {}
            if model_id2label and set(model_id2label.values()) != set(LABELS):
                print("⚠️ ATTENZIONE: Il modello caricato usa un insieme di etichette diverso!")
                print(f"  - Etichette nel modello: {sorted(model_id2label.values())}")
                print(f"  - Etichette correnti: {sorted(LABELS)}")
                print("Questo potrebbe causare problemi nell'addestramento incrementale.")
        except Exception as e_load:
            print(f"⚠️ Errore nel caricamento del modello esistente: {e_load}")
            print("Fallback all'inizializzazione da zero.")
            model_path = Path(args.model_name_or_path)
            config = LayoutLMConfig.from_pretrained(
                str(model_path), 
                num_labels=len(LABELS), 
                id2label=id2label, 
                label2id=label2id,
                local_files_only=args.use_local_model
            )
            model = LayoutLMForTokenClassification.from_pretrained(
                str(model_path), 
                config=config,
                local_files_only=args.use_local_model
            )
    else:
        model_path = Path(args.model_name_or_path)
        print(f"Inizializzazione del modello da: {model_path}")
        config = LayoutLMConfig.from_pretrained(
            str(model_path), 
            num_labels=len(LABELS), 
            id2label=id2label, 
            label2id=label2id,
            local_files_only=args.use_local_model
        )
        model = LayoutLMForTokenClassification.from_pretrained(
            str(model_path), 
            config=config,
            local_files_only=args.use_local_model
        )
        print(f"Modello base caricato da: {model_path}")

    # MODIFICATO: Argomenti di addestramento - aggiunti parametri di validazione se necessario

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.learning_rate * 0.7 if args.continue_from else args.learning_rate,
        logging_dir=f"{args.output_dir}/logs",
        logging_steps=10,
        save_steps=500,
        save_total_limit=2,
        fp16=torch.cuda.is_available()
        # Le seguenti opzioni non sono supportate nella tua versione, quindi le rimuoviamo:
        # evaluation_strategy="steps" if has_eval_data else "no",
        # eval_steps=20 if has_eval_data else None,
        # load_best_model_at_end=has_eval_data,
        # metric_for_best_model="eval_loss" if has_eval_data else None,
        # greater_is_better=False,
        # report_to=["tensorboard"]
    )

    # MODIFICATO: Trainer con supporto di validazione opzionale
    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": train_dataset,
    }
    
    if has_eval_data and eval_dataset is not None:
        trainer_kwargs["eval_dataset"] = eval_dataset
    
    trainer = Trainer(**trainer_kwargs)

    # Addestramento
    print("\nInizio addestramento del modello...")
    try:
        trainer.train()
        print("\n✅ Addestramento completato con successo!")
    except Exception as e_train:
        print(f"\n❌ Errore durante l'addestramento: {e_train}")
        traceback.print_exc()
        print("\nTentativo di salvare il modello comunque...")

    print(f"\nSalvataggio del modello in {args.output_dir}")
    try:
        trainer.save_model(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        
        # Salva informazioni sulle etichette
        labels_path = Path(args.output_dir) / "labels.json"
        with open(labels_path, "w") as f:
            json.dump(LABELS, f, indent=2)
        print(f"Informazioni sulle etichette salvate in {labels_path}")
        
        # Aggiorna o crea un file README per il modello
        if args.continue_from:
            readme_content = f"""
    # Modello di Estrazione Dati Fatture (Aggiornato)

    Modello LayoutLM addestrato per l'estrazione di informazioni da fatture.

    - Ultimo aggiornamento: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    - Modello base originale: {args.model_name_or_path}
    - Continuato da: {args.continue_from}
    - Epoche: {args.epochs}
    - Batch size: {args.batch_size}
    - Learning rate: {args.learning_rate}
    - File di annotazioni: {annotations_path}

    Numero di esempi di addestramento: {len(train_dataset)}
    Numero di esempi di validazione: {len(eval_dataset) if eval_dataset else 0}

    Etichette:
    {json.dumps(LABELS, indent=2)}

    Distribuzione delle etichette nell'ultimo aggiornamento:
    {annotations_df["label"].value_counts().to_dict()}
    """
        else:
            readme_content = f"""
    # Modello di Estrazione Dati Fatture

    Modello LayoutLM addestrato per l'estrazione di informazioni da fatture.

    - Data addestramento: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    - Modello base: {args.model_name_or_path}
    - Epoche: {args.epochs}
    - Batch size: {args.batch_size}
    - Learning rate: {args.learning_rate}
    - File di annotazioni: {annotations_path}

    Numero di esempi di addestramento: {len(train_dataset)}
    Numero di esempi di validazione: {len(eval_dataset) if eval_dataset else 0}

    Etichette:
    {json.dumps(LABELS, indent=2)}

    Distribuzione delle etichette:
    {annotations_df["label"].value_counts().to_dict()}
    """
        with open(Path(args.output_dir) / "README.md", "w") as f_readme:
            f_readme.write(readme_content)
        print(f"File README creato in {Path(args.output_dir) / 'README.md'}")
        
        # NUOVO: Salva la storia di addestramento in un file separato
        history_file = Path(args.output_dir) / "training_history.json"
        
        history_data = {
            "date": datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "annotations_file": str(annotations_path),
            "examples_count": len(annotations_df),
            "label_distribution": annotations_df["label"].value_counts().to_dict(),
            "continued_from": str(args.continue_from) if args.continue_from else None,
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "batch_size": args.batch_size
        }
        
        # Carica la storia esistente se il file esiste
        existing_history = []
        if history_file.exists():
            try:
                with open(history_file, "r") as h_file:
                    existing_history = json.load(h_file)
                    if not isinstance(existing_history, list):
                        existing_history = [existing_history]
            except:
                print("Impossibile caricare la storia di addestramento esistente. Creazione di un nuovo file.")
        
        # Aggiungi l'addestramento corrente alla storia
        existing_history.append(history_data)
        
        # Salva la storia aggiornata
        with open(history_file, "w") as h_file:
            json.dump(existing_history, h_file, indent=2)
        print(f"Storia di addestramento aggiornata in {history_file}")
            
    except Exception as e_save:
        print(f"Errore nel salvare il modello: {e_save}")
        traceback.print_exc()

    # Verifica compatibilità del modello salvato
    try:
        print("Verifica compatibilità del modello salvato...")
        reloaded_model = LayoutLMForTokenClassification.from_pretrained(args.output_dir)
        reloaded_tokenizer = LayoutLMTokenizerFast.from_pretrained(args.output_dir)
        print(f"✓ Modello e tokenizer ricaricati con successo da: {args.output_dir}")
    except Exception as e_reload:
        print(f"X Errore nel ricaricare il modello/tokenizer salvato: {e_reload}")
        traceback.print_exc()

    end_time = datetime.datetime.now()
    print(f"Fine esecuzione: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Tempo totale di esecuzione: {end_time - start_time}")

if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning, module='torchvision')
    warnings.filterwarnings("ignore", category=UserWarning, module='torch.utils.data.dataloader')
    main()















