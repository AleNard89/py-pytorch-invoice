"""
Pre-annotazione automatica: il modello suggerisce label per ogni parola OCR.
L'utente puo' poi caricare il file nell'annotation_tool per correggere.

Workflow:
  1. python preannotate.py --pdf input_data/nuova_fattura.pdf
  2. python annotation_tool.py  ->  Importa Pre-annotazioni
  3. Correggi le label errate, salva
  4. python train_invoice_model.py --annotations ... --continue_from models/invoice_model_v3
"""

import os
import sys
import torch
import warnings
import argparse
from pathlib import Path
from datetime import datetime

import pandas as pd

from config import CONFIG, id2label
from utils import normalize_box
from ocr import convert_pdf_to_images, process_image_with_ocr
from model import load_model_and_tokenizer
from invoice_inference import encode_page_v1, encode_page_v3, run_model

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"


def preannotate_pdf(pdf_path, model, tokenizer_or_processor, version):
    """Esegue il modello su un PDF e restituisce predizioni per ogni parola OCR."""
    images = convert_pdf_to_images(pdf_path)
    if not images:
        return []

    all_predictions = []

    for page_idx, image in enumerate(images):
        words, boxes, word_indices_ocr, ocr_full_result = process_image_with_ocr(image)
        if not words:
            continue

        normalized_boxes = [normalize_box(b, image.width, image.height) for b in boxes]

        if version == "v3":
            encoding = encode_page_v3(
                tokenizer_or_processor, words, normalized_boxes, image, CONFIG["MAX_SEQ_LENGTH"]
            )
        else:
            encoding = encode_page_v1(
                tokenizer_or_processor, words, normalized_boxes, CONFIG["MAX_SEQ_LENGTH"]
            )

        outputs = run_model(model, encoding, version)
        logits = outputs.logits[0]
        probs = torch.softmax(logits, dim=-1)
        pred_ids = probs.argmax(-1).tolist()
        confidences = probs.max(-1).values.tolist()

        word_ids_map = encoding.word_ids()

        pred_map = {}
        for token_idx, (pred_id, conf) in enumerate(zip(pred_ids, confidences)):
            widx = word_ids_map[token_idx]
            if widx is not None and widx < len(word_indices_ocr):
                ocr_idx = word_indices_ocr[widx]
                if ocr_idx not in pred_map:
                    pred_map[ocr_idx] = (id2label[pred_id], conf)

        for i in range(len(ocr_full_result["text"])):
            word = ocr_full_result["text"][i].strip()
            if not word:
                continue

            x = ocr_full_result["left"][i]
            y = ocr_full_result["top"][i]
            w = ocr_full_result["width"][i]
            h = ocr_full_result["height"][i]

            label, conf = pred_map.get(i, ("O", 0.0))

            all_predictions.append({
                "image_id": page_idx,
                "page_num": page_idx + 1,
                "word": word,
                "x1": x,
                "y1": y,
                "x2": x + w,
                "y2": y + h,
                "label": label,
                "confidence": round(conf, 4),
            })

    return all_predictions


def main():
    parser = argparse.ArgumentParser(description="Pre-annotazione fatture con modello LayoutLM")
    parser.add_argument("--pdf", required=True, help="PDF o directory di PDF da pre-annotare")
    parser.add_argument("--model", help="Directory del modello addestrato")
    parser.add_argument("--output_dir", default="output_data/preannotations", help="Directory output")
    parser.add_argument("--min_confidence", type=float, default=0.0,
                        help="Soglia minima di confidenza (sotto questa, label diventa O)")

    args = parser.parse_args()
    start_time = datetime.now()

    model_dir = Path(args.model or CONFIG.get("TRAINED_MODEL_DIR", "models/invoice_model_v3"))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model, tokenizer_or_processor, version = load_model_and_tokenizer(model_dir)
    model.eval()
    print(f"Modello LayoutLM {version} caricato da {model_dir}")

    pdf_path = Path(args.pdf)
    if pdf_path.is_dir():
        pdf_files = sorted(pdf_path.glob("*.pdf"))
    else:
        pdf_files = [pdf_path]

    if not pdf_files:
        print(f"Nessun PDF trovato in {pdf_path}")
        sys.exit(1)

    total_words = 0
    total_labeled = 0

    for pdf_file in pdf_files:
        print(f"\nPre-annotazione: {pdf_file.name}")
        predictions = preannotate_pdf(pdf_file, model, tokenizer_or_processor, version)

        if not predictions:
            print("  Nessuna parola trovata.")
            continue

        df = pd.DataFrame(predictions)

        if args.min_confidence > 0:
            mask = (df["label"] != "O") & (df["confidence"] < args.min_confidence)
            reset_count = mask.sum()
            if reset_count > 0:
                df.loc[mask, "label"] = "O"
                print(f"  {reset_count} label resettate a O (confidenza < {args.min_confidence})")

        labeled = df[df["label"] != "O"]
        total_words += len(df)
        total_labeled += len(labeled)

        print(f"  Parole totali: {len(df)} | Con label: {len(labeled)}")
        if not labeled.empty:
            for label, count in labeled["label"].value_counts().items():
                avg_conf = labeled[labeled["label"] == label]["confidence"].mean()
                print(f"    {label}: {count} (conf media: {avg_conf:.3f})")

        output_path = output_dir / f"{pdf_file.stem}_preannotated.xlsx"
        df.to_excel(output_path, index=False)
        print(f"  Salvato: {output_path}")

    elapsed = datetime.now() - start_time
    print(f"\nCompletato: {len(pdf_files)} PDF, {total_words} parole, {total_labeled} label suggerite")
    print(f"Tempo: {elapsed}")
    print(f"\nProssimo step: apri annotation_tool.py -> Importa Pre-annotazioni -> correggi -> salva")


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning, module="tokenizers")
    main()
