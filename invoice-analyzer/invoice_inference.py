"""
Inferenza con modello LayoutLM (v1/v3) per estrazione dati da fatture.
"""

import os
import sys
import torch
import warnings
import traceback
from pathlib import Path

from config import CONFIG, LABELS, id2label, label2id
from utils import normalize_box
from ocr import convert_pdf_to_images, process_image_with_ocr
from model import load_model_and_tokenizer
from extraction import (
    extract_structured_data,
    apply_positional_heuristics,
    extract_total_with_enhanced_detection,
    enhance_extraction_with_direct_label_matching,
    add_rule_based_entities,
)
from export import export_to_excel

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"


def encode_page_v1(tokenizer, words, normalized_boxes, max_seq_length):
    """Encoding per LayoutLM v1: input_ids + bbox + token_type_ids."""
    try:
        encoding = tokenizer(
            words, boxes=normalized_boxes,
            padding="max_length", truncation=True,
            max_length=max_seq_length, return_tensors="pt",
            is_split_into_words=True,
        )
    except TypeError:
        encoding = tokenizer(
            words, padding="max_length", truncation=True,
            max_length=max_seq_length, return_tensors="pt",
            is_split_into_words=True,
        )

    if "bbox" not in encoding:
        aligned_boxes = []
        word_ids_enc = encoding.word_ids()
        for i, token_id in enumerate(encoding["input_ids"][0]):
            if token_id in (tokenizer.cls_token_id, tokenizer.sep_token_id, tokenizer.pad_token_id):
                aligned_boxes.append([0, 0, 0, 0])
            else:
                widx = word_ids_enc[i]
                if widx is not None and widx < len(normalized_boxes):
                    aligned_boxes.append(normalized_boxes[widx])
                else:
                    aligned_boxes.append([0, 0, 0, 0])
        if len(aligned_boxes) < max_seq_length:
            aligned_boxes.extend([[0, 0, 0, 0]] * (max_seq_length - len(aligned_boxes)))
        encoding["bbox"] = torch.tensor([aligned_boxes[:max_seq_length]], dtype=torch.long)

    return encoding


def encode_page_v3(processor, words, normalized_boxes, image, max_seq_length):
    """Encoding per LayoutLMv3: input_ids + bbox + pixel_values (multimodale)."""
    encoding = processor(
        image, words, boxes=normalized_boxes,
        padding="max_length", truncation=True,
        max_length=max_seq_length, return_tensors="pt",
    )
    return encoding


def run_model(model, encoding, version):
    """Esegue il forward pass del modello."""
    with torch.no_grad():
        if version == "v3":
            outputs = model(
                input_ids=encoding["input_ids"],
                attention_mask=encoding["attention_mask"],
                bbox=encoding["bbox"],
                pixel_values=encoding["pixel_values"],
            )
        else:
            outputs = model(
                input_ids=encoding["input_ids"],
                attention_mask=encoding["attention_mask"],
                token_type_ids=encoding["token_type_ids"],
                bbox=encoding["bbox"],
            )
    return outputs


def main():
    import argparse
    from datetime import datetime

    parser = argparse.ArgumentParser(description="Inferenza con modello LayoutLM per analisi fatture")
    parser.add_argument("--pdf", required=True, help="Percorso al file PDF da analizzare")
    parser.add_argument("--model", help="Directory del modello LayoutLM addestrato")
    parser.add_argument("--output", help="Directory di output per i risultati")
    parser.add_argument("--poppler", help="Percorso alla directory Poppler")
    parser.add_argument("--tesseract", help="Percorso all'eseguibile Tesseract")
    parser.add_argument("--debug", action="store_true", help="Abilita modalita' debug")

    args = parser.parse_args()
    start_time = datetime.now()
    print(f"Inizio esecuzione inferenza: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

    model_dir = Path(args.model or CONFIG.get("TRAINED_MODEL_DIR", "models/invoice_model"))
    output_base_dir = Path(args.output or CONFIG.get("RESULTS_DIR", "output_data/results"))
    poppler_path = args.poppler or CONFIG["POPPLER_PATH"]
    tesseract_path = args.tesseract or CONFIG["TESSERACT_PATH"]

    if args.debug:
        print(f"\n=== DEBUG === Modello: {model_dir} | Output: {output_base_dir}")

    try:
        if not model_dir.exists() or not (model_dir / "config.json").exists():
            raise FileNotFoundError(f"Directory del modello {model_dir} non valida.")

        has_weights = (model_dir / "pytorch_model.bin").exists() or (model_dir / "model.safetensors").exists()
        if not has_weights:
            raise FileNotFoundError(f"Nessun file di pesi trovato in {model_dir}")

        model, tokenizer_or_processor, version = load_model_and_tokenizer(model_dir)
        model.eval()

        pdf_file = Path(args.pdf)
        if not pdf_file.exists():
            raise FileNotFoundError(f"File PDF non trovato: {pdf_file}")

        images = convert_pdf_to_images(pdf_file, poppler_path=poppler_path)
        if not images:
            raise ValueError("Nessuna immagine estratta dal PDF.")

        all_results = []
        pdf_name = pdf_file.stem

        for page_idx, image in enumerate(images):
            print(f"\n--- Elaborazione pagina {page_idx + 1} ---")
            page_output_dir = output_base_dir / pdf_name / f"page_{page_idx + 1}"
            os.makedirs(page_output_dir, exist_ok=True)

            words, boxes, word_indices_ocr, ocr_full_result = process_image_with_ocr(image, tesseract_path=tesseract_path)
            if not words:
                print(f"Nessuna parola OCR per pagina {page_idx + 1}. Salto.")
                continue

            normalized_boxes = [normalize_box(b, image.width, image.height) for b in boxes]
            print(f"Tokenizzazione di {len(words)} parole (LayoutLM {version})...")

            if version == "v3":
                encoding = encode_page_v3(tokenizer_or_processor, words, normalized_boxes, image, CONFIG["MAX_SEQ_LENGTH"])
            else:
                encoding = encode_page_v1(tokenizer_or_processor, words, normalized_boxes, CONFIG["MAX_SEQ_LENGTH"])

            outputs = run_model(model, encoding, version)
            predictions_logits = outputs.logits.argmax(-1).squeeze().tolist()

            token_predictions = {}
            word_ids_map = encoding.word_ids()

            for token_idx, pred_id in enumerate(predictions_logits):
                widx = word_ids_map[token_idx]
                if widx is not None:
                    ocr_idx = word_indices_ocr[widx]
                    label = id2label[pred_id]
                    if str(ocr_idx) not in token_predictions:
                        token_predictions[str(ocr_idx)] = label

            print(f"Predizioni mappate: {len(token_predictions)} token")

            structured_data = extract_structured_data(ocr_full_result, token_predictions)
            structured_data = apply_positional_heuristics(structured_data, ocr_full_result)

            model_found_fields = set(structured_data.keys())
            expected_fields = {"VENDOR", "CUSTOMER", "DATE", "TOTAL", "INVOICE_NUMBER"}
            missing_fields = expected_fields - model_found_fields

            if missing_fields:
                print(f"Campi mancanti: {missing_fields} -- post-processing")

                if "TOTAL" in missing_fields:
                    total_result = extract_total_with_enhanced_detection(ocr_full_result)
                    if total_result:
                        structured_data["TOTAL"] = [total_result]

                enhanced_data = enhance_extraction_with_direct_label_matching(structured_data, ocr_full_result)
                for field in missing_fields:
                    if field in enhanced_data and field not in model_found_fields:
                        structured_data[field] = enhanced_data[field]

                rule_data = add_rule_based_entities(ocr_full_result, structured_data)
                for field in missing_fields:
                    if field in rule_data and field not in structured_data:
                        structured_data[field] = rule_data[field]
            else:
                print(f"Tutti i campi trovati dal modello: {model_found_fields}")

            all_results.append({f"page_{page_idx + 1}": structured_data})

            excel_path = page_output_dir / f"{pdf_name}_page_{page_idx + 1}_structured.xlsx"
            export_to_excel(structured_data, str(excel_path))

        if all_results and "page_1" in all_results[0]:
            excel_main_path = output_base_dir / f"{pdf_name}_data.xlsx"
            success = export_to_excel(all_results[0]["page_1"], str(excel_main_path))
            if success:
                print(f"\n✅ PROCESSO COMPLETATO: {excel_main_path}")
            else:
                print("\n⚠️ Errore nella generazione del file Excel")

    except Exception as e:
        print(f"Errore durante l'inferenza: {e}")
        traceback.print_exc()
    finally:
        end_time = datetime.now()
        print(f"Fine: {end_time.strftime('%Y-%m-%d %H:%M:%S')} | Tempo: {end_time - start_time}")


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning, module="tokenizers")
    main()
