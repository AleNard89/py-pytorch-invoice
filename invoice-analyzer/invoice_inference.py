"""
Inferenza con modello LayoutLM (v1) per estrazione dati da fatture.
"""

import os
import sys
import torch
import warnings
import traceback
from pathlib import Path
from transformers import LayoutLMForTokenClassification, LayoutLMConfig

from config import CONFIG, LABELS, id2label, label2id
from utils import normalize_box
from ocr import convert_pdf_to_images, process_image_with_ocr
from model import load_local_tokenizer
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


def main():
    import argparse
    from datetime import datetime

    parser = argparse.ArgumentParser(description="Inferenza con modello LayoutLM per analisi fatture")
    parser.add_argument("--pdf", required=True, help="Percorso al file PDF da analizzare")
    parser.add_argument("--model", help="Directory del modello LayoutLM addestrato")
    parser.add_argument("--output", help="Directory di output per i risultati")
    parser.add_argument("--poppler", help="Percorso alla directory Poppler")
    parser.add_argument("--tesseract", help="Percorso all'eseguibile Tesseract")
    parser.add_argument("--debug", action="store_true", help="Abilita modalità debug")

    args = parser.parse_args()
    start_time = datetime.now()
    print(f"Inizio esecuzione inferenza: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

    model_dir = Path(args.model or CONFIG.get("TRAINED_MODEL_DIR", CONFIG.get("MODEL_DIR", "models/invoice_model")))
    output_base_dir = Path(args.output or CONFIG.get("RESULTS_DIR", CONFIG.get("OUTPUT_DIR", "output_data/results")))
    poppler_path = args.poppler or CONFIG["POPPLER_PATH"]
    tesseract_path = args.tesseract or CONFIG["TESSERACT_PATH"]

    if args.debug:
        print("\n=== MODALITÀ DEBUG ATTIVA ===")
        print(f"Modello: {model_dir}")
        print(f"Output: {output_base_dir}")
        print(f"Poppler: {poppler_path}")
        print(f"Tesseract: {tesseract_path}")

    try:
        print(f"Caricamento del modello da: {model_dir}")
        if not model_dir.exists() or not (model_dir / "pytorch_model.bin").exists() or not (model_dir / "config.json").exists():
            raise FileNotFoundError(f"Directory del modello \t{model_dir}\t non valida o file mancanti.")

        model_config = LayoutLMConfig.from_pretrained(str(model_dir), local_files_only=True)
        model = LayoutLMForTokenClassification.from_pretrained(str(model_dir), config=model_config, local_files_only=True)
        tokenizer = load_local_tokenizer(str(model_dir))
        model.eval()
        print("Modello e tokenizer caricati con successo.")

        pdf_file = Path(args.pdf)
        if not pdf_file.exists():
            raise FileNotFoundError(f"File PDF non trovato: {pdf_file}")

        images = convert_pdf_to_images(pdf_file, poppler_path=poppler_path)
        if not images:
            raise ValueError("Nessuna immagine estratta dal PDF.")

        all_results = []
        pdf_name = pdf_file.stem

        for page_idx, image in enumerate(images):
            print(f"\n--- Elaborazione pagina {page_idx + 1} --- ")
            page_output_dir = output_base_dir / pdf_name / f"page_{page_idx + 1}"
            os.makedirs(page_output_dir, exist_ok=True)

            words, boxes, word_indices_ocr, ocr_full_result = process_image_with_ocr(image, tesseract_path=tesseract_path)
            if not words:
                print(f"Nessuna parola trovata tramite OCR per la pagina {page_idx + 1}. Salto.")
                continue

            normalized_boxes_for_tokenizer = [normalize_box(b, image.width, image.height) for b in boxes]

            print(f"Tokenizzazione di {len(words)} parole...")
            try:
                encoding = tokenizer(
                    words,
                    boxes=normalized_boxes_for_tokenizer,
                    padding="max_length",
                    truncation=True,
                    max_length=CONFIG["MAX_SEQ_LENGTH"],
                    return_tensors="pt",
                    is_split_into_words=True
                )
            except TypeError:
                print("Fallback: tokenizzazione senza argomento boxes.")
                encoding = tokenizer(
                    words,
                    padding="max_length",
                    truncation=True,
                    max_length=CONFIG["MAX_SEQ_LENGTH"],
                    return_tensors="pt",
                    is_split_into_words=True
                )

            if "bbox" not in encoding:
                print("Aggiunta manuale di bbox all'encoding.")
                aligned_boxes = []
                word_ids_enc = encoding.word_ids()
                for i, token_id_enc in enumerate(encoding["input_ids"][0]):
                    if token_id_enc in (tokenizer.cls_token_id, tokenizer.sep_token_id, tokenizer.pad_token_id):
                        aligned_boxes.append([0, 0, 0, 0])
                    else:
                        word_idx_enc = word_ids_enc[i]
                        if word_idx_enc is not None and word_idx_enc < len(normalized_boxes_for_tokenizer):
                            aligned_boxes.append(normalized_boxes_for_tokenizer[word_idx_enc])
                        else:
                            aligned_boxes.append([0, 0, 0, 0])
                if len(aligned_boxes) < CONFIG["MAX_SEQ_LENGTH"]:
                    aligned_boxes.extend([[0, 0, 0, 0]] * (CONFIG["MAX_SEQ_LENGTH"] - len(aligned_boxes)))
                encoding["bbox"] = torch.tensor([aligned_boxes[:CONFIG["MAX_SEQ_LENGTH"]]], dtype=torch.long)

            input_ids = encoding["input_ids"]
            attention_mask = encoding["attention_mask"]
            token_type_ids = encoding["token_type_ids"]
            bbox = encoding["bbox"]

            with torch.no_grad():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, bbox=bbox)

            predictions_logits = outputs.logits.argmax(-1).squeeze().tolist()

            token_predictions = {}
            word_ids_map = encoding.word_ids()

            for token_idx, pred_id in enumerate(predictions_logits):
                word_idx_from_map = word_ids_map[token_idx]
                if word_idx_from_map is not None:
                    original_word_ocr_index = word_indices_ocr[word_idx_from_map]
                    label = id2label[pred_id]
                    if str(original_word_ocr_index) not in token_predictions:
                        token_predictions[str(original_word_ocr_index)] = label

            print(f"Numero di predizioni a livello di token mappate a parole OCR: {len(token_predictions)}")

            # 1. Estrai i dati strutturati dalle predizioni del modello
            structured_data = extract_structured_data(ocr_full_result, token_predictions)

            # 2. Applica le euristiche di posizione (solo bonus, non sovrascrive)
            structured_data = apply_positional_heuristics(structured_data, ocr_full_result)

            # Salva i campi gia' trovati dal modello
            model_found_fields = set(structured_data.keys())
            expected_fields = {"VENDOR", "CUSTOMER", "DATE", "TOTAL", "INVOICE_NUMBER"}
            missing_fields = expected_fields - model_found_fields

            if missing_fields:
                print(f"\nCampi mancanti dal modello: {missing_fields} -- uso post-processing solo per questi")

                if "TOTAL" in missing_fields:
                    total_result = extract_total_with_enhanced_detection(ocr_full_result)
                    if total_result:
                        structured_data["TOTAL"] = [total_result]
                        print(f"TOTAL trovato con detection avanzata: {total_result['text']}")

                enhanced_data = enhance_extraction_with_direct_label_matching(structured_data, ocr_full_result)
                for field in missing_fields:
                    if field in enhanced_data and field not in model_found_fields:
                        structured_data[field] = enhanced_data[field]

                rule_data = add_rule_based_entities(ocr_full_result, structured_data)
                for field in missing_fields:
                    if field in rule_data and field not in structured_data:
                        structured_data[field] = rule_data[field]
            else:
                print(f"\nTutti i campi trovati dal modello: {model_found_fields}")

            final_data = structured_data

            all_results.append({f"page_{page_idx + 1}": final_data})

            excel_path = page_output_dir / f"{pdf_name}_page_{page_idx + 1}_structured.xlsx"
            export_to_excel(final_data, str(excel_path))

        if all_results and "page_1" in all_results[0]:
            excel_main_path = output_base_dir / f"{pdf_name}_data.xlsx"
            success = export_to_excel(all_results[0]["page_1"], str(excel_main_path))

            if success:
                print(f"\n✅ PROCESSO COMPLETATO: File Excel generato in: {excel_main_path}")
            else:
                print("\n⚠️ Errore nella generazione del file Excel principale")

    except Exception as e:
        print(f"Errore durante l'inferenza: {e}")
        traceback.print_exc()
    finally:
        end_time = datetime.now()
        print(f"Fine esecuzione inferenza: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Tempo totale di esecuzione: {end_time - start_time}")


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning, module="tokenizers")
    main()
