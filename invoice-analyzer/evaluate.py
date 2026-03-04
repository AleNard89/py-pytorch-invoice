"""
Evaluation script per il modello LayoutLM di estrazione dati da fatture.
Confronta le predizioni del modello con le annotazioni ground truth
e calcola metriche precision/recall/F1 per ogni tipo di entita'.
"""

import argparse
import json
import os
import sys
import torch
import warnings
from pathlib import Path
from datetime import datetime

import pandas as pd

from config import CONFIG, LABELS, id2label
from utils import normalize_box, calculate_text_similarity
from ocr import convert_pdf_to_images, process_image_with_ocr
from model import load_model_and_tokenizer
from extraction import extract_structured_data, apply_positional_heuristics
from invoice_inference import encode_page_v1, encode_page_v3, run_model

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

ENTITY_TYPES = ["VENDOR", "CUSTOMER", "DATE", "TOTAL", "INVOICE_NUMBER"]
DEFAULT_SIMILARITY_THRESHOLD = 0.7
SIMILARITY_THRESHOLD = DEFAULT_SIMILARITY_THRESHOLD


def load_ground_truth(annotation_path):
    """Ricostruisce le entita' ground truth dalle annotazioni B-/I- in Excel."""
    df = pd.read_excel(annotation_path)
    entities = {}
    current_type = None
    current_words = []

    for _, row in df.iterrows():
        label = str(row["label"])
        word = str(row["word"])

        if label.startswith("B-"):
            if current_type and current_words:
                entities.setdefault(current_type, []).append(" ".join(current_words))
            current_type = label[2:]
            current_words = [word]
        elif label.startswith("I-") and current_type == label[2:]:
            current_words.append(word)
        else:
            if current_type and current_words:
                entities.setdefault(current_type, []).append(" ".join(current_words))
            current_type = None
            current_words = []

    if current_type and current_words:
        entities.setdefault(current_type, []).append(" ".join(current_words))

    return {k: v[0] for k, v in entities.items()}


def run_inference(pdf_path, model, tokenizer_or_processor, version="v1"):
    """Esegue inferenza su un PDF e restituisce le entita' estratte come {tipo: testo}."""
    images = convert_pdf_to_images(pdf_path)
    if not images:
        return {}

    image = images[0]
    words, boxes, word_indices_ocr, ocr_full_result = process_image_with_ocr(image)
    if not words:
        return {}

    normalized_boxes = [normalize_box(b, image.width, image.height) for b in boxes]

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

    structured_data = extract_structured_data(ocr_full_result, token_predictions)
    structured_data = apply_positional_heuristics(structured_data, ocr_full_result)

    result = {}
    for entity_type, entries in structured_data.items():
        if entity_type in ENTITY_TYPES and entries:
            best = max(entries, key=lambda x: x.get("total_score", x.get("confidence", 0)))
            result[entity_type] = best["text"]

    return result


def evaluate_document(predictions, ground_truth):
    """Confronta predizioni e ground truth per un singolo documento."""
    results = {}
    for entity_type in ENTITY_TYPES:
        gt = ground_truth.get(entity_type)
        pred = predictions.get(entity_type)

        if gt is None:
            continue

        if pred is None:
            results[entity_type] = {
                "found": False,
                "exact_match": False,
                "similarity": 0.0,
                "predicted": "",
                "ground_truth": gt,
            }
        else:
            similarity = calculate_text_similarity(pred, gt)
            exact = pred.strip().lower() == gt.strip().lower()
            results[entity_type] = {
                "found": True,
                "exact_match": exact,
                "similarity": similarity,
                "predicted": pred,
                "ground_truth": gt,
            }

    return results


def compute_metrics(all_doc_results):
    """Calcola precision, recall, F1 per tipo di entita' e overall."""
    metrics = {}

    for entity_type in ENTITY_TYPES:
        tp_exact = 0
        tp_fuzzy = 0
        fp = 0
        fn = 0
        total_similarity = 0.0
        count = 0

        for doc_result in all_doc_results:
            if entity_type not in doc_result:
                continue
            r = doc_result[entity_type]
            count += 1
            total_similarity += r["similarity"]

            if r["exact_match"]:
                tp_exact += 1
                tp_fuzzy += 1
            elif r["found"] and r["similarity"] >= SIMILARITY_THRESHOLD:
                tp_fuzzy += 1
                fp += 0
            elif r["found"]:
                fp += 1
            else:
                fn += 1

        if count == 0:
            continue

        fn_exact = count - tp_exact
        fp_exact = count - tp_exact - (count - tp_exact - fn)

        precision_exact = tp_exact / max(tp_exact + (count - tp_exact - fn), 1)
        recall_exact = tp_exact / max(count, 1)
        f1_exact = 2 * precision_exact * recall_exact / max(precision_exact + recall_exact, 1e-9)

        precision_fuzzy = tp_fuzzy / max(count, 1)
        recall_fuzzy = tp_fuzzy / max(count, 1)
        f1_fuzzy = 2 * precision_fuzzy * recall_fuzzy / max(precision_fuzzy + recall_fuzzy, 1e-9)

        metrics[entity_type] = {
            "count": count,
            "exact_match": tp_exact,
            "fuzzy_match": tp_fuzzy,
            "avg_similarity": total_similarity / count,
            "precision_exact": precision_exact,
            "recall_exact": recall_exact,
            "f1_exact": f1_exact,
            "precision_fuzzy": precision_fuzzy,
            "recall_fuzzy": recall_fuzzy,
            "f1_fuzzy": f1_fuzzy,
        }

    all_exact = sum(m["exact_match"] for m in metrics.values())
    all_fuzzy = sum(m["fuzzy_match"] for m in metrics.values())
    all_count = sum(m["count"] for m in metrics.values())
    all_sim = sum(m["avg_similarity"] * m["count"] for m in metrics.values())

    if all_count > 0:
        metrics["OVERALL"] = {
            "count": all_count,
            "exact_match": all_exact,
            "fuzzy_match": all_fuzzy,
            "avg_similarity": all_sim / all_count,
            "accuracy_exact": all_exact / all_count,
            "accuracy_fuzzy": all_fuzzy / all_count,
        }

    return metrics


def print_results(all_doc_results, metrics, doc_names):
    """Stampa i risultati in formato tabellare."""
    print("\n" + "=" * 90)
    print("RISULTATI PER DOCUMENTO")
    print("=" * 90)

    for i, (doc_name, doc_result) in enumerate(zip(doc_names, all_doc_results)):
        print(f"\n--- {doc_name} ---")
        print(f"{'Entita':<20} {'Ground Truth':<25} {'Predizione':<25} {'Sim':>5} {'Match':>6}")
        print("-" * 85)
        for entity_type in ENTITY_TYPES:
            if entity_type not in doc_result:
                continue
            r = doc_result[entity_type]
            gt_short = r["ground_truth"][:22] + ".." if len(r["ground_truth"]) > 24 else r["ground_truth"]
            pred_short = r["predicted"][:22] + ".." if len(r["predicted"]) > 24 else r["predicted"]
            match_str = "EXACT" if r["exact_match"] else ("FUZZY" if r["similarity"] >= SIMILARITY_THRESHOLD else "MISS")
            print(f"{entity_type:<20} {gt_short:<25} {pred_short:<25} {r['similarity']:>5.2f} {match_str:>6}")

    print("\n" + "=" * 90)
    print("METRICHE AGGREGATE")
    print("=" * 90)
    print(f"\n{'Entita':<20} {'N':>3} {'Exact':>6} {'Fuzzy':>6} {'Avg Sim':>8} {'P(exact)':>9} {'R(exact)':>9} {'F1(exact)':>10} {'F1(fuzzy)':>10}")
    print("-" * 90)

    for entity_type in ENTITY_TYPES:
        if entity_type not in metrics:
            continue
        m = metrics[entity_type]
        print(
            f"{entity_type:<20} {m['count']:>3} {m['exact_match']:>6} {m['fuzzy_match']:>6} "
            f"{m['avg_similarity']:>8.3f} {m['precision_exact']:>9.3f} {m['recall_exact']:>9.3f} "
            f"{m['f1_exact']:>10.3f} {m['f1_fuzzy']:>10.3f}"
        )

    if "OVERALL" in metrics:
        m = metrics["OVERALL"]
        print("-" * 90)
        print(
            f"{'OVERALL':<20} {m['count']:>3} {m['exact_match']:>6} {m['fuzzy_match']:>6} "
            f"{m['avg_similarity']:>8.3f} {m['accuracy_exact']:>9.3f} {m['accuracy_exact']:>9.3f} "
            f"{'':>10} {'':>10}"
        )

    print(f"\nSoglia fuzzy match: {SIMILARITY_THRESHOLD}")


def main():
    parser = argparse.ArgumentParser(description="Evaluation del modello LayoutLM per estrazione fatture")
    parser.add_argument("--annotations_dir", default="output_data/annotations", help="Directory con annotazioni Excel")
    parser.add_argument("--pdfs_dir", default="input_data", help="Directory con i file PDF")
    parser.add_argument("--model", help="Directory del modello addestrato")
    parser.add_argument("--output", help="Salva risultati in JSON")
    parser.add_argument("--threshold", type=float, default=DEFAULT_SIMILARITY_THRESHOLD, help="Soglia per fuzzy match")

    args = parser.parse_args()
    global SIMILARITY_THRESHOLD
    SIMILARITY_THRESHOLD = args.threshold

    start_time = datetime.now()
    print(f"Inizio evaluation: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

    model_dir = Path(args.model or CONFIG.get("TRAINED_MODEL_DIR", "models/invoice_model"))
    annotations_dir = Path(args.annotations_dir)
    pdfs_dir = Path(args.pdfs_dir)

    print(f"Modello: {model_dir}")
    print(f"Annotazioni: {annotations_dir}")
    print(f"PDF: {pdfs_dir}")

    model, tokenizer_or_processor, version = load_model_and_tokenizer(model_dir)
    model.eval()
    print(f"Modello LayoutLM {version} caricato.")

    annotation_files = sorted(annotations_dir.glob("*_annotated.xlsx"))
    if not annotation_files:
        print(f"Nessun file di annotazione trovato in {annotations_dir}")
        sys.exit(1)

    pdf_files = sorted(pdfs_dir.glob("*.pdf"))
    pdf_map = {p.stem: p for p in pdf_files}

    all_doc_results = []
    doc_names = []

    for ann_file in annotation_files:
        doc_stem = ann_file.stem.replace("_annotated", "")
        pdf_path = pdf_map.get(doc_stem)

        if pdf_path is None:
            print(f"PDF non trovato per annotazione: {ann_file.name} (cercato: {doc_stem}.pdf)")
            continue

        print(f"\nValutazione: {doc_stem}")
        ground_truth = load_ground_truth(ann_file)
        print(f"  Ground truth: {ground_truth}")

        predictions = run_inference(pdf_path, model, tokenizer_or_processor, version)
        print(f"  Predizioni:   {predictions}")

        doc_result = evaluate_document(predictions, ground_truth)
        all_doc_results.append(doc_result)
        doc_names.append(doc_stem)

    if not all_doc_results:
        print("Nessun documento valutato.")
        sys.exit(1)

    metrics = compute_metrics(all_doc_results)
    print_results(all_doc_results, metrics, doc_names)

    if args.output:
        output_data = {
            "timestamp": start_time.isoformat(),
            "model": str(model_dir),
            "version": version,
            "threshold": SIMILARITY_THRESHOLD,
            "documents": {name: result for name, result in zip(doc_names, all_doc_results)},
            "metrics": metrics,
        }
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(output_data, f, indent=2, default=str)
        print(f"\nRisultati salvati in: {output_path}")

    end_time = datetime.now()
    print(f"\nTempo totale: {end_time - start_time}")


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning, module="tokenizers")
    main()
