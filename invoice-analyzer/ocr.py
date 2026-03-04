import os
import sys
import traceback
import pytesseract
from pdf2image import convert_from_path
from pathlib import Path

from config import CONFIG


def convert_pdf_to_images(pdf_path, dpi=None, poppler_path=None, max_pages=None):
    dpi = dpi or CONFIG["PDF_DPI"]
    poppler_path = poppler_path or CONFIG["POPPLER_PATH"]
    max_pages = max_pages or CONFIG["MAX_PAGES"]
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"Il file PDF non esiste: {pdf_path}")
    poppler_path = Path(poppler_path)
    if not poppler_path.exists():
        raise FileNotFoundError(f"Il percorso di Poppler non esiste: {poppler_path}")
    try:
        images = convert_from_path(
            str(pdf_path), dpi=dpi, poppler_path=str(poppler_path),
            first_page=1, last_page=max_pages, timeout=CONFIG["PDF_TIMEOUT"]
        )
        print(f"PDF convertito in {len(images)} immagini")
        return images
    except Exception as e:
        print(f"Errore nella conversione del PDF: {e}")
        if "timeout" in str(e).lower():
            try:
                print("Ritentativo con timeout esteso...")
                images = convert_from_path(
                    str(pdf_path), dpi=dpi, poppler_path=str(poppler_path),
                    first_page=1, last_page=max_pages, timeout=CONFIG["PDF_TIMEOUT"] * 2
                )
                print(f"PDF convertito in {len(images)} immagini con timeout esteso")
                return images
            except Exception as e2:
                print(f"Anche il tentativo con timeout esteso è fallito: {e2}")
        try:
            print("Tentativo di conversione pagina per pagina...")
            images = []
            for page_num in range(1, min(max_pages + 1, 6)):
                try:
                    page_images = convert_from_path(
                        str(pdf_path), dpi=dpi, poppler_path=str(poppler_path),
                        first_page=page_num, last_page=page_num, timeout=CONFIG["PDF_TIMEOUT"]
                    )
                    if page_images:
                        images.extend(page_images)
                        print(f"Pagina {page_num} convertita con successo")
                except Exception as e3:
                    print(f"Impossibile convertire la pagina {page_num}: {e3}")
            if images:
                print(f"PDF convertito in {len(images)} immagini con metodo alternativo")
                return images
            else:
                raise Exception("Nessuna pagina convertita con successo")
        except Exception as e4:
            print(f"Tutti i tentativi di conversione sono falliti: {e4}")
            raise


def process_image_with_ocr(image, tesseract_path=None):
    tesseract_path = tesseract_path or CONFIG["TESSERACT_PATH"]
    tesseract_path = Path(tesseract_path)
    if not tesseract_path.exists():
        raise FileNotFoundError(f"Il percorso di Tesseract non esiste: {tesseract_path}")
    pytesseract.pytesseract.tesseract_cmd = str(tesseract_path)
    try:
        print(f"Utilizzando Tesseract versione: {pytesseract.get_tesseract_version()}")
        ocr_result = pytesseract.image_to_data(
            image, output_type=pytesseract.Output.DICT, config="--psm 6 --oem 3 -l ita+eng"
        )
        words, boxes, word_indices = [], [], []
        for i in range(len(ocr_result["text"])):
            word = ocr_result["text"][i].strip()
            if word and ocr_result["conf"][i] > 30:
                x, y, w, h = ocr_result["left"][i], ocr_result["top"][i], ocr_result["width"][i], ocr_result["height"][i]
                if w > 1 and h > 1:
                    words.append(word)
                    boxes.append([x, y, x + w, y + h])
                    word_indices.append(i)
        print(f"Trovate {len(words)} parole nellimmagine con confidenza sufficiente")
        if not words:
            print("Nessuna parola trovata, tentativo con parametri OCR alternativi...")
            ocr_result = pytesseract.image_to_data(
                image, output_type=pytesseract.Output.DICT, config="--psm 4 --oem 3 -l ita+eng"
            )
            for i in range(len(ocr_result["text"])):
                word = ocr_result["text"][i].strip()
                if word:
                    x, y, w, h = ocr_result["left"][i], ocr_result["top"][i], ocr_result["width"][i], ocr_result["height"][i]
                    if w > 1 and h > 1:
                        words.append(word)
                        boxes.append([x, y, x + w, y + h])
                        word_indices.append(i)
            print(f"Trovate {len(words)} parole con metodo alternativo")
        if not words:
            raise ValueError("OCR non ha rilevato testo nellimmagine.")
        return words, boxes, word_indices, ocr_result
    except Exception as e:
        print(f"Errore nel processo OCR: {e}")
        traceback.print_exc()
        raise
