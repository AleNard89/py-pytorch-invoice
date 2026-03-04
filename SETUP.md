# Invoice Analyzer - LayoutLM (PyTorch)

Sistema di estrazione dati da fatture PDF basato su **LayoutLM v1** (Microsoft).
Combina OCR (Tesseract) con un modello transformer che analizza testo + layout posizionale per identificare campi come: fornitore, cliente, data, totale, numero fattura.

## Dipendenze di sistema

### macOS
```bash
brew install tesseract
brew install tesseract-lang   # pacchetti lingua (italiano, francese, ecc.)
brew install poppler
```

### Linux (Debian/Ubuntu)
```bash
sudo apt install tesseract-ocr tesseract-ocr-ita poppler-utils
```

### Windows
- **Poppler**: scarica da https://github.com/oschwartz10612/poppler-windows/releases e aggiungi la cartella `bin` al PATH
- **Tesseract**: scarica da https://github.com/UB-Mannheim/tesseract/wiki (includi il pacchetto italiano) e aggiungi al PATH

## Ambiente Python

```bash
python3 -m venv .venv
source .venv/bin/activate       # macOS/Linux
# .venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

## Configurazione

Il file `invoice-analyzer/invoice_config.json` contiene i percorsi e gli iperparametri.
Esempio per macOS (gia' preconfigurato):

```json
{
    "POPPLER_PATH": "/opt/homebrew/bin",
    "TESSERACT_PATH": "/opt/homebrew/bin/tesseract",
    "MODEL_DIR": "models/layoutlm-base-uncased",
    "OUTPUT_DIR": "models/invoice_model"
}
```

Se i tool sono nel PATH di sistema, i percorsi vengono rilevati automaticamente.

## Modello LayoutLM

Scarica manualmente il modello base da HuggingFace:
https://huggingface.co/microsoft/layoutlm-base-uncased/tree/main

Posiziona i file in `invoice-analyzer/models/layoutlm-base-uncased/`:
- `config.json`
- `pytorch_model.bin`
- `tokenizer.json`
- `tokenizer_config.json`
- `special_tokens_map.json`
- `vocab.txt`

## Workflow

### 1. Annotazione (creazione dati di training)

```bash
cd invoice-analyzer
python annotation_tool.py
```

Apre una GUI dove puoi:
- Caricare un PDF
- Cliccare sulle parole rilevate dall'OCR
- Assegnare etichette (B-VENDOR, I-VENDOR, B-CUSTOMER, ecc.)
- Salvare le annotazioni in formato Excel

### 2. Training

```bash
cd invoice-analyzer
python train_invoice_model.py --annotations <path_annotazioni.xlsx> --pdfs_dir <cartella_pdf>
```

Parametri opzionali:
- `--epochs N` (default: 5)
- `--batch_size N` (default: 2)
- `--learning_rate F` (default: 2e-5)
- `--continue_from <path_modello>` per apprendimento incrementale

Il modello addestrato viene salvato in `models/invoice_model/`.

### 3. Inferenza

```bash
cd invoice-analyzer
python invoice_inference.py --pdf <file.pdf>
```

Parametri opzionali:
- `--model <directory_modello>` (default: da config)
- `--output <directory_output>` (default: da config)
- `--poppler <path>` e `--tesseract <path>` per override dei percorsi
- `--debug` per output dettagliato

I risultati vengono salvati in Excel nella directory di output.
