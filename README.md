# Invoice Analyzer - LayoutLM (PyTorch)

Automated data extraction from PDF invoices using **LayoutLM v1** (Microsoft) -- a transformer model that combines text content with spatial layout information for token classification.

The system performs OCR on invoice PDFs and uses a fine-tuned LayoutLM model to identify and extract key fields:

| Field | Description |
|-------|-------------|
| **VENDOR** | Supplier / issuing company name |
| **CUSTOMER** | Recipient / billing company name |
| **DATE** | Invoice date |
| **TOTAL** | Total amount |
| **INVOICE_NUMBER** | Invoice reference number |

## How it works

```
PDF Invoice
    |
    v
[pdf2image] --> Page images
    |
    v
[Tesseract OCR] --> Words + bounding boxes
    |
    v
[LayoutLM] --> Token classification (B-VENDOR, I-VENDOR, B-TOTAL, ...)
    |
    v
[Post-processing] --> Structured data (heuristics, validation, label matching)
    |
    v
Excel output
```

The extraction pipeline uses a model-first approach:
1. **Model prediction** -- LayoutLM token classification on OCR output (primary source of truth)
2. **Positional heuristics** -- scoring adjustments based on where entities appear in the document
3. **Fallback strategies** (only for fields the model did not find):
   - Enhanced total detection near total-related keywords
   - Direct label-value matching for known patterns (multilingual)
   - Rule-based validation (date, amount, invoice number format checks)

## Configuration

The file `invoice-analyzer/invoice_config.json` controls paths and hyperparameters:

| Key | Description |
|-----|-------------|
| `POPPLER_PATH` | Path to Poppler binaries (auto-detected if in PATH) |
| `TESSERACT_PATH` | Path to Tesseract binary (auto-detected if in PATH) |
| `BASE_MODEL_DIR` | LayoutLM base model for training |
| `TRAINED_MODEL_DIR` | Fine-tuned model for inference |
| `RESULTS_DIR` | Output directory for inference results |

Path auto-detection works on macOS, Linux and Windows -- manual config is only needed if tools are not in system PATH.

## Project structure

```
pytorch_modular/
├── invoice-analyzer/
│   ├── config.py                # Shared configuration, labels, path auto-detection
│   ├── utils.py                 # Format validators, text similarity, normalization
│   ├── ocr.py                   # PDF-to-image conversion, Tesseract OCR
│   ├── model.py                 # LayoutLM tokenizer loading
│   ├── extraction.py            # Entity extraction, postprocessing, heuristics
│   ├── export.py                # Excel export, prediction visualization
│   ├── invoice_inference.py     # Inference entry point
│   ├── train_invoice_model.py   # Training entry point (incremental learning)
│   ├── annotation_tool.py       # GUI for creating training annotations
│   ├── invoice_config.json      # Paths and hyperparameters
│   ├── models/
│   │   ├── layoutlm-base-uncased/  # Base model from HuggingFace (not tracked)
│   │   └── invoice_model/          # Fine-tuned model (not tracked)
│   ├── input_data/              # PDF invoices for training (not tracked)
│   └── output_data/
│       ├── annotations/         # Training annotations (.xlsx)
│       └── results/             # Inference output (.xlsx)
├── requirements.txt
├── SETUP.md                     # Detailed setup instructions
└── README.md
```

## Quick start

### 1. Install system dependencies

**macOS:**
```bash
brew install tesseract tesseract-lang poppler
```

**Linux:**
```bash
sudo apt install tesseract-ocr tesseract-ocr-ita poppler-utils
```

**Windows:** see [SETUP.md](SETUP.md) for download links.

### 2. Install Python dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Download LayoutLM base model

Download from [HuggingFace](https://huggingface.co/microsoft/layoutlm-base-uncased/tree/main) and place files in `invoice-analyzer/models/layoutlm-base-uncased/`.

### 4. Annotate, train, run

```bash
cd invoice-analyzer

# Create training annotations (GUI)
python annotation_tool.py

# Train the model
python train_invoice_model.py --annotations <annotations.xlsx> --pdfs_dir <pdf_folder>

# Run inference on a new invoice
python invoice_inference.py --pdf <invoice.pdf>
```

See [SETUP.md](SETUP.md) for full configuration details and all CLI parameters.

## Tested on

- macOS (Apple Silicon) -- Python 3.14, PyTorch 2.10, Transformers 5.2
- Windows 10/11 -- Python 3.10+

## Tech stack

- **Model**: [LayoutLM v1](https://huggingface.co/microsoft/layoutlm-base-uncased) (HuggingFace Transformers)
- **Deep Learning**: PyTorch
- **Training**: HuggingFace Trainer + Accelerate
- **OCR**: Tesseract (via pytesseract)
- **PDF rendering**: Poppler (via pdf2image)
- **Data**: pandas + openpyxl

## License

This project is provided as-is for educational and personal use.
