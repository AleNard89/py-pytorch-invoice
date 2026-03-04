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

The extraction pipeline applies multiple strategies in sequence:
1. **Model prediction** -- LayoutLM token classification on OCR output
2. **Positional heuristics** -- scoring based on where entities appear in the document (e.g. totals near the bottom)
3. **Enhanced total detection** -- specialized search for monetary amounts near total-related keywords
4. **Direct label-value matching** -- scanning for known label patterns (multilingual) and extracting adjacent values
5. **Rule-based validation** -- cross-checking entity formats (date validation, amount parsing, invoice number patterns)

## Project structure

```
pytorch_modular/
├── invoice-analyzer/
│   ├── annotation_tool.py       # GUI for creating training annotations
│   ├── train_invoice_model.py   # Fine-tuning script (supports incremental learning)
│   ├── invoice_inference.py     # Inference pipeline
│   ├── invoice_config.json      # Paths and hyperparameters
│   ├── models/                  # LayoutLM base model + trained model (not tracked)
│   ├── input_data/              # PDF invoices for training (not tracked)
│   └── output_data/             # Annotations and results (not tracked)
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

## Tech stack

- **Model**: [LayoutLM v1](https://huggingface.co/microsoft/layoutlm-base-uncased) (HuggingFace Transformers)
- **Deep Learning**: PyTorch
- **OCR**: Tesseract (via pytesseract)
- **PDF rendering**: Poppler (via pdf2image)
- **Data**: pandas + openpyxl

## License

This project is provided as-is for educational and personal use.
