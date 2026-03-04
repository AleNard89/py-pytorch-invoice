import platform
import shutil
import json
from pathlib import Path


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
    "MODEL_VERSION": "v3",
    "BASE_MODEL_DIR": "models/layoutlmv3-base",
    "TRAINED_MODEL_DIR": "models/invoice_model_v3",
    "RESULTS_DIR": "output_data/results",
    "PDF_DPI": 300,
    "PDF_TIMEOUT": 120,
    "MAX_PAGES": 5,
    "MAX_SEQ_LENGTH": 512,
    "USE_LOCAL_MODEL": True,
    "BATCH_SIZE": 2,
    "NUM_EPOCHS": 5,
    "LEARNING_RATE": 2e-5,
}


def load_config(config_path="invoice_config.json"):
    config = DEFAULT_CONFIG.copy()
    config_file = Path(config_path)
    
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


CONFIG = load_config()

LABELS = [
    "O", "B-VENDOR", "I-VENDOR", "B-CUSTOMER", "I-CUSTOMER", "B-DATE", "I-DATE",
    "B-TOTAL", "I-TOTAL", "B-ITEM", "I-ITEM", "B-QUANTITY", "I-QUANTITY",
    "B-PRICE", "I-PRICE", "B-INVOICE_NUMBER", "I-INVOICE_NUMBER"
]
id2label = {i: label for i, label in enumerate(LABELS)}
label2id = {label: i for i, label in enumerate(LABELS)}
