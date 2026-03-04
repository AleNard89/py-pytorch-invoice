from pathlib import Path
from transformers import (
    LayoutLMTokenizer,
    LayoutLMForTokenClassification,
    LayoutLMConfig,
    LayoutLMv3ForTokenClassification,
    LayoutLMv3Processor,
    LayoutLMv3Config,
)
from config import LABELS, id2label, label2id


def detect_model_version(model_dir):
    """Rileva la versione del modello dalla config.json."""
    import json
    config_path = Path(model_dir) / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            cfg = json.load(f)
        model_type = cfg.get("model_type", "")
        if "layoutlmv3" in model_type:
            return "v3"
    return "v1"


def load_model_and_tokenizer(model_dir, num_labels=None, for_training=False):
    """Carica modello e tokenizer/processor, auto-detect v1 o v3."""
    model_dir = Path(model_dir)
    if not model_dir.exists():
        raise FileNotFoundError(f"Directory del modello non trovata: {model_dir}")

    version = detect_model_version(model_dir)
    if num_labels is None:
        num_labels = len(LABELS)

    print(f"Caricamento modello LayoutLM {version} da {model_dir}...")

    if version == "v3":
        return _load_v3(model_dir, num_labels, for_training)
    else:
        return _load_v1(model_dir, num_labels)


def _load_v3(model_dir, num_labels, for_training):
    config = LayoutLMv3Config.from_pretrained(str(model_dir), local_files_only=True)
    config.num_labels = num_labels
    config.id2label = id2label
    config.label2id = label2id

    model = LayoutLMv3ForTokenClassification.from_pretrained(
        str(model_dir), config=config, local_files_only=True,
        ignore_mismatched_sizes=True,
    )
    processor = LayoutLMv3Processor.from_pretrained(
        str(model_dir), apply_ocr=False, local_files_only=True,
    )
    print(f"LayoutLMv3 caricato: {sum(p.numel() for p in model.parameters())/1e6:.1f}M params")
    return model, processor, "v3"


def _load_v1(model_dir, num_labels):
    config = LayoutLMConfig.from_pretrained(str(model_dir), local_files_only=True)
    config.num_labels = num_labels
    config.id2label = id2label
    config.label2id = label2id

    model = LayoutLMForTokenClassification.from_pretrained(
        str(model_dir), config=config, local_files_only=True,
        ignore_mismatched_sizes=True,
    )
    tokenizer = _load_v1_tokenizer(model_dir)
    print(f"LayoutLMv1 caricato: {sum(p.numel() for p in model.parameters())/1e6:.1f}M params")
    return model, tokenizer, "v1"


def _load_v1_tokenizer(model_dir):
    model_dir = Path(model_dir)
    try:
        from transformers import LayoutLMTokenizerFast
        tokenizer = LayoutLMTokenizerFast.from_pretrained(str(model_dir), local_files_only=True)
        print("Tokenizer LayoutLM Fast caricato")
        return tokenizer
    except Exception:
        pass
    try:
        tokenizer = LayoutLMTokenizer.from_pretrained(str(model_dir), local_files_only=True)
        print("Tokenizer LayoutLM caricato")
        return tokenizer
    except Exception:
        pass
    from transformers import BertTokenizer
    tokenizer = BertTokenizer.from_pretrained(str(model_dir), local_files_only=True)
    print("Tokenizer BERT caricato come fallback")
    return tokenizer
