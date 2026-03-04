from pathlib import Path
from transformers import LayoutLMTokenizer, LayoutLMForTokenClassification, LayoutLMConfig


def load_local_tokenizer(model_dir):
    print(f"Caricamento del tokenizer locale da {model_dir}...")
    model_dir = Path(model_dir)
    if not model_dir.exists():
        raise FileNotFoundError(f"Directory del modello non trovata: {model_dir}")
    try:
        try:
            from transformers import LayoutLMTokenizerFast
            print("Tentativo di caricamento del tokenizer fast...")
            tokenizer = LayoutLMTokenizerFast.from_pretrained(str(model_dir), local_files_only=True)
            print("Tokenizer LayoutLM Fast caricato con successo")
            return tokenizer
        except (ImportError, Exception) as e:
            print(f"Impossibile caricare il tokenizer fast: {e}")
        tokenizer = LayoutLMTokenizer.from_pretrained(str(model_dir), local_files_only=True)
        print("Tokenizer LayoutLM caricato con successo dalla directory locale")
        return tokenizer
    except Exception as e:
        print(f"Errore nel caricamento del tokenizer LayoutLM: {e}")
        from transformers import BertTokenizer # Fallback estremo
        print("Tentativo fallback con tokenizer BERT...")
        try:
            tokenizer = BertTokenizer.from_pretrained(str(model_dir), local_files_only=True)
            print("Tokenizer BERT caricato come fallback")
            return tokenizer
        except Exception as e_bert:
            print(f"Fallito anche caricamento tokenizer BERT: {e_bert}")
            raise
