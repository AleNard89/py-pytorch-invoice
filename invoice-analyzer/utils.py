import re


def normalize_box(box, width, height):
    if width <= 0 or height <= 0:
        raise ValueError(f"Dimensioni dellimmagine non valide: {width}x{height}")
    x1 = min(max(0, box[0]), width - 1)
    y1 = min(max(0, box[1]), height - 1)
    x2 = min(max(x1 + 1, box[2]), width)
    y2 = min(max(y1 + 1, box[3]), height)
    normalized_box = [
        int(1000 * (x1 / width)),
        int(1000 * (y1 / height)),
        int(1000 * (x2 / width)),
        int(1000 * (y2 / height)),
    ]
    for i, coord in enumerate(normalized_box):
        if coord < 0 or coord > 1000:
            print(f"ATTENZIONE: Coordinata normalizzata fuori range: {coord}. Valore corretto.")
            normalized_box[i] = max(0, min(1000, coord))
    return normalized_box


def contains_digits_or_alphanumeric(text):
    """Verifica se una stringa contiene numeri o è alfanumerica con almeno un numero."""
    return any(c.isdigit() for c in text) or (any(c.isalpha() for c in text) and any(c.isdigit() for c in text))


def is_valid_date_format(text):
    """
    Verifica se il testo è nel formato data valido.
    Supporta formati come:
    - DD/MM/YYYY (07/05/2025)
    - DD-MM-YYYY (07-05-2025)
    - DD.MM.YYYY (07.05.2025)
    - D Month YYYY (7 May 2025, 7 maggio 2025)
    - Month D, YYYY (May 7, 2025, maggio 7, 2025)
    - YYYY-MM-DD (2025-05-07, formato ISO)
    """
    import re
    from datetime import datetime

    text = text.strip()
    
    # Formati numerici separati da /, - o .
    if re.match(r'^(\d{1,2})[/.-](\d{1,2})[/.-](\d{4})$', text):
        # Estrai i componenti della data
        parts = re.split(r'[/.-]', text)
        day, month, year = int(parts[0]), int(parts[1]), int(parts[2])
        
        # Verifica che giorno e mese siano in range valido
        if not (1 <= day <= 31 and 1 <= month <= 12 and 1900 <= year <= 2100):
            return False
            
        # Verifica ulteriore per giorni validi nei mesi specifici
        days_in_month = [0, 31, 29 if (year % 4 == 0 and year % 100 != 0) or year % 400 == 0 else 28, 
                         31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
        if day > days_in_month[month]:
            return False
            
        return True
    
    # Formato ISO: YYYY-MM-DD
    if re.match(r'^(\d{4})-(\d{1,2})-(\d{1,2})$', text):
        parts = text.split('-')
        year, month, day = int(parts[0]), int(parts[1]), int(parts[2])
        
        if not (1 <= day <= 31 and 1 <= month <= 12 and 1900 <= year <= 2100):
            return False
            
        days_in_month = [0, 31, 29 if (year % 4 == 0 and year % 100 != 0) or year % 400 == 0 else 28, 
                         31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
        if day > days_in_month[month]:
            return False
            
        return True
    
    # Formati testuali: D Month YYYY o Month D, YYYY
    months_it = ['gennaio', 'febbraio', 'marzo', 'aprile', 'maggio', 'giugno', 
                'luglio', 'agosto', 'settembre', 'ottobre', 'novembre', 'dicembre']
    months_en = ['january', 'february', 'march', 'april', 'may', 'june', 
                'july', 'august', 'september', 'october', 'november', 'december']
    months_fr = ['janvier', 'février', 'mars', 'avril', 'mai', 'juin', 
                'juillet', 'août', 'septembre', 'octobre', 'novembre', 'décembre']
    
    months_short_it = [m[:3] for m in months_it]
    months_short_en = [m[:3] for m in months_en]
    months_short_fr = [m[:3] for m in months_fr]
    
    # Combina tutti i mesi
    all_months = months_it + months_en + months_fr + months_short_it + months_short_en + months_short_fr
    
    # Formato: D Month YYYY
    pattern1 = r'^(\d{1,2})\s+([a-zA-Z]+)\s+(\d{4})$'
    match1 = re.match(pattern1, text, re.IGNORECASE)
    if match1:
        day, month_name, year = match1.groups()
        day, year = int(day), int(year)
        month_name = month_name.lower()
        
        # Trova il numero del mese
        month = None
        for i, month_list in enumerate([months_it, months_en, months_fr, months_short_it, months_short_en, months_short_fr]):
            if month_name in [m.lower() for m in month_list]:
                month = (month_list.index(month_name) % 12) + 1
                break
        
        if month and 1 <= day <= 31 and 1900 <= year <= 2100:
            days_in_month = [0, 31, 29 if (year % 4 == 0 and year % 100 != 0) or year % 400 == 0 else 28, 
                            31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
            if day <= days_in_month[month]:
                return True
    
    # Formato: Month D, YYYY
    pattern2 = r'^([a-zA-Z]+)\s+(\d{1,2})(?:,|\.|\s+)?\s*(\d{4})$'
    match2 = re.match(pattern2, text, re.IGNORECASE)
    if match2:
        month_name, day, year = match2.groups()
        day, year = int(day), int(year)
        month_name = month_name.lower()
        
        month = None
        for i, month_list in enumerate([months_it, months_en, months_fr, months_short_it, months_short_en, months_short_fr]):
            if month_name in [m.lower() for m in month_list]:
                month = (month_list.index(month_name) % 12) + 1
                break
        
        if month and 1 <= day <= 31 and 1900 <= year <= 2100:
            days_in_month = [0, 31, 29 if (year % 4 == 0 and year % 100 != 0) or year % 400 == 0 else 28, 
                            31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
            if day <= days_in_month[month]:
                return True
    
    return False


def is_amount_format(text):
    """
    Verifica se il testo rappresenta un importo numerico valido.
    """
    import re
    
    # Rimuovi spazi iniziali e finali
    text = text.strip()
    if not text:
        return False
    
    # Rimuovi valute e simboli
    for symbol in ["€", "$", "£", "¥", "EUR", "USD", "GBP", "JPY"]:
        text = text.replace(symbol, "").strip()
    
    # Verifica formati numerici (supporta punti, virgole e spazi come separatori)
    # Formato europeo: 1.234,56 o 1234,56
    if re.match(r'^-?(\d{1,3}(?:[.\s]\d{3})*|\d+),\d{1,4}$', text):
        return True
    
    # Formato US: 1,234.56 o 1234.56
    if re.match(r'^-?(\d{1,3}(?:[,\s]\d{3})*|\d+)\.\d{1,4}$', text):
        return True
    
    # Numeri interi con separatori
    if re.match(r'^-?(\d{1,3}(?:[.,\s]\d{3})+|\d+)$', text):
        return True
    
    # Numeri semplici (almeno 1 cifra)
    if re.match(r'^-?\d+$', text):
        return True
    
    return False


def is_invoice_number_format(text):
    """
    Verifica se il testo ha un formato compatibile con un numero di fattura.
    Supporta formati come:
    - 2023/001
    - 001-2023
    - F123456
    - INV-001-2023
    - A/123/23
    - 00123 (numeri lunghi almeno 4-5 cifre)
    """
    import re
    
    text = str(text).strip()
    if not text:
        return False
    
    # Pattern comuni per numeri di fattura
    patterns = [
        # Anno/Numero (2023/001)
        r'^(20\d{2})[/.-](\d{2,6})$',
        
        # Numero/Anno (001/2023)
        r'^(\d{2,6})[/.-](20\d{2})$',
        
        # Prefisso-Numero (F-12345, INV-12345)
        r'^([a-zA-Z]{1,4})[/.-]?(\d{3,8})$',
        
        # Numero con prefisso e anno (INV-001-2023, F/001/23)
        r'^([a-zA-Z]{1,4})[/.-]?(\d{2,6})[/.-](20\d{2}|2[0-9]|[0-9]{2})$',
        
        # Solo numeri, ma almeno 4 cifre 
        r'^(\d{4,10})$',
        
        # Formato con lettere e numeri separati (A/123/23, INV/2023/001)
        r'^([a-zA-Z]{1,4})[/.-](\d{1,6})[/.-](\d{2,4})$'
    ]
    
    for pattern in patterns:
        if re.match(pattern, text):
            return True
            
    # Evita di interpretare date come numeri di fattura
    if is_valid_date_format(text):
        return False
        
    # Evita di interpretare importi come numeri di fattura
    if is_amount_format(text):
        return False
        
    return False


def validate_entity(entity_type, text, confidence):
    """Funzione di validazione migliorata per filtrare entità non valide."""
    
    if confidence < 70:  # Filtro base sulla confidenza
        return False
        
    # Controlli specifici migliorati per tipo di entità
    if entity_type == "VENDOR":
        # Escludiamo parole singole molto comuni o generiche
        lower_text = text.lower()
        generic_words = ["client", "customer", "vendor", "fornitore", "address", "adresse", 
                        "purchase", "order", "number", "shipping", "delivery", "reference"]
        
        # Controlla se contiene solo parole generiche
        words = lower_text.split()
        if all(word in generic_words for word in words):
            print(f"  Rifiutato {entity_type} '{text}' - termini generici")
            return False
        
        # Se è una singola parola generica, rifiuta
        if lower_text in generic_words and len(text.split()) == 1:
            print(f"  Rifiutato {entity_type} '{text}' - termine generico singolo")
            return False
            
        # Un fornitore non dovrebbe essere solo un numero
        if is_amount_format(text):
            print(f"  Rifiutato {entity_type} '{text}' - sembra un importo")
            return False
        
        # I vendor tendono ad avere nomi più complessi di 1-2 parole
        if len(text.split()) > 1:
            return True
            
    elif entity_type == "TOTAL":
        # Un totale deve essere un formato numerico valido
        return is_amount_format(text)
            
    elif entity_type == "DATE":
        # Una data deve essere in formato valido 
        return is_valid_date_format(text)
            
    elif entity_type == "INVOICE_NUMBER":
        # Un numero di fattura deve seguire un formato specifico
        return is_invoice_number_format(text)
            
    return True


def evaluate_entity_semantics(entity_type, text):
    """Valuta la coerenza semantica dell'entità in base al suo contenuto."""
    
    # Liste di termini che NON dovrebbero essere identificati come specifiche entità
    non_vendor_terms = ["purchase", "order", "number", "shipping", "delivery", "reference", 
                       "ref", "no.", "code", "vat", "fiscal", "payment", "terms"]
    
    non_customer_terms = ["ship", "to", "delivery", "destination", "deliver", "consignee"]
    
    non_total_terms = ["subtotal", "tax", "vat", "payment", "delivery", "shipping", 
                      "discount", "description", "code"]
    
    semantic_score = 0
    
    if entity_type == "VENDOR":
        # Verifica se il testo contiene termini che NON dovrebbero essere un venditore
        lower_text = text.lower()
        matched_terms = [term for term in non_vendor_terms if term.lower() in lower_text]
        if matched_terms:
            penalty = min(50, 20 * len(matched_terms))  # Penalizzazione proporzionale al numero di termini trovati
            semantic_score -= penalty
            print(f"  Penalizzazione semantica per VENDOR '{text}': -{penalty} (contiene termini generici: {', '.join(matched_terms)})")
    
    elif entity_type == "CUSTOMER":
        lower_text = text.lower()
        matched_terms = [term for term in non_customer_terms if term.lower() in lower_text]
        if matched_terms:
            penalty = min(30, 15 * len(matched_terms))
            semantic_score -= penalty
            print(f"  Penalizzazione semantica per CUSTOMER '{text}': -{penalty} (contiene termini generici)")
            
    elif entity_type == "TOTAL":
        lower_text = text.lower()
        matched_terms = [term for term in non_total_terms if term.lower() in lower_text]
        if matched_terms:
            penalty = min(30, 15 * len(matched_terms))
            semantic_score -= penalty
            print(f"  Penalizzazione semantica per TOTAL '{text}': -{penalty} (contiene termini descrittivi)")
            
    return semantic_score


def parse_amount_to_float(text):
    """Converte un importo testuale in float, gestendo vari formati."""
    # Copia di sicurezza del testo originale
    original_text = text
    
    # Rimuovi simboli di valuta e spazi
    for symbol in ["€", "$", "£", "¥", "EUR", "USD", "GBP", "JPY"]:
        text = text.replace(symbol, "").strip()
    
    text = text.replace(" ", "")
    
    # Estrai solo la parte che sembra un numero con eventuali separatori
    digits_and_separators = ""
    for char in text:
        if char.isdigit() or char in ",.":
            digits_and_separators += char
    
    text = digits_and_separators
    if not text:
        raise ValueError(f"Nessun formato numerico trovato in: {original_text}")
    
    try:
        # Gestisci formato europeo (1.234,56)
        if "," in text and ("." in text or any(c.isdigit() for c in text.split(",")[0])):
            # Formato europeo: 1.234,56 o 1234,56
            text = text.replace(".", "")  # Rimuovi separatori migliaia
            text = text.replace(",", ".")  # Converti separatore decimale
            return float(text)
        
        # Gestisci formato US (1,234.56)
        elif "." in text:
            # Formato US: 1,234.56 o 1234.56
            text = text.replace(",", "")  # Rimuovi separatori migliaia
            return float(text)
        
        # Gestisci numeri con virgola senza punto (es. 1234,56)
        elif "," in text:
            text = text.replace(",", ".")
            return float(text)
        
        # Gestisci interi 
        elif text.isdigit():
            return float(text)
        
        # Non riconosciuto, solleva eccezione
        raise ValueError(f"Formato importo non riconosciuto: {original_text}")
    except:
        raise ValueError(f"Impossibile convertire in float: {original_text}")


def calculate_text_similarity(text1, text2):
    """
    Calcola la somiglianza tra due stringhe di testo.
    Ritorna un valore tra 0 e 1, dove 1 significa testi identici.
    """
    # Normalizza i testi (rimuovi spazi aggiuntivi, converti in minuscolo)
    text1 = text1.lower().strip()
    text2 = text2.lower().strip()
    
    # Se i testi sono identici, ritorna 1
    if text1 == text2:
        return 1.0
    
    # Calcola la distanza di Levenshtein (numero di modifiche necessarie per trasformare text1 in text2)
    def levenshtein_distance(s1, s2):
        if len(s1) < len(s2):
            return levenshtein_distance(s2, s1)
        
        if len(s2) == 0:
            return len(s1)
        
        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        
        return previous_row[-1]
    
    # Calcola la distanza
    distance = levenshtein_distance(text1, text2)
    
    # Calcola la similarità come 1 - (distanza / lunghezza massima)
    max_length = max(len(text1), len(text2))
    if max_length == 0:
        return 0.0
    
    similarity = 1.0 - (distance / max_length)
    return similarity
