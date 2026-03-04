"""
Utility di annotazione per le fatture
Aiuta a creare annotazioni manualmente per le fatture, con interfaccia visuale
"""

import os
import sys
import json
import platform
import shutil
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import pandas as pd
import traceback
from pathlib import Path
from PIL import Image, ImageTk
import pytesseract

def _detect_path(binary_name, fallbacks=None):
    path = shutil.which(binary_name)
    if path:
        return path
    if fallbacks:
        for fb in fallbacks:
            if Path(fb).exists():
                return fb
    return binary_name

config_file = Path("invoice_config.json")
CONFIG = {
    "POPPLER_PATH": str(Path(_detect_path("pdftoppm", ["/opt/homebrew/bin/pdftoppm", "/usr/bin/pdftoppm"])).parent),
    "TESSERACT_PATH": _detect_path("tesseract", ["/opt/homebrew/bin/tesseract", "/usr/bin/tesseract"]),
    "PDF_DPI": 300
}

if config_file.exists():
    try:
        with open(config_file, "r") as f:
            CONFIG.update(json.load(f))
    except Exception as e:
        print(f"Errore nel caricamento della configurazione: {e}")

pytesseract.pytesseract.tesseract_cmd = CONFIG["TESSERACT_PATH"]

# Importazione condizionale di pdf2image
try:
    from pdf2image import convert_from_path
except ImportError:
    print("ERRORE: La libreria pdf2image non è installata.")
    print("Installa con: pip install pdf2image")
    print("Assicurati anche di avere Poppler installato sul tuo sistema.")
    sys.exit(1)

# Etichette disponibili
LABELS = [
    "O",                # Outside/Background
    "B-VENDOR",         # Beginning of vendor name
    "I-VENDOR",         # Inside vendor name
    "B-CUSTOMER",       # Beginning of customer name
    "I-CUSTOMER",       # Inside customer name
    "B-DATE",           # Beginning of date
    "I-DATE",           # Inside date
    "B-TOTAL",          # Beginning of total amount
    "I-TOTAL",          # Inside total amount
    "B-ITEM",           # Beginning of item description
    "I-ITEM",           # Inside item description
    "B-QUANTITY",       # Beginning of quantity
    "I-QUANTITY",       # Inside quantity
    "B-PRICE",          # Beginning of price
    "I-PRICE",          # Inside price
    "B-INVOICE_NUMBER", # Beginning of invoice number
    "I-INVOICE_NUMBER"  # Inside invoice number
]

# Colori per ciascuna categoria di etichetta
COLORS = {
    "VENDOR": "#ff6666",         # Rosso
    "CUSTOMER": "#6666ff",       # Blu
    "DATE": "#66ff66",           # Verde
    "TOTAL": "#ff66ff",          # Viola
    "ITEM": "#ffaa66",           # Arancione
    "QUANTITY": "#66ffff",       # Ciano
    "PRICE": "#ff66aa",          # Rosa
    "INVOICE_NUMBER": "#aa6666", # Marrone
    "O": "#eeeeee"               # Grigio chiaro (sfondo)
}

class ImageAnnotationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Annotazione Fatture")
        self.root.geometry("1400x800")
        
        # Variabili
        self.pdf_path = None
        self.images = []
        self.current_page = 0
        self.ocr_results = []
        self.annotations = []
        self.scale_factor = 1.0
        self.selected_label = tk.StringVar(value=LABELS[0])
        
        # Layout principale
        self.setup_layout()
        
        # Bindings
        self.canvas.bind("<ButtonPress-1>", self.on_click)
        self.canvas.bind("<MouseWheel>", self.on_mousewheel)  # Windows
        self.canvas.bind("<Button-4>", lambda e: self.on_mousewheel(e, 1))  # Linux
        self.canvas.bind("<Button-5>", lambda e: self.on_mousewheel(e, -1))  # Linux
        
    def setup_layout(self):
        # Menu superiore
        menu_frame = ttk.Frame(self.root)
        menu_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(menu_frame, text="Apri PDF", command=self.open_pdf).pack(side=tk.LEFT, padx=5)
        ttk.Button(menu_frame, text="Salva", command=self.save_annotations).pack(side=tk.LEFT, padx=5)
        ttk.Button(menu_frame, text="Pagina precedente", command=self.prev_page).pack(side=tk.LEFT, padx=5)
        ttk.Button(menu_frame, text="Pagina successiva", command=self.next_page).pack(side=tk.LEFT, padx=5)
        
        # Informazioni
        self.info_label = ttk.Label(menu_frame, text="Nessun file aperto")
        self.info_label.pack(side=tk.RIGHT, padx=5)
        
        # Selettore etichetta
        label_frame = ttk.LabelFrame(self.root, text="Seleziona Etichetta")
        label_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)
        
        for label in LABELS:
            # Determina il colore di sfondo
            bg_color = COLORS["O"]
            if label != "O":
                entity_type = label[2:]  # Rimuovi "B-" o "I-"
                bg_color = COLORS.get(entity_type, "#ffffff")
            
            rb = ttk.Radiobutton(
                label_frame, 
                text=label, 
                value=label, 
                variable=self.selected_label
            )
            rb.pack(anchor=tk.W, padx=5, pady=2)
        
        # Frame principale con canvas per mostrare l'immagine
        main_frame = ttk.Frame(self.root)
        main_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Canvas per l'immagine
        self.canvas = tk.Canvas(main_frame, bg="white")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # Scrollbars
        h_scrollbar = ttk.Scrollbar(main_frame, orient=tk.HORIZONTAL, command=self.canvas.xview)
        h_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
        
        v_scrollbar = ttk.Scrollbar(main_frame, orient=tk.VERTICAL, command=self.canvas.yview)
        v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.canvas.configure(xscrollcommand=h_scrollbar.set, yscrollcommand=v_scrollbar.set)
        
        # Tabella di annotazioni
        self.tree_frame = ttk.LabelFrame(self.root, text="Annotazioni")
        self.tree_frame.pack(side=tk.LEFT, fill=tk.BOTH, padx=5, pady=5)
        
        columns = ("Word", "Label", "X1", "Y1", "X2", "Y2")
        self.tree = ttk.Treeview(self.tree_frame, columns=columns, show="headings")
        
        for col in columns:
            self.tree.heading(col, text=col)
            self.tree.column(col, width=70)
        
        self.tree.pack(fill=tk.BOTH, expand=True)
        self.tree.bind("<ButtonRelease-1>", self.on_tree_select)
        
        # Bottoni per la tabella
        tree_buttons = ttk.Frame(self.tree_frame)
        tree_buttons.pack(fill=tk.X)
        
        ttk.Button(tree_buttons, text="Modifica", command=self.edit_annotation).pack(side=tk.LEFT, padx=5, pady=5)
        ttk.Button(tree_buttons, text="Elimina", command=self.delete_annotation).pack(side=tk.LEFT, padx=5, pady=5)
    
    def open_pdf(self):
        """Apre un file PDF e ne mostra la prima pagina"""
        file_path = filedialog.askopenfilename(
            title="Seleziona un file PDF",
            filetypes=[("PDF files", "*.pdf")]
        )
        
        if not file_path:
            return
        
        self.pdf_path = file_path
        self.current_page = 0
        self.images = []
        self.ocr_results = []
        self.annotations = []
        
        try:
            # Converti PDF in immagini
            self.images = convert_from_path(
                self.pdf_path, 
                dpi=CONFIG["PDF_DPI"], 
                poppler_path=CONFIG["POPPLER_PATH"]
            )
            
            # Esegui OCR su tutte le pagine
            for img in self.images:
                ocr_result = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
                words_data = []
                
                for i in range(len(ocr_result["text"])):
                    word = ocr_result["text"][i].strip()
                    if word:  # Salta parole vuote
                        x, y, w, h = (
                            ocr_result["left"][i],
                            ocr_result["top"][i],
                            ocr_result["width"][i],
                            ocr_result["height"][i]
                        )
                        
                        words_data.append({
                            "word": word,
                            "x1": x,
                            "y1": y,
                            "x2": x + w,
                            "y2": y + h,
                            "label": "O"  # Default label
                        })
                
                self.ocr_results.append(words_data)
                
                # Inizializza annotazioni vuote per ogni pagina
                self.annotations.append([])
            
            # Mostra la prima pagina
            self.show_current_page()
            
            self.info_label.config(text=f"File: {os.path.basename(self.pdf_path)} | Pagina: 1/{len(self.images)}")
            
        except Exception as e:
            messagebox.showerror("Errore", f"Errore nell'apertura del PDF: {e}")
            traceback.print_exc()
    
    def show_current_page(self):
        """Mostra la pagina corrente"""
        if not self.images:
            return
        
        # Pulisci canvas
        self.canvas.delete("all")
        
        # Ottieni l'immagine corrente
        img = self.images[self.current_page]
        
        # Converti per Tkinter
        self.tk_img = ImageTk.PhotoImage(img)
        
        # Configura scrollregion del canvas
        self.canvas.config(scrollregion=(0, 0, img.width, img.height))
        
        # Mostra immagine
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_img)
        
        # Aggiorna la tabella con le annotazioni della pagina corrente
        self.update_tree()
        
        # Disegna i box delle parole
        self.draw_word_boxes()
    
    def draw_word_boxes(self):
        """Disegna i box delle parole con OCR"""
        if not self.images or self.current_page >= len(self.ocr_results):
            return
        
        words_data = self.ocr_results[self.current_page]
        
        for word_idx, word_data in enumerate(words_data):
            x1, y1, x2, y2 = word_data["x1"], word_data["y1"], word_data["x2"], word_data["y2"]
            word = word_data["word"]
            label = word_data["label"]
            
            # Determina il colore in base all'etichetta
            color = COLORS["O"]
            if label != "O":
                entity_type = label[2:] if label.startswith(("B-", "I-")) else label
                color = COLORS.get(entity_type, "#ffffff")
            
            # Disegna il box
            self.canvas.create_rectangle(x1, y1, x2, y2, outline=color, width=1, tags=(f"word_{word_idx}",))
            
            # Disegna il testo
            self.canvas.create_text(x1, y1-10, text=word, anchor=tk.SW, fill=color, tags=(f"word_{word_idx}",))
    
    def on_click(self, event):
        """Gestisce il click sul canvas"""
        if not self.images:
            return
        
        # Converti coordinate del canvas in coordinate immagine
        x, y = self.canvas.canvasx(event.x), self.canvas.canvasy(event.y)
        
        # Trova la parola cliccata
        words_data = self.ocr_results[self.current_page]
        found_idx = None
        
        for word_idx, word_data in enumerate(words_data):
            x1, y1, x2, y2 = word_data["x1"], word_data["y1"], word_data["x2"], word_data["y2"]
            if x1 <= x <= x2 and y1 <= y <= y2:
                found_idx = word_idx
                break
        
        if found_idx is not None:
            # Aggiorna l'etichetta della parola
            selected_label = self.selected_label.get()
            words_data[found_idx]["label"] = selected_label
            
            # Ridisegna tutti i box per aggiornare i colori
            self.draw_word_boxes()
            
            # Aggiorna la tabella
            self.update_tree()
    
    def update_tree(self):
        """Aggiorna la tabella con le annotazioni correnti"""
        # Pulisci la tabella
        for item in self.tree.get_children():
            self.tree.delete(item)
        
        if not self.images or self.current_page >= len(self.ocr_results):
            return
        
        # Aggiungi le parole con etichette non-O
        words_data = self.ocr_results[self.current_page]
        
        for word_idx, word_data in enumerate(words_data):
            if word_data["label"] != "O":
                values = (
                    word_data["word"],
                    word_data["label"],
                    word_data["x1"],
                    word_data["y1"],
                    word_data["x2"],
                    word_data["y2"]
                )
                self.tree.insert("", tk.END, iid=str(word_idx), values=values)
    
    def on_tree_select(self, event):
        selection = self.tree.selection()
        if selection:
            item_id = selection[0]
            word_idx = int(item_id)
            
            if word_idx < len(self.ocr_results[self.current_page]):
                word_data = self.ocr_results[self.current_page][word_idx]
                x1, y1, x2, y2 = word_data["x1"], word_data["y1"], word_data["x2"], word_data["y2"]
                
                # Rimuovi self.canvas.see(x1, y1)
                
                # Crea un rettangolo di evidenziazione
                self.canvas.delete("highlight")
                self.canvas.create_rectangle(
                    x1-2, y1-2, x2+2, y2+2, 
                    outline="red", width=2, 
                    tags=("highlight",)
                )
    
    def edit_annotation(self):
        """Modifica l'annotazione selezionata"""
        selection = self.tree.selection()
        if not selection:
            messagebox.showinfo("Info", "Seleziona un'annotazione da modificare")
            return
        
        item_id = selection[0]
        word_idx = int(item_id)
        
        if word_idx < len(self.ocr_results[self.current_page]):
            # Mostra finestra di dialogo per modificare l'etichetta
            edit_window = tk.Toplevel(self.root)
            edit_window.title("Modifica Etichetta")
            edit_window.transient(self.root)
            edit_window.grab_set()
            
            ttk.Label(edit_window, text="Seleziona nuova etichetta:").pack(padx=10, pady=5)
            
            label_var = tk.StringVar(value=self.ocr_results[self.current_page][word_idx]["label"])
            
            for label in LABELS:
                ttk.Radiobutton(
                    edit_window, 
                    text=label, 
                    value=label, 
                    variable=label_var
                ).pack(anchor=tk.W, padx=10, pady=2)
            
            def apply_change():
                self.ocr_results[self.current_page][word_idx]["label"] = label_var.get()
                self.update_tree()
                self.draw_word_boxes()
                edit_window.destroy()
            
            ttk.Button(edit_window, text="Applica", command=apply_change).pack(pady=10)
    
    def delete_annotation(self):
        """Rimuove l'etichetta dell'annotazione selezionata (imposta a O)"""
        selection = self.tree.selection()
        if not selection:
            messagebox.showinfo("Info", "Seleziona un'annotazione da eliminare")
            return
        
        item_id = selection[0]
        word_idx = int(item_id)
        
        if word_idx < len(self.ocr_results[self.current_page]):
            self.ocr_results[self.current_page][word_idx]["label"] = "O"
            self.update_tree()
            self.draw_word_boxes()
    
    def next_page(self):
        """Passa alla pagina successiva"""
        if self.images and self.current_page < len(self.images) - 1:
            self.current_page += 1
            self.show_current_page()
            self.info_label.config(text=f"File: {os.path.basename(self.pdf_path)} | Pagina: {self.current_page + 1}/{len(self.images)}")
    
    def prev_page(self):
        """Passa alla pagina precedente"""
        if self.images and self.current_page > 0:
            self.current_page -= 1
            self.show_current_page()
            self.info_label.config(text=f"File: {os.path.basename(self.pdf_path)} | Pagina: {self.current_page + 1}/{len(self.images)}")
    
    def on_mousewheel(self, event, delta=None):
        """Gestisce lo zoom con la rotella del mouse"""
        # Non implementato in questa versione semplificata
        pass
    
    def save_annotations(self):
        """Salva le annotazioni in formato Excel"""
        if not self.images or not self.pdf_path:
            messagebox.showinfo("Info", "Nessun file aperto da salvare")
            return
        
        save_path = filedialog.asksaveasfilename(
            title="Salva annotazioni",
            defaultextension=".xlsx",
            filetypes=[("Excel files", "*.xlsx")],
            initialfile=f"{Path(self.pdf_path).stem}_annotated.xlsx"
        )
        
        if not save_path:
            return
        
        try:
            # Prepara i dati per l'Excel
            data = []
            
            for page_idx, words_data in enumerate(self.ocr_results):
                for word_data in words_data:
                    if word_data["label"] != "O":  # Salva solo le annotazioni non-O
                        data.append({
                            "image_id": page_idx,
                            "page_num": page_idx + 1,
                            "word": word_data["word"],
                            "x1": word_data["x1"],
                            "y1": word_data["y1"],
                            "x2": word_data["x2"],
                            "y2": word_data["y2"],
                            "label": word_data["label"]
                        })
            
            # Crea DataFrame e salva
            df = pd.DataFrame(data)
            df.to_excel(save_path, index=False)
            
            messagebox.showinfo("Info", f"Annotazioni salvate in {save_path}")
            
        except Exception as e:
            messagebox.showerror("Errore", f"Errore nel salvataggio: {e}")
            traceback.print_exc()

def main():
    root = tk.Tk()
    app = ImageAnnotationApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()



















