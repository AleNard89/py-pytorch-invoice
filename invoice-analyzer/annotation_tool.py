"""
Tool di annotazione fatture con PyQt6.
Supporta zoom, HiDPI (Retina), import pre-annotazioni dal modello.
"""

import os
import sys
import traceback
from pathlib import Path

import pandas as pd
import pytesseract
from PIL import Image
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QGraphicsView, QGraphicsScene,
    QGraphicsRectItem, QGraphicsSimpleTextItem, QDockWidget, QListWidget,
    QListWidgetItem, QToolBar, QFileDialog, QMessageBox, QStatusBar,
    QTreeWidget, QTreeWidgetItem, QHeaderView, QWidget, QVBoxLayout,
    QHBoxLayout, QPushButton, QLabel, QFrame, QSizePolicy,
)
from PyQt6.QtGui import (
    QPixmap, QImage, QPen, QColor, QBrush, QFont, QAction, QWheelEvent,
    QPainter, QIcon, QPalette, QLinearGradient,
)
from PyQt6.QtCore import Qt, QRectF, QSize

from config import CONFIG, LABELS
from pdf2image import convert_from_path

pytesseract.pytesseract.tesseract_cmd = CONFIG["TESSERACT_PATH"]

# Palette moderna con colori piu' saturi e distinti
ENTITY_COLORS = {
    "VENDOR":         {"hex": "#E74C3C", "name": "Fornitore"},
    "CUSTOMER":       {"hex": "#3498DB", "name": "Cliente"},
    "DATE":           {"hex": "#2ECC71", "name": "Data"},
    "TOTAL":          {"hex": "#9B59B6", "name": "Totale"},
    "ITEM":           {"hex": "#E67E22", "name": "Voce"},
    "QUANTITY":       {"hex": "#1ABC9C", "name": "Quantita'"},
    "PRICE":          {"hex": "#E91E63", "name": "Prezzo"},
    "INVOICE_NUMBER": {"hex": "#F39C12", "name": "N. Fattura"},
}

DARK_BG = "#1E1E2E"
PANEL_BG = "#2B2B3C"
SURFACE = "#363649"
TEXT_PRIMARY = "#CDD6F4"
TEXT_SECONDARY = "#A6ADC8"
ACCENT = "#89B4FA"
BORDER = "#45475A"

STYLESHEET = f"""
QMainWindow {{
    background-color: {DARK_BG};
}}
QToolBar {{
    background-color: {PANEL_BG};
    border-bottom: 1px solid {BORDER};
    spacing: 4px;
    padding: 4px 8px;
}}
QToolBar QToolButton {{
    background-color: {SURFACE};
    color: {TEXT_PRIMARY};
    border: 1px solid {BORDER};
    border-radius: 6px;
    padding: 6px 14px;
    font-size: 13px;
    font-weight: 500;
}}
QToolBar QToolButton:hover {{
    background-color: {ACCENT};
    color: {DARK_BG};
    border-color: {ACCENT};
}}
QToolBar QToolButton:pressed {{
    background-color: #7BA3E0;
}}
QToolBar::separator {{
    width: 1px;
    background-color: {BORDER};
    margin: 4px 8px;
}}
QDockWidget {{
    color: {TEXT_PRIMARY};
    titlebar-close-icon: none;
    titlebar-normal-icon: none;
    font-size: 13px;
    font-weight: bold;
}}
QDockWidget::title {{
    background-color: {PANEL_BG};
    padding: 8px;
    border-bottom: 1px solid {BORDER};
}}
QListWidget {{
    background-color: {PANEL_BG};
    border: none;
    outline: none;
    font-size: 13px;
}}
QListWidget::item {{
    padding: 8px 12px;
    border-radius: 4px;
    margin: 2px 4px;
}}
QListWidget::item:selected {{
    background-color: {SURFACE};
    border: 1px solid {ACCENT};
}}
QListWidget::item:hover {{
    background-color: {SURFACE};
}}
QTreeWidget {{
    background-color: {PANEL_BG};
    color: {TEXT_PRIMARY};
    border: none;
    outline: none;
    font-size: 12px;
    alternate-background-color: {SURFACE};
}}
QTreeWidget::item {{
    padding: 4px 2px;
    border-bottom: 1px solid {BORDER};
}}
QTreeWidget::item:selected {{
    background-color: {SURFACE};
    border: 1px solid {ACCENT};
}}
QHeaderView::section {{
    background-color: {SURFACE};
    color: {TEXT_SECONDARY};
    padding: 6px;
    border: none;
    border-bottom: 2px solid {ACCENT};
    font-weight: bold;
    font-size: 11px;
    text-transform: uppercase;
}}
QPushButton {{
    background-color: {SURFACE};
    color: {TEXT_PRIMARY};
    border: 1px solid {BORDER};
    border-radius: 6px;
    padding: 8px 16px;
    font-size: 13px;
    font-weight: 500;
}}
QPushButton:hover {{
    background-color: {ACCENT};
    color: {DARK_BG};
    border-color: {ACCENT};
}}
QPushButton#dangerBtn {{
    border-color: #E74C3C;
    color: #E74C3C;
}}
QPushButton#dangerBtn:hover {{
    background-color: #E74C3C;
    color: white;
}}
QStatusBar {{
    background-color: {PANEL_BG};
    color: {TEXT_SECONDARY};
    border-top: 1px solid {BORDER};
    font-size: 12px;
    padding: 4px 12px;
}}
QGraphicsView {{
    background-color: #16161E;
    border: none;
}}
QLabel#sectionTitle {{
    color: {TEXT_SECONDARY};
    font-size: 10px;
    font-weight: bold;
    text-transform: uppercase;
    letter-spacing: 1px;
    padding: 4px 8px;
}}
QLabel#statsLabel {{
    color: {TEXT_PRIMARY};
    font-size: 12px;
    padding: 4px 8px;
    background-color: {SURFACE};
    border-radius: 4px;
}}
QFrame#separator {{
    background-color: {BORDER};
    max-height: 1px;
}}
"""


def get_entity_color(label):
    if label == "O":
        return QColor("#666677")
    entity = label[2:] if label.startswith(("B-", "I-")) else label
    info = ENTITY_COLORS.get(entity)
    return QColor(info["hex"]) if info else QColor("#666677")


def pil_to_qpixmap(pil_image):
    if pil_image.mode != "RGB":
        pil_image = pil_image.convert("RGB")
    data = pil_image.tobytes("raw", "RGB")
    qimg = QImage(data, pil_image.width, pil_image.height, 3 * pil_image.width, QImage.Format.Format_RGB888)
    return QPixmap.fromImage(qimg)


class InvoiceView(QGraphicsView):
    def __init__(self, scene, parent=None):
        super().__init__(scene, parent)
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        self.setDragMode(QGraphicsView.DragMode.NoDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self._zoom = 0
        self._panning = False

    def wheelEvent(self, event: QWheelEvent):
        factor = 1.15
        if event.angleDelta().y() > 0:
            self._zoom += 1
            self.scale(factor, factor)
        else:
            self._zoom -= 1
            self.scale(1 / factor, 1 / factor)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.RightButton:
            self._panning = True
            self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
            fake = type(event)(
                event.type(), event.position(), event.globalPosition(),
                Qt.MouseButton.LeftButton, Qt.MouseButton.LeftButton, event.modifiers(),
            )
            super().mousePressEvent(fake)
            return
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.RightButton and self._panning:
            self._panning = False
            self.setDragMode(QGraphicsView.DragMode.NoDrag)
            return
        super().mouseReleaseEvent(event)

    def fit_page(self):
        self.resetTransform()
        self._zoom = 0
        self.fitInView(self.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)


class ClickableView(InvoiceView):
    def __init__(self, scene, app, parent=None):
        super().__init__(scene, parent)
        self.app = app

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and not self._panning:
            scene_pos = self.mapToScene(event.position().toPoint())
            self.app._on_scene_click(scene_pos)
            return
        super().mousePressEvent(event)


class AnnotationApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Invoice Annotation Tool")
        self.resize(1600, 950)

        self.pdf_path = None
        self.images = []
        self.current_page = 0
        self.ocr_results = []
        self.selected_label = LABELS[0]
        self.word_rects = []

        self._build_ui()

    def _build_ui(self):
        toolbar = QToolBar("Azioni")
        toolbar.setMovable(False)
        toolbar.setIconSize(QSize(18, 18))
        toolbar.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextOnly)
        self.addToolBar(toolbar)

        for text, slot in [
            ("Apri PDF", self.open_pdf),
            ("Importa Pre-annotazioni", self.import_preannotations),
            ("Salva", self.save_annotations),
        ]:
            act = QAction(text, self)
            act.triggered.connect(slot)
            toolbar.addAction(act)

        toolbar.addSeparator()

        for text, slot in [
            ("\u25C0  Pagina", self.prev_page),
            ("Pagina  \u25B6", self.next_page),
            ("Fit", lambda: self.view.fit_page()),
        ]:
            act = QAction(text, self)
            act.triggered.connect(slot)
            toolbar.addAction(act)

        toolbar.addSeparator()

        act_reset = QAction("Reset Pagina", self)
        act_reset.triggered.connect(self.reset_page_labels)
        toolbar.addAction(act_reset)

        # Scene + View
        self.scene = QGraphicsScene()
        self.view = ClickableView(self.scene, self)
        self.view.setMinimumWidth(700)
        self.setCentralWidget(self.view)

        # --- Pannello sinistro: etichette ---
        label_dock = QDockWidget("", self)
        label_dock.setFeatures(QDockWidget.DockWidgetFeature.NoDockWidgetFeatures)

        label_panel = QWidget()
        label_layout = QVBoxLayout(label_panel)
        label_layout.setContentsMargins(0, 0, 0, 0)
        label_layout.setSpacing(0)

        title = QLabel("ETICHETTE")
        title.setObjectName("sectionTitle")
        label_layout.addWidget(title)

        sep = QFrame()
        sep.setObjectName("separator")
        sep.setFrameShape(QFrame.Shape.HLine)
        label_layout.addWidget(sep)

        self.label_list = QListWidget()
        self.label_list.setFixedWidth(200)

        # "O" separato, poi raggruppati per entita'
        item_o = QListWidgetItem("  O  (Background)")
        item_o.setForeground(QBrush(QColor(TEXT_SECONDARY)))
        font = QFont("Helvetica Neue", 13)
        item_o.setFont(font)
        self.label_list.addItem(item_o)

        entities_added = set()
        for label in LABELS:
            if label == "O":
                continue
            entity = label[2:]
            color = get_entity_color(label)
            info = ENTITY_COLORS.get(entity, {})
            name = info.get("name", entity)

            if entity not in entities_added:
                entities_added.add(entity)
                spacer = QListWidgetItem("")
                spacer.setFlags(Qt.ItemFlag.NoItemFlags)
                spacer.setSizeHint(QSize(0, 6))
                self.label_list.addItem(spacer)

            prefix = "B" if label.startswith("B-") else "I"
            display = f"  {prefix}  {name}"
            item = QListWidgetItem(display)
            item.setForeground(QBrush(color))
            f = QFont("Helvetica Neue", 13)
            if prefix == "B":
                f.setBold(True)
            item.setFont(f)
            self.label_list.addItem(item)

        self.label_list.setCurrentRow(0)
        self.label_list.currentRowChanged.connect(self._on_label_changed)
        label_layout.addWidget(self.label_list)

        # Legenda colori compatta
        legend_title = QLabel("LEGENDA")
        legend_title.setObjectName("sectionTitle")
        label_layout.addWidget(legend_title)

        for entity, info in ENTITY_COLORS.items():
            leg = QLabel(f"  \u25CF  {info['name']}")
            leg.setStyleSheet(f"color: {info['hex']}; font-size: 11px; padding: 2px 8px;")
            label_layout.addWidget(leg)

        label_layout.addStretch()
        label_dock.setWidget(label_panel)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, label_dock)

        # --- Pannello destro: annotazioni ---
        ann_dock = QDockWidget("", self)
        ann_dock.setFeatures(QDockWidget.DockWidgetFeature.NoDockWidgetFeatures)

        ann_panel = QWidget()
        ann_layout = QVBoxLayout(ann_panel)
        ann_layout.setContentsMargins(0, 0, 0, 0)
        ann_layout.setSpacing(0)

        ann_title = QLabel("ANNOTAZIONI")
        ann_title.setObjectName("sectionTitle")
        ann_layout.addWidget(ann_title)

        sep2 = QFrame()
        sep2.setObjectName("separator")
        sep2.setFrameShape(QFrame.Shape.HLine)
        ann_layout.addWidget(sep2)

        self.stats_label = QLabel("Nessuna annotazione")
        self.stats_label.setObjectName("statsLabel")
        ann_layout.addWidget(self.stats_label)

        self.ann_tree = QTreeWidget()
        self.ann_tree.setHeaderLabels(["Parola", "Label", "Conf"])
        self.ann_tree.setColumnCount(3)
        self.ann_tree.setAlternatingRowColors(True)
        self.ann_tree.setRootIsDecorated(False)
        header = self.ann_tree.header()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        self.ann_tree.setFixedWidth(320)
        self.ann_tree.itemClicked.connect(self._on_annotation_clicked)
        ann_layout.addWidget(self.ann_tree)

        btn_delete = QPushButton("Rimuovi Label Selezionata")
        btn_delete.setObjectName("dangerBtn")
        btn_delete.clicked.connect(self.delete_selected_annotation)
        ann_layout.addWidget(btn_delete)

        ann_dock.setWidget(ann_panel)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, ann_dock)

        # Status bar
        self.status = QStatusBar()
        self.setStatusBar(self.status)
        self.status.showMessage("Apri un PDF per iniziare  |  Click SX = annota  |  Click DX = pan  |  Rotella = zoom")

    def _on_label_changed(self, row):
        real_idx = 0
        for i in range(row + 1):
            item = self.label_list.item(i)
            if item and not (item.flags() & Qt.ItemFlag.ItemIsEnabled) == Qt.ItemFlag.NoItemFlags:
                pass
        # Map row back to LABELS index
        valid_rows = []
        for i in range(self.label_list.count()):
            item = self.label_list.item(i)
            if item and item.flags() != Qt.ItemFlag.NoItemFlags:
                valid_rows.append(i)

        if row in valid_rows:
            label_idx = valid_rows.index(row)
            if 0 <= label_idx < len(LABELS):
                self.selected_label = LABELS[label_idx]

    # --- PDF / OCR ---

    def open_pdf(self):
        path, _ = QFileDialog.getOpenFileName(self, "Apri PDF", "input_data", "PDF (*.pdf)")
        if not path:
            return

        self.pdf_path = path
        self.current_page = 0
        self.images = []
        self.ocr_results = []

        try:
            self.images = convert_from_path(path, dpi=CONFIG["PDF_DPI"], poppler_path=CONFIG["POPPLER_PATH"])
            for img in self.images:
                ocr = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
                words_data = []
                for i in range(len(ocr["text"])):
                    word = ocr["text"][i].strip()
                    if word:
                        words_data.append({
                            "word": word,
                            "x1": ocr["left"][i],
                            "y1": ocr["top"][i],
                            "x2": ocr["left"][i] + ocr["width"][i],
                            "y2": ocr["top"][i] + ocr["height"][i],
                            "label": "O",
                            "confidence": None,
                        })
                self.ocr_results.append(words_data)

            self._show_page()
            self.view.fit_page()
            self.status.showMessage(
                f"{Path(path).name}  |  Pagina 1/{len(self.images)}  |  "
                f"{len(self.ocr_results[0])} parole OCR  |  Click SX = annota"
            )
        except Exception as e:
            QMessageBox.critical(self, "Errore", f"Errore apertura PDF:\n{e}")
            traceback.print_exc()

    # --- Rendering ---

    def _show_page(self):
        self.scene.clear()
        self.word_rects = []

        if not self.images:
            return

        img = self.images[self.current_page]
        pixmap = pil_to_qpixmap(img)
        self.scene.addPixmap(pixmap)
        self.scene.setSceneRect(QRectF(pixmap.rect().toRectF()))

        words_data = self.ocr_results[self.current_page]
        for idx, wd in enumerate(words_data):
            self._add_word_rect(idx, wd)

        self._update_annotation_table()

    def _add_word_rect(self, idx, wd):
        x1, y1, x2, y2 = wd["x1"], wd["y1"], wd["x2"], wd["y2"]
        w, h = x2 - x1, y2 - y1
        label = wd["label"]
        color = get_entity_color(label)

        if label != "O":
            pen = QPen(color, 2.5)
            pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
            rect = self.scene.addRect(x1, y1, w, h, pen)

            fill = QColor(color)
            fill.setAlpha(35)
            rect.setBrush(QBrush(fill))

            tag = label.split("-", 1)[1] if "-" in label else label
            prefix = "B" if label.startswith("B-") else "I"
            text_item = QGraphicsSimpleTextItem(f"{prefix}:{tag}")
            font = QFont("Helvetica Neue", 6, QFont.Weight.Bold)
            text_item.setFont(font)
            text_item.setBrush(QBrush(color))
            text_item.setPos(x1 + 1, y1 - 10)
            text_item.setZValue(3)
            self.scene.addItem(text_item)
        else:
            pen = QPen(QColor(100, 100, 120, 60), 0.5)
            rect = self.scene.addRect(x1, y1, w, h, pen)

        rect.setData(0, idx)
        rect.setZValue(1 if label == "O" else 2)
        rect.setCursor(Qt.CursorShape.PointingHandCursor)
        self.word_rects.append(rect)

    def _on_scene_click(self, scene_pos):
        items = self.scene.items(scene_pos)
        for item in items:
            if isinstance(item, QGraphicsRectItem):
                idx = item.data(0)
                if idx is not None:
                    words_data = self.ocr_results[self.current_page]
                    words_data[idx]["label"] = self.selected_label
                    words_data[idx]["confidence"] = None
                    self._show_page()
                    return

    # --- Annotation table ---

    def _update_annotation_table(self):
        self.ann_tree.clear()
        if not self.ocr_results:
            self.stats_label.setText("Nessuna annotazione")
            return

        words_data = self.ocr_results[self.current_page]
        count = 0
        entity_counts = {}

        for idx, wd in enumerate(words_data):
            if wd["label"] != "O":
                count += 1
                conf = f"{wd['confidence']:.2f}" if wd.get("confidence") is not None else ""
                item = QTreeWidgetItem([wd["word"], wd["label"], conf])
                item.setData(0, Qt.ItemDataRole.UserRole, idx)
                color = get_entity_color(wd["label"])
                item.setForeground(0, QBrush(QColor(TEXT_PRIMARY)))
                item.setForeground(1, QBrush(color))
                item.setForeground(2, QBrush(QColor(TEXT_SECONDARY)))
                self.ann_tree.addTopLevelItem(item)

                entity = wd["label"][2:] if wd["label"].startswith(("B-", "I-")) else wd["label"]
                entity_counts[entity] = entity_counts.get(entity, 0) + 1

        parts = [f"{count} label"]
        for ent, cnt in sorted(entity_counts.items()):
            parts.append(f"{ent}: {cnt}")
        self.stats_label.setText("  |  ".join(parts))

    def _on_annotation_clicked(self, item, column):
        idx = item.data(0, Qt.ItemDataRole.UserRole)
        if idx is not None:
            wd = self.ocr_results[self.current_page][idx]
            cx = (wd["x1"] + wd["x2"]) / 2
            cy = (wd["y1"] + wd["y2"]) / 2
            self.view.centerOn(cx, cy)

    def delete_selected_annotation(self):
        item = self.ann_tree.currentItem()
        if not item:
            return
        idx = item.data(0, Qt.ItemDataRole.UserRole)
        if idx is not None:
            self.ocr_results[self.current_page][idx]["label"] = "O"
            self.ocr_results[self.current_page][idx]["confidence"] = None
            self._show_page()

    def reset_page_labels(self):
        if not self.ocr_results:
            return
        for wd in self.ocr_results[self.current_page]:
            wd["label"] = "O"
            wd["confidence"] = None
        self._show_page()

    # --- Import / Export ---

    def import_preannotations(self):
        if not self.images:
            QMessageBox.information(self, "Info", "Apri prima un PDF")
            return

        path, _ = QFileDialog.getOpenFileName(
            self, "Importa Pre-annotazioni", "output_data/preannotations", "Excel (*.xlsx)"
        )
        if not path:
            return

        try:
            df = pd.read_excel(path)
            required = ["word", "x1", "y1", "x2", "y2", "label"]
            if not all(c in df.columns for c in required):
                QMessageBox.critical(self, "Errore", f"Colonne mancanti: {required}")
                return

            has_confidence = "confidence" in df.columns
            applied = 0

            for page_idx, words_data in enumerate(self.ocr_results):
                if "page_num" in df.columns:
                    page_df = df[df["page_num"] == page_idx + 1]
                elif "image_id" in df.columns:
                    page_df = df[df["image_id"] == page_idx]
                else:
                    page_df = df

                for _, row in page_df.iterrows():
                    pre_word = str(row["word"]).strip()
                    pre_label = str(row["label"])
                    if pre_label == "O":
                        continue

                    pre_x1, pre_y1 = int(row["x1"]), int(row["y1"])
                    best_idx, best_dist = None, float("inf")
                    for widx, wd in enumerate(words_data):
                        if wd["word"] == pre_word:
                            dist = abs(wd["x1"] - pre_x1) + abs(wd["y1"] - pre_y1)
                            if dist < best_dist:
                                best_dist = dist
                                best_idx = widx

                    if best_idx is not None and best_dist < 20:
                        words_data[best_idx]["label"] = pre_label
                        if has_confidence:
                            words_data[best_idx]["confidence"] = float(row["confidence"])
                        applied += 1

            self._show_page()
            QMessageBox.information(
                self, "Pre-annotazioni importate",
                f"Applicate {applied} label dal modello.\nCorreggi le label errate e salva.",
            )
        except Exception as e:
            QMessageBox.critical(self, "Errore", f"Errore importazione:\n{e}")
            traceback.print_exc()

    def save_annotations(self):
        if not self.images or not self.pdf_path:
            QMessageBox.information(self, "Info", "Nessun file aperto")
            return

        default_name = f"{Path(self.pdf_path).stem}_annotated.xlsx"
        path, _ = QFileDialog.getSaveFileName(
            self, "Salva annotazioni", f"output_data/annotations/{default_name}", "Excel (*.xlsx)"
        )
        if not path:
            return

        try:
            data = []
            for page_idx, words_data in enumerate(self.ocr_results):
                for wd in words_data:
                    if wd["label"] != "O":
                        data.append({
                            "image_id": page_idx,
                            "page_num": page_idx + 1,
                            "word": wd["word"],
                            "x1": wd["x1"],
                            "y1": wd["y1"],
                            "x2": wd["x2"],
                            "y2": wd["y2"],
                            "label": wd["label"],
                        })

            df = pd.DataFrame(data)
            os.makedirs(Path(path).parent, exist_ok=True)
            df.to_excel(path, index=False)
            QMessageBox.information(self, "Salvato", f"Annotazioni salvate: {len(data)} label\n{path}")
        except Exception as e:
            QMessageBox.critical(self, "Errore", f"Errore salvataggio:\n{e}")
            traceback.print_exc()

    # --- Navigation ---

    def prev_page(self):
        if self.images and self.current_page > 0:
            self.current_page -= 1
            self._show_page()
            self.view.fit_page()
            self._update_status()

    def next_page(self):
        if self.images and self.current_page < len(self.images) - 1:
            self.current_page += 1
            self._show_page()
            self.view.fit_page()
            self._update_status()

    def _update_status(self):
        if self.pdf_path and self.images:
            n_labels = sum(1 for wd in self.ocr_results[self.current_page] if wd["label"] != "O")
            self.status.showMessage(
                f"{Path(self.pdf_path).name}  |  "
                f"Pagina {self.current_page + 1}/{len(self.images)}  |  "
                f"{len(self.ocr_results[self.current_page])} parole  |  "
                f"{n_labels} annotate"
            )


def main():
    app = QApplication(sys.argv)
    app.setStyleSheet(STYLESHEET)

    window = AnnotationApp()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
