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
    QApplication, QMainWindow, QGraphicsView, QGraphicsScene, QGraphicsPixmapItem,
    QGraphicsRectItem, QGraphicsSimpleTextItem, QDockWidget, QListWidget,
    QListWidgetItem, QToolBar, QFileDialog, QMessageBox, QStatusBar,
    QTreeWidget, QTreeWidgetItem, QHeaderView, QWidget, QVBoxLayout,
    QPushButton, QLabel, QSplitter,
)
from PyQt6.QtGui import (
    QPixmap, QImage, QPen, QColor, QBrush, QFont, QAction, QWheelEvent,
    QPainter,
)
from PyQt6.QtCore import Qt, QRectF

from config import CONFIG, LABELS
from pdf2image import convert_from_path

pytesseract.pytesseract.tesseract_cmd = CONFIG["TESSERACT_PATH"]

COLORS = {
    "VENDOR": "#ff6666",
    "CUSTOMER": "#6666ff",
    "DATE": "#66ff66",
    "TOTAL": "#ff66ff",
    "ITEM": "#ffaa66",
    "QUANTITY": "#66ffff",
    "PRICE": "#ff66aa",
    "INVOICE_NUMBER": "#cc8844",
    "O": "#999999",
}


def get_entity_color(label):
    if label == "O":
        return QColor(COLORS["O"])
    entity = label[2:] if label.startswith(("B-", "I-")) else label
    return QColor(COLORS.get(entity, "#999999"))


def pil_to_qpixmap(pil_image):
    if pil_image.mode != "RGB":
        pil_image = pil_image.convert("RGB")
    data = pil_image.tobytes("raw", "RGB")
    qimg = QImage(data, pil_image.width, pil_image.height, 3 * pil_image.width, QImage.Format.Format_RGB888)
    return QPixmap.fromImage(qimg)


class InvoiceView(QGraphicsView):
    """QGraphicsView con zoom via rotella e pan con click destro."""

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


class AnnotationApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Annotazione Fatture")
        self.resize(1500, 900)

        self.pdf_path = None
        self.images = []
        self.current_page = 0
        self.ocr_results = []
        self.selected_label = LABELS[0]
        self.word_rects = []

        self._build_ui()

    def _build_ui(self):
        # Toolbar
        toolbar = QToolBar("Azioni")
        toolbar.setMovable(False)
        self.addToolBar(toolbar)

        act_open = QAction("Apri PDF", self)
        act_open.triggered.connect(self.open_pdf)
        toolbar.addAction(act_open)

        act_import = QAction("Importa Pre-annotazioni", self)
        act_import.triggered.connect(self.import_preannotations)
        toolbar.addAction(act_import)

        act_save = QAction("Salva", self)
        act_save.triggered.connect(self.save_annotations)
        toolbar.addAction(act_save)

        toolbar.addSeparator()

        act_prev = QAction("< Pagina", self)
        act_prev.triggered.connect(self.prev_page)
        toolbar.addAction(act_prev)

        act_next = QAction("Pagina >", self)
        act_next.triggered.connect(self.next_page)
        toolbar.addAction(act_next)

        act_fit = QAction("Fit", self)
        act_fit.triggered.connect(lambda: self.view.fit_page())
        toolbar.addAction(act_fit)

        act_reset = QAction("Reset Label Pagina", self)
        act_reset.triggered.connect(self.reset_page_labels)
        toolbar.addAction(act_reset)

        # Scene e View
        self.scene = QGraphicsScene()
        self.view = InvoiceView(self.scene, self)
        self.view.setMinimumWidth(700)

        # Pannello etichette (sinistra)
        label_dock = QDockWidget("Etichette", self)
        label_dock.setFeatures(QDockWidget.DockWidgetFeature.NoDockWidgetFeatures)
        self.label_list = QListWidget()
        self.label_list.setFixedWidth(180)
        for label in LABELS:
            item = QListWidgetItem(label)
            color = get_entity_color(label)
            item.setForeground(QBrush(color if label != "O" else QColor("#333333")))
            font = QFont()
            font.setBold(True)
            item.setFont(font)
            self.label_list.addItem(item)
        self.label_list.setCurrentRow(0)
        self.label_list.currentRowChanged.connect(self._on_label_changed)
        label_dock.setWidget(self.label_list)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, label_dock)

        # Tabella annotazioni (destra)
        ann_dock = QDockWidget("Annotazioni", self)
        ann_dock.setFeatures(QDockWidget.DockWidgetFeature.NoDockWidgetFeatures)
        ann_widget = QWidget()
        ann_layout = QVBoxLayout(ann_widget)
        ann_layout.setContentsMargins(2, 2, 2, 2)

        self.ann_tree = QTreeWidget()
        self.ann_tree.setHeaderLabels(["Parola", "Label", "Conf"])
        self.ann_tree.setColumnCount(3)
        header = self.ann_tree.header()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        self.ann_tree.setFixedWidth(300)
        self.ann_tree.itemClicked.connect(self._on_annotation_clicked)
        ann_layout.addWidget(self.ann_tree)

        btn_delete = QPushButton("Rimuovi Label")
        btn_delete.clicked.connect(self.delete_selected_annotation)
        ann_layout.addWidget(btn_delete)

        ann_dock.setWidget(ann_widget)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, ann_dock)

        self.setCentralWidget(self.view)

        # Status bar
        self.status = QStatusBar()
        self.setStatusBar(self.status)
        self.status.showMessage("Apri un PDF per iniziare")

    def _on_label_changed(self, row):
        if 0 <= row < len(LABELS):
            self.selected_label = LABELS[row]

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
            self.status.showMessage(f"{Path(path).name} | Pagina 1/{len(self.images)} | {len(self.ocr_results[0])} parole")
        except Exception as e:
            QMessageBox.critical(self, "Errore", f"Errore apertura PDF:\n{e}")
            traceback.print_exc()

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
        label = wd["label"]
        color = get_entity_color(label)

        pen = QPen(color, 2 if label != "O" else 0.5)
        rect = self.scene.addRect(x1, y1, x2 - x1, y2 - y1, pen)
        rect.setData(0, idx)
        rect.setZValue(1)
        rect.setCursor(Qt.CursorShape.PointingHandCursor)

        if label != "O":
            fill = QColor(color)
            fill.setAlpha(40)
            rect.setBrush(QBrush(fill))

            text_item = QGraphicsSimpleTextItem(label)
            font = QFont("Arial", 7)
            text_item.setFont(font)
            text_item.setBrush(QBrush(color))
            text_item.setPos(x1, y1 - 12)
            text_item.setZValue(2)
            self.scene.addItem(text_item)

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

    def mousePressEvent_view(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            scene_pos = self.view.mapToScene(event.position().toPoint())
            self._on_scene_click(scene_pos)

    def _update_annotation_table(self):
        self.ann_tree.clear()
        if not self.ocr_results:
            return

        words_data = self.ocr_results[self.current_page]
        for idx, wd in enumerate(words_data):
            if wd["label"] != "O":
                conf = ""
                if wd.get("confidence") is not None:
                    conf = f"{wd['confidence']:.2f}"
                item = QTreeWidgetItem([wd["word"], wd["label"], conf])
                item.setData(0, Qt.ItemDataRole.UserRole, idx)
                color = get_entity_color(wd["label"])
                item.setForeground(1, QBrush(color))
                self.ann_tree.addTopLevelItem(item)

    def _on_annotation_clicked(self, item, column):
        idx = item.data(0, Qt.ItemDataRole.UserRole)
        if idx is not None and idx < len(self.word_rects):
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

    def prev_page(self):
        if self.images and self.current_page > 0:
            self.current_page -= 1
            self._show_page()
            self.view.fit_page()
            self.status.showMessage(f"Pagina {self.current_page + 1}/{len(self.images)}")

    def next_page(self):
        if self.images and self.current_page < len(self.images) - 1:
            self.current_page += 1
            self._show_page()
            self.view.fit_page()
            self.status.showMessage(f"Pagina {self.current_page + 1}/{len(self.images)}")


class ClickableView(InvoiceView):
    """View che intercetta click sinistro per annotare."""

    def __init__(self, scene, app, parent=None):
        super().__init__(scene, parent)
        self.app = app

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and not self._panning:
            scene_pos = self.mapToScene(event.position().toPoint())
            self.app._on_scene_click(scene_pos)
            return
        super().mousePressEvent(event)


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    window = AnnotationApp()
    # Sostituisci la view con ClickableView
    old_view = window.view
    window.view = ClickableView(window.scene, window)
    window.view.setMinimumWidth(700)
    window.setCentralWidget(window.view)

    window.show()
    window.status.showMessage("Apri un PDF per iniziare | Click sinistro = annota | Click destro = pan | Rotella = zoom")
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
