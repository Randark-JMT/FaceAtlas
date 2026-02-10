"""人物归类面板 - 按人物分组展示人脸"""

import cv2
import numpy as np
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QScrollArea,
    QFrame, QLineEdit, QApplication, QPushButton, QComboBox,
)
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtCore import Qt, Signal

from core.database import DatabaseManager
from core.face_engine import FaceEngine, imread_unicode

# 每个人物分组最多显示的缩略图数量
MAX_THUMBS_PER_GROUP = 8


class ClickableThumb(QLabel):
    """可双击的人脸缩略图"""

    double_clicked = Signal(int)  # image_id

    def __init__(self, image_id: int, parent=None):
        super().__init__(parent)
        self._image_id = image_id
        self.setCursor(Qt.CursorShape.PointingHandCursor)

    def mouseDoubleClickEvent(self, event):
        self.double_clicked.emit(self._image_id)


class ClickableMoreLabel(QLabel):
    """可双击的"+N"展开标签"""
    
    double_clicked = Signal()
    
    def __init__(self, text: str, parent=None):
        super().__init__(text, parent)
        self.setFixedSize(56, 56)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setStyleSheet(
            "border: 1px solid #555; color: #4a9eff; font-size: 14px; font-weight: bold;"
        )
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setToolTip("双击加载更多（每次10张）")
    
    def mouseDoubleClickEvent(self, event):
        self.double_clicked.emit()


class PersonGroup(QFrame):
    """单个人物的折叠分组"""

    name_changed = Signal(int, str)  # person_id, new_name
    face_double_clicked = Signal(int)  # image_id

    def __init__(self, person_id: int, name: str, face_rows: list, parent=None):
        super().__init__(parent)
        self.person_id = person_id
        self.face_rows = face_rows  # 保存所有人脸数据
        self._shown_count = MAX_THUMBS_PER_GROUP  # 当前显示的数量
        
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setStyleSheet(
            "PersonGroup { background: #2a2a2a; border: 1px solid #444; "
            "border-radius: 4px; margin: 2px; }"
        )

        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(8, 8, 8, 8)
        self.layout.setSpacing(6)

        # 标题行：编号 + 名称编辑 + 人脸数
        header = QHBoxLayout()
        # 显示人物编号
        person_id_label = QLabel(f"P{person_id}")
        person_id_label.setStyleSheet("font-weight: bold; color: #4a9eff; font-size: 13px;")
        person_id_label.setFixedWidth(40)
        header.addWidget(person_id_label)
        
        self._name_edit = QLineEdit(name)
        self._name_edit.setFixedWidth(120)
        self._name_edit.setStyleSheet("background: #3a3a3a; border: 1px solid #555; padding: 2px 4px;")
        self._name_edit.editingFinished.connect(self._on_name_changed)
        header.addWidget(self._name_edit)
        header.addWidget(QLabel(f"({len(face_rows)} 张人脸)"))
        header.addStretch()
        self.layout.addLayout(header)

        # 人脸缩略图容器（可动态更新）
        self.thumb_container = QWidget()
        self.thumb_layout = QHBoxLayout(self.thumb_container)
        self.thumb_layout.setSpacing(4)
        self.thumb_layout.setContentsMargins(0, 0, 0, 0)
        self.layout.addWidget(self.thumb_container)
        
        # 初始显示
        self._update_thumbs()

    @staticmethod
    def _make_thumb(row) -> ClickableThumb:
        thumb = ClickableThumb(row["image_id"])
        thumb.setFixedSize(56, 56)
        thumb.setStyleSheet("border: 1px solid #555;")
        thumb.setToolTip(f"双击跳转 | 来源: {row['filename']}")

        img = imread_unicode(row["file_path"])
        if img is not None:
            bbox = (row["bbox_x"], row["bbox_y"], row["bbox_w"], row["bbox_h"])
            crop = FaceEngine.crop_face(img, bbox)
            pix = _cv_to_pixmap(crop, 56, 56)
            thumb.setPixmap(pix)
        else:
            thumb.setText("?")
            thumb.setAlignment(Qt.AlignmentFlag.AlignCenter)
        return thumb

    def _update_thumbs(self):
        """更新缩略图显示"""
        # 清空现有缩略图
        while self.thumb_layout.count():
            item = self.thumb_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        
        # 显示当前数量的人脸
        shown = self.face_rows[:self._shown_count]
        for row in shown:
            thumb = self._make_thumb(row)
            thumb.double_clicked.connect(self.face_double_clicked)
            self.thumb_layout.addWidget(thumb)

        # 如果还有更多人脸，显示+N标签
        overflow = len(self.face_rows) - len(shown)
        if overflow > 0:
            more_label = ClickableMoreLabel(f"+{overflow}")
            more_label.double_clicked.connect(self._on_show_more)
            self.thumb_layout.addWidget(more_label)

        self.thumb_layout.addStretch()
    
    def _on_show_more(self):
        """双击+号，展开更多人脸"""
        self._shown_count = min(self._shown_count + 10, len(self.face_rows))
        self._update_thumbs()

    def _on_name_changed(self):
        self.name_changed.emit(self.person_id, self._name_edit.text().strip())


def _cv_to_pixmap(cv_img: np.ndarray, w: int, h: int) -> QPixmap:
    rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    ih, iw, ch = rgb.shape
    qimg = QImage(rgb.data, iw, ih, ch * iw, QImage.Format.Format_RGB888)
    return QPixmap.fromImage(qimg).scaled(
        w, h,
        Qt.AspectRatioMode.KeepAspectRatio,
        Qt.TransformationMode.SmoothTransformation,
    )


class PersonPanel(QWidget):
    """人物归类面板"""

    navigate_to_image = Signal(int)  # image_id — 外部连接此信号实现跳转

    def __init__(self, db: DatabaseManager, parent=None):
        super().__init__(parent)
        self.db = db
        
        # 排序状态
        self._sort_by = "id"  # "id" 或 "count"
        self._sort_order = "asc"  # "asc" 或 "desc"

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # 标题栏 + 排序控件
        header_layout = QHBoxLayout()
        header_layout.setSpacing(8)
        
        title = QLabel("人物归类")
        title.setStyleSheet("font-weight: bold; font-size: 14px; padding: 4px;")
        header_layout.addWidget(title)
        
        header_layout.addStretch()
        
        # 排序方式选择
        sort_label = QLabel("排序:")
        sort_label.setStyleSheet("font-size: 12px;")
        header_layout.addWidget(sort_label)
        
        self._sort_combo = QComboBox()
        self._sort_combo.addItem("编号", "id")
        self._sort_combo.addItem("出现次数", "count")
        self._sort_combo.setStyleSheet(
            "QComboBox { background: #2a2a2a; border: 1px solid #555; padding: 2px 6px; }"
            "QComboBox::drop-down { border: none; }"
            "QComboBox QAbstractItemView { background: #2a2a2a; border: 1px solid #555; }"
        )
        self._sort_combo.currentIndexChanged.connect(self._on_sort_changed)
        header_layout.addWidget(self._sort_combo)
        
        # 排序顺序切换按钮
        self._order_btn = QPushButton("↑")
        self._order_btn.setFixedSize(30, 24)
        self._order_btn.setToolTip("切换升序/降序")
        self._order_btn.setStyleSheet(
            "QPushButton { background: #2a2a2a; border: 1px solid #555; font-size: 16px; }"
            "QPushButton:hover { background: #3a3a3a; }"
        )
        self._order_btn.clicked.connect(self._toggle_sort_order)
        header_layout.addWidget(self._order_btn)
        
        layout.addLayout(header_layout)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._container = QWidget()
        self._container_layout = QVBoxLayout(self._container)
        self._container_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self._container_layout.setSpacing(4)
        scroll.setWidget(self._container)
        layout.addWidget(scroll)

    def refresh(self):
        """从数据库重新加载人物分组（逐个添加，保持 UI 响应）"""
        # 清空
        while self._container_layout.count():
            item = self._container_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        persons = self.db.get_all_persons()
        if not persons:
            placeholder = QLabel("尚未进行人脸归类")
            placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
            placeholder.setStyleSheet("color: #888; padding: 20px;")
            self._container_layout.addWidget(placeholder)
            return

        # 应用排序
        persons = self._sort_persons(persons)

        for i, person in enumerate(persons):
            face_rows = self.db.get_faces_by_person(person["id"])
            if not face_rows:
                continue
            group = PersonGroup(person["id"], person["name"], face_rows)
            group.name_changed.connect(self._on_name_changed)
            group.face_double_clicked.connect(self.navigate_to_image)
            self._container_layout.addWidget(group)
            if (i + 1) % 3 == 0:
                QApplication.processEvents()

    def _on_name_changed(self, person_id: int, new_name: str):
        if new_name:
            self.db.update_person_name(person_id, new_name)
    
    def _on_sort_changed(self):
        """排序方式改变"""
        self._sort_by = self._sort_combo.currentData()
        self.refresh()
    
    def _toggle_sort_order(self):
        """切换排序顺序"""
        if self._sort_order == "asc":
            self._sort_order = "desc"
            self._order_btn.setText("↓")
        else:
            self._sort_order = "asc"
            self._order_btn.setText("↑")
        self.refresh()
    
    def _sort_persons(self, persons: list) -> list:
        """根据当前排序设置对人物列表进行排序"""
        if self._sort_by == "id":
            key_func = lambda p: p["id"]
        else:  # count
            key_func = lambda p: p["face_count"]
        
        reverse = (self._sort_order == "desc")
        return sorted(persons, key=key_func, reverse=reverse)
