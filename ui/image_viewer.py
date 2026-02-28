"""可缩放图像显示控件（性能优化版）

优化：不再保存完整 cv_image 副本，只保留 QPixmap，减少 ~1/3 内存占用。
支持：滚轮缩放、拖拽平移、双击恢复默认。
"""

import cv2
import numpy as np
from PySide6.QtWidgets import QLabel, QSizePolicy
from PySide6.QtGui import QImage, QPixmap, QPainter
from PySide6.QtCore import Qt, Signal


class ImageViewer(QLabel):
    """显示 OpenCV 图像的控件，支持滚轮缩放、拖拽平移、双击恢复"""

    # zoom, pan_x, pan_y
    transform_changed = Signal(float, float, float)

    ZOOM_MIN = 0.2
    ZOOM_MAX = 5.0
    ZOOM_STEP = 1.15

    def __init__(self, placeholder: str = "无图像", parent=None):
        super().__init__(parent)
        self._pixmap: QPixmap | None = None
        self._placeholder = placeholder
        self._zoom = 1.0
        self._pan_x = 0
        self._pan_y = 0
        self._drag_start: tuple[int, int] | None = None
        self._pan_start: tuple[int, int] | None = None
        self._syncing = False  # 避免同步时触发递归

        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMinimumSize(200, 150)
        self.setStyleSheet("background-color: #1e1e1e; color: #888; border: 1px solid #333;")
        self.setText(self._placeholder)

    def set_image(self, cv_image: np.ndarray | None):
        """设置 OpenCV BGR 图像"""
        if cv_image is None:
            self._pixmap = None
            self.setText(self._placeholder)
            return

        # 直接转换为 QPixmap，不保存 cv_image 引用
        rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
        self._pixmap = QPixmap.fromImage(qimg)
        self._reset_transform()
        self._update_display()

    def _reset_transform(self):
        """恢复默认缩放与平移"""
        self._zoom = 1.0
        self._pan_x = 0
        self._pan_y = 0

    def set_transform(self, zoom: float, pan_x: float, pan_y: float):
        """由外部调用以同步另一侧的变换（不发射信号）"""
        self._syncing = True
        try:
            self._zoom = max(self.ZOOM_MIN, min(self.ZOOM_MAX, zoom))
            self._pan_x = pan_x
            self._pan_y = pan_y
            self._update_display()
        finally:
            self._syncing = False

    def _emit_transform(self):
        if not self._syncing:
            self.transform_changed.emit(self._zoom, self._pan_x, self._pan_y)

    def _update_display(self):
        self.update()

    def paintEvent(self, event):
        if self._pixmap is None or self._pixmap.isNull():
            super().paintEvent(event)
            return

        pw, ph = self._pixmap.width(), self._pixmap.height()
        vw, vh = self.width(), self.height()
        if pw <= 0 or ph <= 0 or vw <= 0 or vh <= 0:
            return

        fit_scale = min(vw / pw, vh / ph)
        display_scale = fit_scale * self._zoom
        dw = int(pw * display_scale)
        dh = int(ph * display_scale)
        scaled = self._pixmap.scaled(
            dw, dh,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        x = (vw - scaled.width()) // 2 + self._pan_x
        y = (vh - scaled.height()) // 2 + self._pan_y

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        painter.fillRect(self.rect(), self.palette().color(self.backgroundRole()))
        painter.drawPixmap(x, y, scaled)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._update_display()

    def wheelEvent(self, event):
        if self._pixmap is None:
            super().wheelEvent(event)
            return
        delta = event.angleDelta().y()
        if delta == 0:
            super().wheelEvent(event)
            return
        factor = self.ZOOM_STEP if delta > 0 else 1.0 / self.ZOOM_STEP
        old_zoom = self._zoom
        self._zoom = max(self.ZOOM_MIN, min(self.ZOOM_MAX, self._zoom * factor))
        # 以鼠标位置为缩放中心
        pos = event.position().toPoint()
        cx, cy = self.width() // 2, self.height() // 2
        dx = pos.x() - cx
        dy = pos.y() - cy
        self._pan_x += dx * (1 - self._zoom / old_zoom)
        self._pan_y += dy * (1 - self._zoom / old_zoom)
        self._update_display()
        self._emit_transform()
        event.accept()

    def mousePressEvent(self, event):
        if self._pixmap is None:
            super().mousePressEvent(event)
            return
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_start = (event.position().toPoint().x(), event.position().toPoint().y())
            self._pan_start = (self._pan_x, self._pan_y)
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            event.accept()
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._pixmap is None:
            super().mouseMoveEvent(event)
            return
        if self._drag_start is not None and self._pan_start is not None:
            x, y = event.position().toPoint().x(), event.position().toPoint().y()
            dx = x - self._drag_start[0]
            dy = y - self._drag_start[1]
            self._pan_x = self._pan_start[0] + dx
            self._pan_y = self._pan_start[1] + dy
            self._update_display()
            self._emit_transform()
            event.accept()
        else:
            if self._pixmap is not None:
                self.setCursor(Qt.CursorShape.OpenHandCursor)
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_start = None
            self._pan_start = None
            self.unsetCursor()
            event.accept()
        else:
            super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event):
        if self._pixmap is None:
            super().mouseDoubleClickEvent(event)
            return
        if event.button() == Qt.MouseButton.LeftButton:
            self._reset_transform()
            self._update_display()
            self._emit_transform()
            event.accept()
        else:
            super().mouseDoubleClickEvent(event)

    def enterEvent(self, event):
        if self._pixmap is not None and self._drag_start is None:
            self.setCursor(Qt.CursorShape.OpenHandCursor)
        super().enterEvent(event)

    def leaveEvent(self, event):
        self.unsetCursor()
        super().leaveEvent(event)

    def clear_image(self):
        self.set_image(None)
