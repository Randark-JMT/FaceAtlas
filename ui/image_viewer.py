"""可缩放图像显示控件"""

import cv2
import numpy as np
from PySide6.QtWidgets import QLabel, QSizePolicy
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtCore import Qt


class ImageViewer(QLabel):
    """显示 OpenCV 图像的控件，自动按比例缩放"""

    def __init__(self, placeholder: str = "无图像", parent=None):
        super().__init__(parent)
        self._cv_image: np.ndarray | None = None
        self._pixmap: QPixmap | None = None
        self._placeholder = placeholder

        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMinimumSize(200, 150)
        self.setStyleSheet("background-color: #1e1e1e; color: #888; border: 1px solid #333;")
        self.setText(self._placeholder)

    def set_image(self, cv_image: np.ndarray | None):
        """设置 OpenCV BGR 图像"""
        if cv_image is None:
            self._cv_image = None
            self._pixmap = None
            self.setText(self._placeholder)
            return

        self._cv_image = cv_image
        rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
        self._pixmap = QPixmap.fromImage(qimg)
        self._update_display()

    def _update_display(self):
        if self._pixmap is None:
            return
        scaled = self._pixmap.scaled(
            self.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.setPixmap(scaled)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._update_display()

    def clear_image(self):
        self.set_image(None)
