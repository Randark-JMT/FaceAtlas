"""已标记数据集导入：人物文件夹 → 人脸检测 + 多图特征综合 → labeled_persons 表"""

import os
from collections.abc import Iterator

import numpy as np
from PySide6.QtCore import QThread, Signal

from core.database import DatabaseManager
from core.face_engine import FaceEngine, FaceData, imread_unicode
from core.logger import get_logger

SUPPORTED_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


def load_labeled_dataset(root_folder: str) -> Iterator[tuple[str, str, list[str]]]:
    """遍历根目录下各子文件夹，返回 (人物编号, 文件夹路径, 该文件夹内图片路径列表)。

    文件夹名 = 人物编号，文件夹内为不定张数的大头照。
    """
    if not os.path.isdir(root_folder):
        return
    for entry in os.scandir(root_folder):
        if not entry.is_dir():
            continue
        paths = [
            os.path.join(entry.path, f)
            for f in os.listdir(entry.path)
            if os.path.isfile(os.path.join(entry.path, f))
            and os.path.splitext(f)[1].lower() in SUPPORTED_EXT
        ]
        if paths:
            yield entry.name, entry.path, sorted(paths)


def _pick_largest_face(faces: list[FaceData]) -> FaceData | None:
    """取面积最大的人脸。"""
    if not faces:
        return None
    return max(faces, key=lambda f: f.bbox[2] * f.bbox[3])


def compute_aggregated_feature(
    engine: FaceEngine,
    image_paths: list[str],
) -> tuple[np.ndarray | None, int]:
    """对多张大头照提取特征并综合（平均后 L2 归一化）。

    Returns:
        (综合特征向量, 成功参与的图片数)，若无人脸则 (None, 0)。
    """
    features: list[np.ndarray] = []
    for path in image_paths:
        img = imread_unicode(path)
        if img is None:
            continue
        faces = engine.detect(img)
        if not faces:
            continue
        face = _pick_largest_face(faces)
        if face is None:
            continue
        engine.extract_feature(img, face)
        if face.feature is not None and face.feature.size > 0:
            features.append(face.feature.flatten().astype(np.float32))

    if not features:
        return None, 0
    avg = np.mean(features, axis=0)
    norm = np.linalg.norm(avg)
    if norm < 1e-10:
        return None, 0
    avg = (avg / norm).astype(np.float32)
    return avg, len(features)


class LabeledImportWorker(QThread):
    """后台导入已标记数据集：遍历人物文件夹，多图特征综合，写入 labeled_persons。"""

    progress = Signal(int, int, str)  # current, total, person_id
    finished_all = Signal(int)        # 实际导入的人物数
    error = Signal(str)

    def __init__(
        self,
        engine: FaceEngine,
        db: DatabaseManager,
        root_folder: str,
    ):
        super().__init__()
        self.engine = engine
        self.db = db
        self.root_folder = root_folder
        self.logger = get_logger()
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    def run(self):
        items = list(load_labeled_dataset(self.root_folder))
        total = len(items)
        if total == 0:
            self.logger.warning("导入参考库: 根目录下无有效人物文件夹")
            self.finished_all.emit(0)
            return

        self.logger.info(f"导入参考库: {total} 个人物文件夹")
        imported = 0
        for i, (person_id, folder_path, image_paths) in enumerate(items):
            if self._cancelled:
                break
            self.progress.emit(i + 1, total, person_id)
            try:
                feature, photo_count = compute_aggregated_feature(self.engine, image_paths)
                if feature is None or photo_count == 0:
                    self.logger.warning(f"导入参考库: 人物 {person_id} 无有效人脸，跳过")
                    continue
                self.db.add_labeled_person(person_id, folder_path, feature, photo_count)
                imported += 1
            except Exception as e:
                self.logger.warning("导入参考库: 人物 %s 处理失败: %s", person_id, e, exc_info=True)
                self.error.emit(f"{person_id}: {e}")

        self.logger.info(f"导入参考库完成: 成功 {imported}/{total} 人")
        self.finished_all.emit(imported)
