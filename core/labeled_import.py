"""已标记数据集导入：人物文件夹 → 人脸检测 + 多图特征综合 → labeled_persons 表（多线程）"""

import os
import threading
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from itertools import islice

import numpy as np
from PySide6.QtCore import QThread, Signal

from core.database import DatabaseManager
from core.face_engine import FaceEngine, FaceData, imread_unicode, get_cuda_recommended_workers
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


def _process_one_person(
    engine: FaceEngine,
    person_id: str,
    folder_path: str,
    image_paths: list[str],
) -> tuple[str, str, np.ndarray | None, int, str | None]:
    """在 worker 线程中处理单个人物，返回 (person_id, folder_path, feature, photo_count, error_msg)。

    若成功: feature 非 None，error_msg 为 None。
    若无有效人脸: feature 为 None，photo_count 为 0，error_msg 为 None。
    若异常: feature 为 None，error_msg 为异常信息。
    """
    try:
        feature, photo_count = compute_aggregated_feature(engine, image_paths)
        return (person_id, folder_path, feature, photo_count if feature is not None else 0, None)
    except Exception as e:
        return (person_id, folder_path, None, 0, str(e))


class LabeledImportWorker(QThread):
    """后台导入已标记数据集：多线程遍历人物文件夹，多图特征综合，写入 labeled_persons。"""

    progress = Signal(int, int, str)  # current, total, person_id
    finished_all = Signal(int)        # 实际导入的人物数
    error = Signal(str)

    def __init__(
        self,
        engine: FaceEngine,
        db: DatabaseManager,
        root_folder: str,
        num_workers: int | None = None,
    ):
        super().__init__()
        self.engine = engine
        self.db = db
        self.root_folder = root_folder
        self.logger = get_logger()
        self._cancelled = False
        if num_workers is None:
            if engine.backend_name == "CUDA":
                self.num_workers = get_cuda_recommended_workers()
            else:
                self.num_workers = max(os.cpu_count() or 4, 4)
        else:
            self.num_workers = num_workers

    def cancel(self):
        self._cancelled = True

    def run(self):
        items = list(load_labeled_dataset(self.root_folder))
        total = len(items)
        if total == 0:
            self.logger.warning("导入参考库: 根目录下无有效人物文件夹")
            self.finished_all.emit(0)
            return

        self.logger.info(f"导入参考库: {total} 个人物文件夹，线程数: {self.num_workers}")

        # 每个 worker 线程使用独立的 engine，避免并发冲突
        _local = threading.local()
        engines = [self.engine.clone() for _ in range(self.num_workers)]
        _engine_idx = [0]
        _idx_lock = threading.Lock()

        def get_thread_engine() -> FaceEngine:
            if not hasattr(_local, "engine"):
                with _idx_lock:
                    idx = _engine_idx[0]
                    _engine_idx[0] += 1
                _local.engine = engines[idx]
            return _local.engine

        def worker(args: tuple[str, str, list[str]]):
            person_id, folder_path, image_paths = args
            eng = get_thread_engine()
            return _process_one_person(eng, person_id, folder_path, image_paths)

        imported = 0
        processed = 0
        INFLIGHT_LIMIT = max(self.num_workers * 2, 4)

        with ThreadPoolExecutor(max_workers=self.num_workers) as pool:
            item_iter = iter(items)
            active: dict = {}  # future -> (person_id, folder_path, image_paths)

            # 初始提交一批任务
            for item in islice(item_iter, INFLIGHT_LIMIT):
                fut = pool.submit(worker, item)
                active[fut] = item

            while active:
                if self._cancelled:
                    for f in active:
                        f.cancel()
                    break
                done, _ = wait(active.keys(), return_when=FIRST_COMPLETED)
                for fut in done:
                    person_id, folder_path, image_paths = active.pop(fut)
                    processed += 1
                    self.progress.emit(processed, total, person_id)
                    try:
                        result = fut.result()
                        _pid, _path, feature, photo_count, err_msg = result
                        if err_msg:
                            self.logger.warning("导入参考库: 人物 %s 处理失败: %s", _pid, err_msg, exc_info=True)
                            self.error.emit(f"{_pid}: {err_msg}")
                        elif feature is None or photo_count == 0:
                            self.logger.warning("导入参考库: 人物 %s 无有效人脸，跳过", _pid)
                        else:
                            self.db.add_labeled_person(_pid, _path, feature, photo_count)
                            imported += 1
                    except Exception as e:
                        self.logger.warning("导入参考库: 人物 %s 处理失败: %s", person_id, e, exc_info=True)
                        self.error.emit(f"{person_id}: {e}")

                    # 补充新任务，保持并发度
                    try:
                        next_item = next(item_iter)
                        new_fut = pool.submit(worker, next_item)
                        active[new_fut] = next_item
                    except StopIteration:
                        pass

        self.logger.info(f"导入参考库完成: 成功 {imported}/{total} 人")
        self.finished_all.emit(imported)
