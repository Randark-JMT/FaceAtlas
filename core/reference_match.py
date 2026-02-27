"""参考库匹配：将无标记人脸与 labeled_persons 相似度比对，匹配则标记人物，未匹配则标记未知"""

import numpy as np
from PySide6.QtCore import QThread, Signal

from core.database import DatabaseManager
from core.logger import get_logger


class ReferenceMatchWorker(QThread):
    """后台参考库匹配：无标记人脸与 labeled_persons 向量化相似度计算，更新 face.person_id。"""

    progress = Signal(int, int, str)  # current, total, stage_text
    finished_match = Signal(dict)     # {matched: int, unknown: int}
    error = Signal(str)

    def __init__(
        self,
        db: DatabaseManager,
        cosine_threshold: float = 0.60,
    ):
        super().__init__()
        self.db = db
        self.threshold = cosine_threshold
        self.logger = get_logger()
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    def run(self):
        refs = self.db.get_labeled_persons_with_features()
        faces = self.db.get_unassigned_faces_with_features()

        if not refs:
            self.logger.warning("参考库匹配: 无参考人物，请先导入参考库")
            self.finished_match.emit({"matched": 0, "unknown": 0})
            return

        if not faces:
            self.logger.info("参考库匹配: 无待匹配人脸")
            self.finished_match.emit({"matched": 0, "unknown": 0})
            return

        self.progress.emit(0, 1, "加载参考库特征...")
        ref_ids = [r["id"] for r in refs]
        ref_names = {r["id"]: r["person_id"] for r in refs}
        ref_matrix = np.vstack([
            DatabaseManager.feature_from_blob(r["feature"]).flatten().astype(np.float32)
            for r in refs
        ])
        norms = np.linalg.norm(ref_matrix, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-10)
        ref_matrix = ref_matrix / norms  # (n_ref, 512)

        self.progress.emit(0, 1, "加载人脸特征...")
        face_ids = [f["id"] for f in faces]
        face_matrix = np.vstack([
            DatabaseManager.feature_from_blob(f["feature"]).flatten().astype(np.float32)
            for f in faces
        ])
        fnorms = np.linalg.norm(face_matrix, axis=1, keepdims=True)
        fnorms = np.maximum(fnorms, 1e-10)
        face_matrix = face_matrix / fnorms  # (n_face, 512)

        self.progress.emit(0, 1, "计算相似度...")
        sim = ref_matrix @ face_matrix.T  # (n_ref, n_face)
        # 每列对应一张人脸，取最大相似度及对应 ref 索引
        max_sim_per_face = np.max(sim, axis=0)
        best_ref_idx_per_face = np.argmax(sim, axis=0)

        unknown_person_id = self.db.get_or_create_unknown_person()
        updates: list[tuple[int, int]] = []  # (person_id, face_id) 供 batch_update_face_persons
        matched = 0
        unknown_count = 0

        for i, face_id in enumerate(face_ids):
            if self._cancelled:
                break
            sim_val = float(max_sim_per_face[i])
            ref_idx = int(best_ref_idx_per_face[i])
            if sim_val >= self.threshold:
                ref_label_id = ref_ids[ref_idx]
                person_name = ref_names[ref_label_id]
                person_id = self.db.get_or_create_person_by_name(person_name)
                updates.append((person_id, face_id))
                matched += 1
            else:
                updates.append((unknown_person_id, face_id))
                unknown_count += 1

        self.progress.emit(1, 1, "写入数据库...")
        self.db.batch_update_face_persons(updates)

        # 更新 persons.face_count
        for person_id in set(p for p, _ in updates):
            count = self.db.get_person_face_count(person_id)
            self.db.update_person_face_count(person_id, count)

        self.logger.info(
            f"参考库匹配完成: 匹配 {matched} 张，未知 {unknown_count} 张，阈值={self.threshold:.2f}"
        )
        self.finished_match.emit({"matched": matched, "unknown": unknown_count})
