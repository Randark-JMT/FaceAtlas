""" 人脸聚类模块 - 基于向量化余弦相似度 + Union-Find 将相似人脸归为同一人"""

from collections import defaultdict

import numpy as np

from core.database import DatabaseManager
from core.logger import get_logger

# 人脸分批大小，百万级时降低内存峰值
FACE_CHUNK_SIZE = 10000


class UnionFind:
    """并查集（路径压缩 + 按秩合并）"""

    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]  # 路径压半
            x = self.parent[x]
        return x

    def union(self, x: int, y: int):
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return
        if self.rank[rx] < self.rank[ry]:
            rx, ry = ry, rx
        self.parent[ry] = rx
        if self.rank[rx] == self.rank[ry]:
            self.rank[rx] += 1


class FaceCluster:
    """人脸聚类器（向量化加速版）"""

    # 分块大小：控制内存占用，每块最多占 BLOCK * n * 4 字节
    BLOCK_SIZE = 512

    def __init__(self, db: DatabaseManager, recognizer=None):
        self.db = db
        # recognizer 保留兼容，但不再用于逐对 match
        self.recognizer = recognizer
        self.logger = get_logger()

    def cluster(self, cosine_threshold: float = 0.60,
                progress_cb=None, incremental: bool = True) -> dict[int, list[int]]:
        """
        对数据库中所有有特征的人脸进行聚类。
        使用 numpy 向量化矩阵乘法计算余弦相似度，性能远优于逐对调用。

        Args:
            cosine_threshold: 余弦相似度阈值（FaceNet 512 维常用约 0.5~0.7，原 SFace 约 0.363）
            progress_cb: 进度回调 (current, total, stage_text)
            incremental: 是否增量聚类（保留已有人物归类）

        Returns:
            {person_id: [face_id, ...]} 聚类结果
        """

        def _report(current, total, text):
            if progress_cb:
                progress_cb(current, total, text)
        
        self.logger.info(f"开始人脸聚类，阈值={cosine_threshold:.3f}，增量模式={incremental}")

        if incremental:
            # 增量模式：只清除未命名的人物
            _report(0, 1, "清除未命名人物...")
            self.db.clear_all_persons(keep_named=True)
            
            # 获取已有人物及其特征（直接从 row 读取，避免逐条 get_person_feature）
            _report(0, 1, "加载已有人物特征...")
            existing_persons = self.db.get_persons_with_features()
            person_features: dict[int, np.ndarray] = {}
            for person in existing_persons:
                if person.get("feature"):
                    feat = DatabaseManager.feature_from_blob(person["feature"])
                    if feat is not None:
                        person_features[person["id"]] = feat.flatten()
            
            row_iter = self.db.iter_unassigned_faces_with_features(FACE_CHUNK_SIZE)
            self.logger.info(f"增量模式：已有 {len(existing_persons)} 个人物，将分批匹配未分配人脸")
        else:
            # 全量模式：清除所有旧归类
            _report(0, 1, "清除旧归类数据...")
            self.db.clear_all_persons(keep_named=False)
            row_iter = None
            person_features = {}

        assigned_to_existing: dict[int, list[int]] = defaultdict(list)
        unmatched_rows: list = []

        if incremental and person_features:
            # 分批加载 + 内存矩阵比对（避免 pgvector 百万次 DB 往返）
            person_ids_list = list(person_features.keys())
            person_feat_matrix = np.vstack([person_features[pid] for pid in person_ids_list])
            person_feat_matrix = person_feat_matrix.astype(np.float32)
            norms = np.linalg.norm(person_feat_matrix, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-10)
            person_feat_matrix /= norms

            processed = 0
            matched_total = 0
            for chunk in row_iter:
                _report(processed, processed + len(chunk), f"与已有人物匹配: {processed + len(chunk)} 张...")
                face_feat_list = []
                for row in chunk:
                    feat = DatabaseManager.feature_from_blob(row["feature"])
                    face_feat_list.append(feat.flatten().astype(np.float32))
                face_feat_matrix = np.vstack(face_feat_list)
                fnorms = np.linalg.norm(face_feat_matrix, axis=1, keepdims=True)
                fnorms = np.maximum(fnorms, 1e-10)
                face_feat_matrix /= fnorms

                similarities = person_feat_matrix @ face_feat_matrix.T
                best_person_idx_per_face = np.argmax(similarities, axis=0)
                max_sim_per_face = np.max(similarities, axis=0)

                for i, row in enumerate(chunk):
                    max_sim = float(max_sim_per_face[i])
                    if max_sim >= cosine_threshold:
                        best_person_id = person_ids_list[int(best_person_idx_per_face[i])]
                        assigned_to_existing[best_person_id].append(row["id"])
                        matched_total += 1
                    else:
                        unmatched_rows.append(row)
                processed += len(chunk)
                del face_feat_matrix  # 及时释放块内存

            self.logger.info(
                f"匹配完成：{matched_total} 张人脸匹配到已有人物，"
                f"{len(unmatched_rows)} 张人脸需要新建归类"
            )
        elif not incremental and row_iter is None:
            # 全量模式：需要从 iter_all_faces 加载
            for chunk in self.db.iter_all_faces_with_features(FACE_CHUNK_SIZE):
                unmatched_rows.extend(chunk)
        else:
            # incremental 但 person_features 为空：无已有人物，全部入 unmatched
            if row_iter is not None:
                for chunk in row_iter:
                    unmatched_rows.extend(chunk)

        if not unmatched_rows and not assigned_to_existing:
            return {}

        # 对未匹配的人脸进行聚类
        face_ids: list[int] = []
        feat_list: list[np.ndarray] = []
        for row in unmatched_rows:
            face_ids.append(row["id"])
            feat = DatabaseManager.feature_from_blob(row["feature"])
            feat_list.append(feat.flatten())

        n = len(face_ids)
        
        if n == 0:
            # 所有人脸都已匹配到已有人物
            _report(1, 1, "正在更新数据库...")
            self.db.begin()
            try:
                all_updates: list[tuple[int, int]] = []
                result: dict[int, list[int]] = {}
                for person_id, fids in assigned_to_existing.items():
                    for fid in fids:
                        all_updates.append((person_id, fid))
                    result[person_id] = fids

                    # 用已有代表特征 + 新人脸特征计算平均值
                    person_feats = []
                    existing_feat = self.db.get_person_feature(person_id)
                    if existing_feat is not None:
                        person_feats.append(existing_feat.flatten())
                    for fid in fids:
                        face_row = self.db.get_face(fid)
                        if face_row and face_row["feature"]:
                            feat = DatabaseManager.feature_from_blob(face_row["feature"])
                            person_feats.append(feat.flatten())
                    if person_feats:
                        avg_feat = np.mean(person_feats, axis=0)
                        avg_feat = avg_feat / max(np.linalg.norm(avg_feat), 1e-10)
                        self.db.update_person_feature(person_id, avg_feat)

                    existing_count = self.db.get_person_face_count(person_id)
                    self.db.update_person_face_count(person_id, existing_count + len(fids))

                self.db.batch_update_face_persons(all_updates)
                self.db.commit()
                self.logger.info(f"所有人脸已分配到已有人物，更新了 {len(result)} 个人物")
            except Exception as e:
                self.logger.error(f"更新数据库失败: {e}", exc_info=True)
                self.db.rollback()
                raise
            return result

        # 构建特征矩阵 (n, dim) 并 L2 归一化
        feat_matrix = np.vstack(feat_list).astype(np.float32)  # (n, dim)
        norms = np.linalg.norm(feat_matrix, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-10)  # 避免除零
        feat_matrix /= norms

        _report(0, n, f"加载 {n} 张待聚类人脸，开始向量化比对...")

        # Union-Find 聚类（基于索引 0..n-1）
        uf = UnionFind(n)

        # 分块计算相似度矩阵，避免一次性分配 n*n 矩阵导致内存不足
        block = self.BLOCK_SIZE
        total_blocks = (n + block - 1) // block
        processed_blocks = 0

        for i_start in range(0, n, block):
            i_end = min(i_start + block, n)
            # 只计算上三角部分：j >= i_start
            # 对于当前块的行 [i_start, i_end)，与所有列 [i_start, n) 比较
            chunk_i = feat_matrix[i_start:i_end]          # (block_i, dim)
            chunk_j = feat_matrix[i_start:]               # (n - i_start, dim)
            sim_block = chunk_i @ chunk_j.T               # (block_i, n - i_start)

            # 提取超过阈值的配对
            rows_idx, cols_idx = np.where(sim_block >= cosine_threshold)
            for r, c in zip(rows_idx, cols_idx):
                abs_i = i_start + r
                abs_j = i_start + c
                if abs_i < abs_j:  # 只取上三角
                    uf.union(abs_i, abs_j)

            processed_blocks += 1
            _report(processed_blocks, total_blocks,
                    f"比对进度: 第 {processed_blocks}/{total_blocks} 块 "
                    f"(行 {i_start}-{i_end-1}/{n-1})")

        # 收集分组：face_id 列表 + 特征索引列表（O(n) 一次遍历）
        groups: dict[int, list[int]] = defaultdict(list)          # root → [face_id]
        group_indices: dict[int, list[int]] = defaultdict(list)   # root → [feat_matrix 行号]
        for idx in range(n):
            root = uf.find(idx)
            groups[root].append(face_ids[idx])
            group_indices[root].append(idx)

        _report(total_blocks, total_blocks,
                f"比对完成，正在写入 {len(groups)} 个人物分组...")

        self.logger.info(f"人脸相似度比对完成，发现 {len(groups)} 个聚类组")

        # ★ 预计算每个组的代表性特征（纯 numpy，不涉及 DB）
        group_avg_feats: dict[int, np.ndarray] = {}
        for root_idx, indices in group_indices.items():
            avg_feat = np.mean(feat_matrix[indices], axis=0)
            norm = np.linalg.norm(avg_feat)
            if norm > 1e-10:
                avg_feat /= norm
            group_avg_feats[root_idx] = avg_feat

        # 写入数据库（单事务，最少 SQL 次数）
        result: dict[int, list[int]] = {}
        self.db.begin()
        try:
            all_updates: list[tuple[int, int]] = []

            # 处理新增的人物组（每组只需 1 次 INSERT + 2 次 UPDATE）
            for root_idx, group_face_ids in groups.items():
                person_id = self.db.add_person()
                self.db.update_person_face_count(person_id, len(group_face_ids))
                self.db.update_person_feature(person_id, group_avg_feats[root_idx])

                for fid in group_face_ids:
                    all_updates.append((person_id, fid))
                result[person_id] = group_face_ids

            # 处理匹配到已有人物的人脸（增量模式时才有）
            for person_id, fids in assigned_to_existing.items():
                for fid in fids:
                    all_updates.append((person_id, fid))

                # 用已有特征 + 新人脸特征计算平均值
                person_feats = []
                existing_feat = self.db.get_person_feature(person_id)
                if existing_feat is not None:
                    person_feats.append(existing_feat.flatten())
                for fid in fids:
                    face_row = self.db.get_face(fid)
                    if face_row and face_row["feature"]:
                        feat = DatabaseManager.feature_from_blob(face_row["feature"])
                        person_feats.append(feat.flatten())

                if person_feats:
                    avg_feat = np.mean(person_feats, axis=0)
                    avg_feat = avg_feat / max(np.linalg.norm(avg_feat), 1e-10)
                    self.db.update_person_feature(person_id, avg_feat)

                existing_count = self.db.get_person_face_count(person_id)
                self.db.update_person_face_count(person_id, existing_count + len(fids))

                if person_id not in result:
                    result[person_id] = []
                result[person_id].extend(fids)

            # 一次性批量更新所有 face → person 映射
            self.db.batch_update_face_persons(all_updates)
            self.db.commit()
            self.logger.info(f"聚类结果已写入数据库，新建 {len(groups)} 个人物，"
                           f"更新 {len(assigned_to_existing)} 个已有人物")
        except Exception as e:
            self.logger.error(f"聚类写入数据库失败: {e}", exc_info=True)
            self.db.rollback()
            raise

        return result
