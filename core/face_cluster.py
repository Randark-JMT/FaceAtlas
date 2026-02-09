"""人脸聚类模块 - 基于 Union-Find 将相似人脸归为同一人"""

from collections import defaultdict

from core.database import DatabaseManager


class UnionFind:
    """并查集"""

    def __init__(self):
        self.parent: dict[int, int] = {}
        self.rank: dict[int, int] = {}

    def find(self, x: int) -> int:
        if x not in self.parent:
            self.parent[x] = x
            self.rank[x] = 0
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

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
    """人脸聚类器"""

    def __init__(self, db: DatabaseManager, recognizer):
        self.db = db
        self.recognizer = recognizer

    def cluster(self, cosine_threshold: float = 0.363) -> dict[int, list[int]]:
        """
        对数据库中所有有特征的人脸进行聚类。

        Returns:
            {person_id: [face_id, ...]} 聚类结果
        """
        # 清除旧的归类
        self.db.clear_all_persons()

        rows = self.db.get_all_faces_with_features()
        if not rows:
            return {}

        # 加载 face_id 和特征
        face_ids = []
        features = []
        for row in rows:
            face_ids.append(row["id"])
            feat = DatabaseManager.feature_from_blob(row["feature"])
            features.append(feat)

        # Union-Find 聚类
        uf = UnionFind()
        n = len(face_ids)
        for i in range(n):
            for j in range(i + 1, n):
                score = float(self.recognizer.match(
                    features[i], features[j],
                    0  # cv2.FaceRecognizerSF_FR_COSINE == 0
                ))
                if score >= cosine_threshold:
                    uf.union(face_ids[i], face_ids[j])

        # 收集分组
        groups: dict[int, list[int]] = defaultdict(list)
        for fid in face_ids:
            root = uf.find(fid)
            groups[root].append(fid)

        # 写入数据库
        result: dict[int, list[int]] = {}
        for group_face_ids in groups.values():
            person_id = self.db.add_person()
            self.db.update_person_face_count(person_id, len(group_face_ids))
            for fid in group_face_ids:
                self.db.update_face_person(fid, person_id)
            result[person_id] = group_face_ids

        return result
