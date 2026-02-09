"""人脸检测与识别引擎，基于 OpenCV YuNet + SFace"""

import os
import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class FaceData:
    """单个人脸的检测数据"""

    bbox: tuple  # (x, y, w, h)
    landmarks: list  # 5 个关键点 [(x,y), ...]
    score: float
    feature: Optional[np.ndarray] = field(default=None, repr=False)

    # 用于将 FaceData 还原为 detector 格式的 numpy 数组
    def to_detect_array(self) -> np.ndarray:
        flat_landmarks = [coord for pt in self.landmarks for coord in pt]
        return np.array(
            [*self.bbox, *flat_landmarks, self.score], dtype=np.float32
        )


# 关键点颜色 (BGR)
def imread_unicode(filepath: str) -> np.ndarray | None:
    """读取 Unicode 路径的图像（解决 Windows 上 cv2.imread 不支持非 ASCII 路径的问题）"""
    try:
        buf = np.fromfile(filepath, dtype=np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        return img
    except Exception:
        return None


LANDMARK_COLORS = [
    (255, 0, 0),    # 右眼
    (0, 0, 255),    # 左眼
    (0, 255, 0),    # 鼻子
    (255, 0, 255),  # 右嘴角
    (0, 255, 255),  # 左嘴角
]


class FaceEngine:
    """人脸检测 + 识别引擎"""

    SUPPORTED_FORMATS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

    def __init__(
        self,
        detection_model: str,
        recognition_model: str,
        score_threshold: float = 0.6,
        nms_threshold: float = 0.3,
        top_k: int = 5000,
    ):
        if not os.path.exists(detection_model):
            raise FileNotFoundError(
                f"找不到检测模型: {detection_model}\n"
                "请从 https://github.com/opencv/opencv_zoo/tree/master/models/face_detection_yunet 下载"
            )
        if not os.path.exists(recognition_model):
            raise FileNotFoundError(
                f"找不到识别模型: {recognition_model}\n"
                "请从 https://github.com/opencv/opencv_zoo/tree/master/models/face_recognition_sface 下载"
            )

        self.detector = cv2.FaceDetectorYN.create(
            detection_model, "", (320, 320),
            score_threshold, nms_threshold, top_k,
        )
        self.recognizer = cv2.FaceRecognizerSF.create(recognition_model, "")

    def detect(self, image: np.ndarray) -> list[FaceData]:
        """检测图像中所有人脸"""
        h, w = image.shape[:2]
        self.detector.setInputSize((w, h))
        _, raw_faces = self.detector.detect(image)

        if raw_faces is None or len(raw_faces) == 0:
            return []

        results = []
        for face_row in raw_faces:
            bbox = tuple(map(int, face_row[:4]))
            landmarks = [
                (int(face_row[4 + j * 2]), int(face_row[5 + j * 2]))
                for j in range(5)
            ]
            score = float(face_row[14])
            results.append(FaceData(bbox=bbox, landmarks=landmarks, score=score))
        return results

    def extract_feature(self, image: np.ndarray, face: FaceData) -> np.ndarray:
        """提取单张人脸的特征向量"""
        aligned = self.recognizer.alignCrop(image, face.to_detect_array())
        feature = self.recognizer.feature(aligned)
        face.feature = feature
        return feature

    def extract_features(self, image: np.ndarray, faces: list[FaceData]) -> list[FaceData]:
        """批量提取人脸特征"""
        for face in faces:
            self.extract_feature(image, face)
        return faces

    def compare(self, feat1: np.ndarray, feat2: np.ndarray) -> float:
        """比较两个特征向量的余弦相似度"""
        return float(
            self.recognizer.match(feat1, feat2, cv2.FaceRecognizerSF_FR_COSINE)
        )

    def visualize(
        self,
        image: np.ndarray,
        faces: list[FaceData],
        person_ids: Optional[list] = None,
        thickness: int = 2,
    ) -> np.ndarray:
        """在图像上绘制人脸框、关键点、人物ID"""
        result = image.copy()
        for idx, face in enumerate(faces):
            x, y, w, h = face.bbox
            cv2.rectangle(result, (x, y), (x + w, y + h), (0, 255, 0), thickness)

            # 标签
            if person_ids and idx < len(person_ids) and person_ids[idx] is not None:
                label = f"P{person_ids[idx]} ({face.score:.2f})"
            else:
                label = f"#{idx} ({face.score:.2f})"
            cv2.putText(result, label, (x, y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

            # 关键点
            for j, (lx, ly) in enumerate(face.landmarks):
                cv2.circle(result, (lx, ly), 3, LANDMARK_COLORS[j], thickness)

        cv2.putText(result, f"Faces: {len(faces)}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        return result

    @staticmethod
    def crop_face(image: np.ndarray, bbox: tuple, padding: float = 0.2) -> np.ndarray:
        """裁剪人脸区域（带 padding）"""
        x, y, w, h = bbox
        img_h, img_w = image.shape[:2]
        pad_w = int(w * padding)
        pad_h = int(h * padding)
        x1 = max(0, x - pad_w)
        y1 = max(0, y - pad_h)
        x2 = min(img_w, x + w + pad_w)
        y2 = min(img_h, y + h + pad_h)
        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            return image[y:y + h, x:x + w]
        return crop
