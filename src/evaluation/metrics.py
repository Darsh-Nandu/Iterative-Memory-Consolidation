import numpy as np
from difflib import SequenceMatcher
from sentence_transformers import SentenceTransformer

class QuantitativeMetricsSuite:
    """
    Computes exact mathematical matrix distances, alignments, and traditional semantic indicators.
    """
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    @staticmethod
    def compute_exact_match(prediction: str, ground_truth: str) -> float:
        return 1.0 if prediction.strip().lower() == ground_truth.strip().lower() else 0.0

    @staticmethod
    def compute_token_f1(prediction: str, ground_truth: str) -> float:
        pred_words = prediction.strip().lower().split()
        gt_words = ground_truth.strip().lower().split()
        if not pred_words or not gt_words:
            return 0.0
        common = set(pred_words) & set(gt_words)
        if not common:
            return 0.0
        precision = len(common) / len(pred_words)
        recall = len(common) / len(gt_words)
        return 2 * (precision * recall) / (precision + recall + 1e-9)

    def compute_semantic_similarity(self, prediction: str, ground_truth: str) -> float:
        emb1 = self.model.encode(prediction)
        emb2 = self.model.encode(ground_truth)
        return float(np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2) + 1e-9))