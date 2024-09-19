from typing import List, Optional, Dict
from base_metric import Metric
from bert_score import BERTScorer

class BERTScore(Metric):
    def __init__(self, model_type: str, device: int, rescale_with_baseline: bool = True, language: str = "en"):
        super().__init__()
        self.scorer = BERTScorer(lang=language, model_type=model_type, device=f"cuda:{device}", rescale_with_baseline=rescale_with_baseline)

    def evaluate_batch(self, summaries: List[str], batch_size: int, articles: Optional[List[str]], references: Optional[List[str]]) -> List[Dict]:
        P, R, F1 = self.scorer.score(summaries, references, batch_size=batch_size)
        return [{
            f"bertscore_f1": F1[i].item(),
            f"bertscore_p": P[i].item(),
            f"bertscore_r": R[i].item(),
        } for i in range(len(summaries))]