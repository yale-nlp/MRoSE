from typing import List, Optional, Dict
from base_metric import Metric
from blanc import Estime

class EstimeMetric(Metric):
    def __init__(self, checkpoint: str, device: int):
        super().__init__()
        self.estimator = Estime(
            output=['alarms', 'soft', 'coherence'], 
            path_mdl=checkpoint, 
            path_mdl_raw=checkpoint, 
            device=f"cuda:{device}"
        )

    def evaluate_batch(self, summaries: List[str], batch_size: int, articles: List[str], references: Optional[List[str]]) -> List[Dict]:
        result = []
        for summary, article in zip(summaries, articles):
            print("article: ", article, len(article))
            print("summary: ", summary, len(summary))
            score = self.estimator.evaluate_claims(article, [summary])
            result.append({
                f"estime-alarm": score[0][0],
                f"estime-soft": score[0][1],
                f"estime-coherence": score[0][2]
            })
        return result