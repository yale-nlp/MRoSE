from typing import List, Optional, Dict
from base_metric import Metric
from blanc import BlancHelp, BlancTune

class Blanc(Metric):
    def __init__(self, device: int):
        super().__init__()
        self.blanc_help = BlancHelp(device=f"cuda:{device}", inference_batch_size=80)
        self.blanc_tune = BlancTune(device=f"cuda:{device}", inference_batch_size=15, finetune_mask_evenly=False, finetune_batch_size=15)

    def evaluate_example(self, summary: str, article: str, reference: Optional[str]) -> Dict:
        return {
            f"blanc_help": self.blanc_help.eval_once(article, summary),
            f"blanc_tune": self.blanc_tune.eval_once(article, summary)
        }
    
    def evaluate_batch(self, summaries: List[str], batch_size: int, articles: List[str], references: Optional[List[str]]) -> List[Dict]:
        blanc_help_scores = self.blanc_help.eval_pairs(articles, summaries)
        blanc_tune_scores = self.blanc_tune.eval_pairs(articles, summaries)
        return [{
            f"blanc_help": blanc_help_scores[i],
            f"blanc_tune": blanc_tune_scores[i],
        } for i in range(len(summaries))]