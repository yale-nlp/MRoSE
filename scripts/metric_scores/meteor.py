from base_metric import Metric
from nltk.translate import meteor
from nltk import word_tokenize
from typing import Optional

class Meteor(Metric):
    def __init__(self):
        super().__init__()
        
    def evaluate_example(self, summary: str, article: Optional[str], reference: str):
        summ_tokens = word_tokenize(summary)
        ref_tokens = word_tokenize(reference)
        score = meteor([ref_tokens], summ_tokens)
        # print(score)
        return {"meteor": score}
