
from base_metric import Metric
from transformers import AutoTokenizer
from moverscore_v2 import get_idf_dict, word_mover_score
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

class MoverScore(Metric):
    def __init__(self, pretrained_model_name):
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-multilingual-cased')

    def evaluate_example(self, summary, article=None, reference=None):
        
        idf_dict_hyp = get_idf_dict(reference)  # idf dictionary for hypotheses
        idf_dict_ref = get_idf_dict(summary)    # idf dictionary for references
        scores = word_mover_score(reference, summary, idf_dict_ref, idf_dict_hyp, stop_words=[], n_gram=1, remove_subwords=True)
        return [{"moverscore": score} for score in scores]

    def evaluate_batch(self, summaries, batch_size=1, articles=None, references=None):
        if articles is None:
            articles = [None] * len(summaries)
        if references is None:
            references = [None] * len(summaries)
        
        results = []
        for i in range(0, len(summaries), batch_size):
            batch_summaries = summaries[i:i+batch_size]
            batch_articles = articles[i:i+batch_size]
            batch_references = references[i:i+batch_size]
            results.extend(self.evaluate_example(batch_summaries, batch_articles, batch_references))
        return results