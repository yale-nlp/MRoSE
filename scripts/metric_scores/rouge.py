from base_metric import Metric
from rouge_score import rouge_scorer
from nltk import word_tokenize
from typing import Optional

class Rouge(Metric):
    def __init__(self):
        super().__init__()
        # change ROUGE tokenizer to accept Arabic
        self.modify_tokenizer()
        self.scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeLsum'], use_stemmer=True, split_summaries=True)

    def modify_tokenizer(self):
        # your file path here
        fin = open("", "rt")
        data = fin.read()
        #replace all occurrences of the required string
        data = data.replace('a-z0-9', 'a-z0-9\\u0600-\\u06ff\\u0750-\\u077f\\ufb50-\\ufbc1\\ufbd3-\\ufd3f\\ufd50-\\ufd8f\\ufd50-\\ufd8f\\ufe70-\\ufefc\\uFDF0-\\uFDFD.0-9')
        fin.close()
        fin = open("", "wt")
        fin.write(data)
        fin.close()
        
    def evaluate_example(self, summary: str, article: Optional[str], reference: str):
        summary = " ".join(word_tokenize(summary.strip())).lower()
        reference = " ".join(word_tokenize(reference.strip())).lower()
        score = self.scorer.score(reference, summary)
        return {
            "rouge1-fmeasure": score["rouge1"].fmeasure,
            "rouge1-precision": score["rouge1"].precision,
            "rouge1-recall": score["rouge1"].recall,
            "rouge2-fmeasure": score["rouge2"].fmeasure,
            "rouge2-precision": score["rouge2"].precision,
            "rouge2-recall": score["rouge2"].recall,
            "rougeL-fmeasure": score["rougeLsum"].fmeasure,
            "rougeL-precision": score["rougeLsum"].precision,
            "rougeL-recall": score["rougeLsum"].recall,
        }
