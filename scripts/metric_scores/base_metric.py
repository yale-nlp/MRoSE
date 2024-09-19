from typing import List, Optional, Dict
from tqdm import tqdm
import json
from multiprocessing import Pool

class Metric:
    def evaluate_example(self, summary: str, article: Optional[str], reference: Optional[str]) -> Dict:
        raise NotImplementedError

    def evaluate_batch(
        self, 
        summaries: List[str], 
        batch_size: int,
        articles: Optional[List[str]], 
        references: Optional[List[str]]
        ) -> List[Dict]:
        raise NotImplementedError

    def _evaluate_example(self, input):
        return self.evaluate_example(*input)
        
    def evaluate(
        self, 
        summaries: List[str], 
        articles: Optional[List[str]], 
        references: Optional[List[str]], 
        output_path: Optional[str] = None,
        batch_size: Optional[int] = -1,
        multi_process: Optional[int] = -1
        ) -> List[Dict]:
        articles = articles if articles is not None else [None] * len(summaries)
        references = references if references is not None else [None] * len(summaries)
        assert len(summaries) == len(articles) == len(references)
        results = []
        if batch_size > 0:
            for i in tqdm(range(0, len(summaries), batch_size)):
                batch_summaries = summaries[i:i+batch_size]
                batch_articles = articles[i:i+batch_size]
                batch_references = references[i:i+batch_size]
                _results = self.evaluate_batch(batch_summaries, batch_size, batch_articles, batch_references)
                results.extend(_results)
        elif multi_process > 0:
            with Pool(multi_process) as pool:
                _results = pool.imap(self._evaluate_example, zip(summaries, articles, references), chunksize=64)
                for x in tqdm(_results, total=len(summaries)):
                    results.append(x)
        else:
            for (summary, article, reference) in tqdm(zip(summaries, articles, references), total=len(summaries)):
                results.append(self.evaluate_example([summary], [article], [reference]))
                return results
        if output_path is not None:
            with open(output_path, "w") as f:
                for r in results:
                    print(json.dumps(r), file=f)
        return results



if __name__ == "__main__":
    metric = Metric()