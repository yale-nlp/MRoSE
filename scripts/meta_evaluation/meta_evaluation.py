import json
import numpy as np
import os
import csv
from scipy.stats import pearsonr, spearmanr, kendalltau
from collections import OrderedDict
from tabulate import tabulate
import warnings
import jsonlines

warnings.filterwarnings("ignore")

system = ['bart', 'gold', 'pegasus', 'brio', 'gsum', 'simcls', 'cliff', 'ctrlsum', 'frost', 'glob', 'matchsum', 'brio-ext']

LANGUAGE_NAMES = ["ar", "cs", "de", "es", "fi", "fr", "he", "hi", "id", "ja", "ko", "nl", "pt", "ru", "sv", "ta", "tr", "uk", "vi", "zh","bg","bn","el","fa","ga","hu","it","lt","pl","yi"]
ROUGE_SCORE_LIST = ["rouge1-fmeasure", "rouge1-precision", "rouge1-recall", "rouge2-fmeasure", "rouge2-precision", "rouge2-recall", "rougeL-fmeasure", "rougeL-precision", "rougeL-recall"]
BERTSCORE_LIST = ["bertscore_f1", "bertscore_p", "bertscore_r"]
MBARTSCORE_LIST = ["bartscore_f1", "bartscore_p", "bartscore_r"]
BLANC_LIST = ["blanc_help", "blanc_tune"]
BLEU_LIST = ["bleu"]
METEOR_LIST = ["meteor"]
MOVERSCORE_LIST = ["moverscore"]

METRIC_MAP = {
    "rouge": ROUGE_SCORE_LIST,
    "bertscore": BERTSCORE_LIST,
    "mbartscore": MBARTSCORE_LIST,
    "moverscore": MOVERSCORE_LIST,
    "bleu": BLEU_LIST,
    "meteor": METEOR_LIST

}
METRIC_FILES = ["BERTScore.jsonl", "BLEU.jsonl", "MBARTScore.jsonl", "METEOR.jsonl", "MoverScore.jsonl", "ROUGE.jsonl"]
RESULTS = ""
BASE_DIR = ""
HUMAN_SCORE = ""

def read_jsonl(src_path):
    with jsonlines.open(src_path) as f:
        return [line for line in f.iter()]

def load_scores_from_file(file_path):
    with open(file_path, 'r') as f:
        scores = [json.loads(line) for line in f]
    return scores

def load_human_scores_from_file(file_path):
    data = read_jsonl(file_path)
    acu = [{f"example {i}": d["human_scores"]["acu"]} for i, d in enumerate(data)]
    normalized_acu = [{f"example {i}": d["human_scores"]["normalized_acu"]} for i, d in enumerate(data)]
    return acu, normalized_acu

def _cal_pearson(x, y):
    v, p = pearsonr(x, y)
    if np.isnan(v):
        return 0, 1
    return v, p

def _cal_spearman(x, y):
    v, p = spearmanr(x, y)
    if np.isnan(v):
        return 0, 1
    return v, p

def _cal_kendall(x, y):
    v, p = kendalltau(x, y)
    if np.isnan(v):
        return 0, 1
    return v, p


def correlation_summ(refs, cands, corr_func):
    """
    refs: [num_system, num_summ] array
    cands: [num_system, num_summ] array
    corr_func: correlation_function
    """
    corr = 0
    p_value = 0
    assert refs.shape == cands.shape
    for i in range(refs.shape[1]):
        if np.std(refs[:, i]) == 0 or np.std(cands[:, i]) == 0:
            print(f"Warning: constant values in example {i+1}")
        corr_value, p = corr_func(refs[:, i], cands[:, i])
        corr += corr_value
        p_value += p
    return corr / refs.shape[1], p_value / refs.shape[1]

def correlation_system(refs, cands, corr_func):
    """
    refs: [num_system, num_summ] array
    cands: [num_system, num_summ] array
    corr_func: correlation_function
    """
    assert refs.shape == cands.shape
    ref = refs.mean(axis=1)
    cand = cands.mean(axis=1)
    if np.std(ref) == 0 or np.std(cand) == 0:
        print("Warning: constant values when averaging over systems")
    corr_value, p = corr_func(ref, cand)
    return corr_value, p

def compute_correlation(human_scores, metric_scores, metric_key):

    systems = list(list(human_scores[0].values())[0].keys())
    num_systems = len(systems)
    num_examples = len(human_scores)

    human = np.zeros((num_systems, num_examples))
    metric_mat = np.zeros((num_systems, num_examples))  

    for i, system in enumerate(systems):
        for j, example in enumerate(human_scores):
            human[i, j] = list(example.values())[0][system]
            metric_mat[i, j] = metric_scores[j]['metric_scores'][metric_key][system]







    summ_results = OrderedDict()
    system_results = OrderedDict()

    summ_results["pearson"], summ_results["pearson_p_value"] = correlation_summ(human, metric_mat, _cal_pearson)
    summ_results["spearman"], summ_results["spearman_p_value"] = correlation_summ(human, metric_mat, _cal_spearman)
    summ_results["kendall"], summ_results["kendall_p_value"] = correlation_summ(human, metric_mat, _cal_kendall)

    system_results["pearson"], system_results["pearson_p_value"] = correlation_system(human, metric_mat, _cal_pearson)
    system_results["spearman"], system_results["spearman_p_value"] = correlation_system(human, metric_mat, _cal_spearman)
    system_results["kendall"], system_results["kendall_p_value"] = correlation_system(human, metric_mat, _cal_kendall)

    return system_results, summ_results



def meta_evaluation_correlation(csv_path, lang_base_dir, human_dir):
    # human_scores, _ = load_human_scores_from_file(human_dir)  
    _, human_scores= load_human_scores_from_file(human_dir)

    with open(csv_path, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow([
            "Language", 
            "Metric", 
            "System Result Pearson", 
            "System Result Pearson p-value",
            "System Result Spearman", 
            "System Result Spearman p-value",
            "System Result Kendall", 
            "System Result Kendall p-value",
            "Summary Result Pearson",
            "Summary Result Pearson p-value",
            "Summary Result Spearman",
            "Summary Result Spearman p-value",
            "Summary Result Kendall",
            "Summary Result Kendall p-value"
        ])

        for language in LANGUAGE_NAMES:
            print(language)
            language_path = os.path.join(lang_base_dir, language)
            
            if os.path.isdir(language_path):
                for metric_file in METRIC_FILES:
                    metric_filepath = os.path.join(language_path, metric_file)

                    if os.path.exists(metric_filepath):
                        metric_name = metric_file.split('.')[0].lower()
                        metric_scores = load_scores_from_file(metric_filepath)

                        for metric_key in METRIC_MAP.get(metric_name, []):  
                            system_results, summ_results = compute_correlation(human_scores, metric_scores, metric_key)
                        
                            system_pearson = system_results["pearson"]
                            system_pearson_p = system_results["pearson_p_value"]
                            system_spearman = system_results["spearman"]
                            system_spearman_p = system_results["spearman_p_value"]
                            system_kendall = system_results["kendall"]
                            system_kendall_p = system_results["kendall_p_value"]

                            summ_pearson = summ_results["pearson"]
                            summ_pearson_p = summ_results["pearson_p_value"]
                            summ_spearman = summ_results["spearman"]
                            summ_spearman_p = summ_results["spearman_p_value"]
                            summ_kendall = summ_results["kendall"]
                            summ_kendall_p = summ_results["kendall_p_value"]

                            # Write to CSV
                            csvwriter.writerow([
                                language, 
                                metric_key,
                                system_pearson,
                                system_pearson_p,
                                system_spearman,
                                system_spearman_p,
                                system_kendall,
                                system_kendall_p,
                                summ_pearson,
                                summ_pearson_p,
                                summ_spearman,
                                summ_spearman_p,
                                summ_kendall,
                                summ_kendall_p
                            ])

    print("Results written to", csv_path)

if __name__ == "__main__":
    meta_evaluation_correlation(RESULTS, BASE_DIR, HUMAN_SCORE)
