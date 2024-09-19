import json
from rouge import Rouge
from bertscore import BERTScore
from mbartscore import MBARTScore
from bleu import BleuMetric
from meteor import Meteor
from moverscore import MoverScore
import os
import torch
import nltk

import logging
logger = logging.getLogger("spacy")
logger.setLevel(logging.ERROR)
print(torch.cuda.is_available())
LANGUAGE_NAMES = ["ar", "cs", "de", "es", "fi", "fr", "he", "hi", "id", "ja", "ko", "nl", "pt", "ru", "sv", "ta", "tr", "uk", "vi", "zh","bg","bn","el","fa","ga","hu","it","lt","pl","yi"]


SYSTEM_NAMES = ["bart", "brio", "cliff", "ctrlsum", "frost", "glob", "gold", "gsum", "pegasus", "simcls", "matchsum", "brio-ext"]
ROUGE_SCORE_LIST = ["rouge1-fmeasure", "rouge1-precision", "rouge1-recall", "rouge2-fmeasure", "rouge2-precision", "rouge2-recall", "rougeL-fmeasure", "rougeL-precision", "rougeL-recall"]
BERTSCORE_LIST = ["bertscore_f1", "bertscore_p", "bertscore_r"]
BARTSCORE_LIST = ["bartscore_f1", "bartscore_p", "bartscore_r"]
MOVERSCORE_LIST = ["moverscore"]
BLANC_LIST = ["blanc_help", "blanc_tune"]
ESTIME_LIST = ['estime-alarm', 'estime-soft', 'estime-coherence']
BLEU_SCORE_LIST = ["bleu"]
METEOR_LIST = ["meteor"]

def read_json(src_path):
    '''
    Read json files.
    '''
    with open(src_path) as file:
        json_content = file.read()
    return json.loads(json_content)

def compute_score(metric, system_names, score_list, src_path, output_path, batch_size=-1, multi_process=-1):
    '''
    Compute scores for a given metric.
    '''

    print(src_path)
    src = read_json(src_path)
    results = {}
    num = len(src)
    print("example nums: ", num)

    references = [x["reference"] for x in src]
    articles = [x["source"] for x in src]
    summ = [x["system_output"] for x in src]

    for i in range(len(system_names)):
        print(f"system: {system_names[i]}, {i + 1}/{len(system_names)}")
        system_name = system_names[i]
        summaries = [x[system_name] for x in summ]

        results[system_name] = metric.evaluate(summaries, articles, references,batch_size=batch_size, multi_process=multi_process)

        print(results)
        return
    
    with open(output_path, "w") as f:
        for i in range(num):
            data = dict()
            data["example_id"] = src[i]["example_id"]
            data["count_id"] = src[i]["count_id"]
            data["metric_scores"] = {metric: {system: {}} for metric in score_list for system in system_names}

            for sys_name in system_names:
                for score_name, score in results[sys_name][i].items():
                    data["metric_scores"][score_name][sys_name] = score
            print(json.dumps(data), file=f)


def compute_rouge(system_names, language_names, src_path, output_path):
    '''
    Compute ROUGE scores for all languages.
    '''
    for language in language_names:    
        print("computing ROUGE scores for " + language + "...")
        rouge = Rouge()
        _src_path = src_path + language + ".json"
        _output_path = output_path + language + "/ROUGE.jsonl"    
        compute_score(rouge, system_names, ROUGE_SCORE_LIST, _src_path, _output_path, multi_process=16)

def compute_bleu(system_names, language_names, src_path, output_path):
    '''
    Compute BLEU scores for all languages.
    '''
    for language in language_names:    
        print("computing BLEU scores for " + language + "...")
        bleu = BleuMetric()
        _src_path = src_path + language + ".json"
        _output_path = output_path + language + "/BLEU.jsonl"    
        compute_score(bleu, system_names, BLEU_SCORE_LIST, _src_path, _output_path, multi_process=16)


def compute_meteor(system_names, language_names, src_path, output_path):
    '''
    Compute METEOR scores for all languages.
    '''
    for language in language_names:    
        print("computing METEOR scores for " + language + "...")
        bleu = Meteor()
        _src_path = src_path + language + ".json"
        _output_path = output_path + language + "/METEOR.jsonl"    
        compute_score(bleu, system_names, METEOR_LIST, _src_path, _output_path, multi_process=16)


def compute_bertscore(system_names, language_names, src_path, output_path, device):
    '''
    Compute BERTScore scores for all languages.
    '''    
    for language in language_names:
        print("computing BERTScore for " + language + "...")
        bert_score = BERTScore("bert-base-multilingual-cased", device, False, language)
        _src_path = src_path + language + ".json"
        _output_path = output_path + language + "/BERTScore.jsonl"
        compute_score(bert_score, system_names, BERTSCORE_LIST, _src_path, _output_path, batch_size=64)

def compute_mbartscore(system_names, language_names, src_path, output_path, device):
    '''
    Compute MBARTScore scores for all languages.
    '''
    bart_score = MBARTScore(device=device, checkpoint="facebook/mbart-large-cc25")    
    for language in language_names:
        print("computing MBARTScore for " + language + "...")
        _src_path = src_path + language + ".json"
        _output_path = output_path + language + "/MBARTScore.jsonl"
        compute_score(bart_score, system_names, BARTSCORE_LIST, _src_path, _output_path, batch_size=16) 

def compute_moverscore(system_names, language_names, src_path, output_path, device):
    '''
    Compute MOVERScore scores for all languages.
    '''    
    mover_score = MoverScore("bert-base-multilingual-cased")    
    for language in language_names:
        print("computing MOVERScore for " + language + "...")
        _src_path = src_path + language + ".json"
        _output_path = output_path + language + "/MoverScore.jsonl"
        compute_score(mover_score, system_names, MOVERSCORE_LIST, _src_path, _output_path)

if __name__ == "__main__":
    # your file path here
    translated_src_path = ""
    translated_output_path = ""
    back_src_path = ""
    back_output_path = ""

    compute_rouge(SYSTEM_NAMES, LANGUAGE_NAMES, translated_src_path, translated_output_path)
    compute_bleu(SYSTEM_NAMES, LANGUAGE_NAMES, translated_src_path, translated_output_path)    
    compute_meteor(SYSTEM_NAMES, LANGUAGE_NAMES, translated_src_path, translated_output_path)
    compute_bertscore(SYSTEM_NAMES, LANGUAGE_NAMES, translated_src_path, translated_output_path, device=2)
    compute_mbartscore(SYSTEM_NAMES, LANGUAGE_NAMES, translated_src_path, translated_output_path, device=2)
    compute_moverscore(SYSTEM_NAMES, LANGUAGE_NAMES, translated_src_path, translated_output_path, device=2)

    compute_rouge(SYSTEM_NAMES, LANGUAGE_NAMES, back_src_path, back_output_path)
    compute_bleu(SYSTEM_NAMES, LANGUAGE_NAMES, back_src_path, back_output_path)    
    compute_meteor(SYSTEM_NAMES, LANGUAGE_NAMES, back_src_path, back_output_path)
    compute_bertscore(SYSTEM_NAMES, LANGUAGE_NAMES, back_src_path, back_output_path, device=2)
    compute_mbartscore(SYSTEM_NAMES, LANGUAGE_NAMES, back_src_path, back_output_path, device=2)
    compute_moverscore(SYSTEM_NAMES, LANGUAGE_NAMES, back_src_path, back_output_path, device=2)

