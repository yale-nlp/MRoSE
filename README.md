
# MRoSE
This is the repository for our 2024 ACL Findings paper "[Rethinking Efficient Multilingual Text Summarization Meta-Evaluation](https://aclanthology.org/2024.findings-acl.930)".

## Dataset Overview

MRoSE is created by translating the derived form of CNN/DailyMail test set from the RoSE Dataset into 30 languages using M2M-100 and GPT-3.5 models.

### Languages
The dataset encompasses a diverse range of languages, as listed below:


| Language Code | Language Name |
|---------------|---------------|
| ar            | Arabic        |
| cs            | Czech         |
| de            | German        |
| es            | Spanish       |
| fi            | Finnish       |
| fr            | French        |
| he            | Hebrew        |
| hi            | Hindi         |
| id            | Indonesian    |
| ja            | Japanese      |
| ko            | Korean        |
| nl            | Dutch         |
| pt            | Portuguese    |
| ru            | Russian       |
| sv            | Swedish       |
| ta            | Tamil         |
| tr            | Turkish       |
| uk            | Ukrainian     |
| vi            | Vietnamese    |
| zh            | Chinese       |
| bg            | Bulgarian     |
| bn            | Bengali       |
| el            | Greek         |
| fa            | Persian       |
| ga            | Irish         |
| hu            | Hungarian     |
| it            | Italian       |
| lt            | Lithuanian    |
| pl            | Polish        |
| yi            | Yiddish       |

### Translation Models

We utilized the following models for translating our dataset:

| Model Name    | Description                                         |
|---------------|-----------------------------------------------------|
| GPT-3.5 Turbo | A large language model by OpenAI                |
| M2M-100       | a multilingual seq-to-seq model primarily intended for translation tasks|

Our transformed datasets can be found in `./rose_translation/` directory.

## Evaluation Metrics
Our analysis employs a range of metrics to assess text summarization performance comprehensively:

| Metric Type        | Metric Name  | Description and Reference                          |
|--------------------|--------------|----------------------------------------------------|
| Text Matching Metrics | ROUGE-1,2,L | /                                        |
|                    | BLEU         | /                             |
|                    | METEOR       | /                           |
| Neural Metrics     | BERTScore    | Uses bert-base-multilingual-cased version |
|                    | BARTScore    | Uses mbart-large-50 model from MBART |
|                    | MoverScore   | Uses bert-base-multilingual-cased version |

## Data Structure 

The `/metrics/` directory contains all computed evaluation metrics, which are further organized into two subdirectories: `/translated_metrics/` and `/back_translated_metrics/`. The structure for each file is displayed below:


```{json}
[
    {
        "example_id": str, 
        "count_id": 0, 
        "metric_scores": 
            {
                "bertscore_f1": {"brio-ext": float, "bart": float, ....}, 
                "bertscore_p": {"brio-ext": float, "bart": float, ....}, 
                ....
            }
    },
    {
        "example_id": str, 
        "count_id": 1, 
        "metric_scores": 
            {
                "bertscore_f1": {"brio-ext": float, "bart": float, ....}, 
                "bertscore_p": {"brio-ext": float, "bart": float, ....}, 
                ....
            }
    },
    ....
]
```

## Evaluation

### Dependenecies 
The following versions are recommended:

* Python 3.9
* Pytorch 2.0.1
* Numpy 1.24.3
* Scipy 1.10.1
* Pandas 1.5.3
* Scikit-learn 1.2.2

### Evaluating the results

1. In `./scripts/metric_scores/compute_metrics.py`, set back_src_path, back_output_path to compute back translated metric scores, and set src_path, output_path to compute translated metric scores.
```
python ./scripts/metric_scores/compute_metrics.py
```
2. Configure the `./scripts/meta_evaluation/meta_evaluation.py` script with your root directory paths for RESULTS, BASE_DIR, and HUMAN_SCORE. 
```
python `./scripts/meta_evaluation/meta_evaluation.py`
```
Results will be saved in your desired path.




