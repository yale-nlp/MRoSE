import numpy as np
import json
from multiprocessing import Pool
from itertools import combinations
import numpy as np
from correlation import _cal_kendall, _cal_spearman, _cal_pearson, correlation_system, correlation_summ
from tabulate import tabulate
from load_metrics import load_cnndm_val_metrics, load_cnndm_test_metrics, ALL_CNNDM_METRICS


def bootstraping(fn, refs, sys1, sys2, corr_func, sample_size=-1, repetitions=1e5, num_workers=32):
    # perform bootstraping test, multi process
    with Pool(processes=num_workers) as pool:
        results = [pool.apply_async(fn, args=(refs, sys1, sys2, corr_func, sample_size, repetitions / num_workers)) for _ in range(num_workers)]
        cnts = [res.get() for res in results]
    return sum(cnts) / repetitions


## bootstraping significance values, system level
def _bootstraping_system(refs, sys1, sys2, corr_func, sample_size=-1, repetitions=1e5):
    # perform bootstraping test, single process
    np.random.seed()
    if sample_size < 0:
        sample_size = refs.shape[1]
    cnt = 0
    for i in range(int(repetitions)):
        # number of tests
        idx = np.random.choice(refs.shape[1], sample_size)  # sample examples
        refs_sample = refs[:, idx]
        sys1_sample = sys1[:, idx]
        sys2_sample = sys2[:, idx]
        corr1 = correlation_system(refs_sample, sys1_sample, corr_func)
        corr2 = correlation_system(refs_sample, sys2_sample, corr_func)
        now_delta = corr1 - corr2 # calculate delta (difference)
        if now_delta < 0:
            cnt += 1
    return cnt


def significance_system(refs, sys1, sys2, corr_func, verbose=True):
    """
    refs: [num_system, num_summ] array
    sys1, sys2: [num_system, num_summ] array
    corr_func: correlation_function
    """
    corr_sys1 = correlation_system(refs, sys1, corr_func)
    corr_sys2 = correlation_system(refs, sys2, corr_func)   
    diff = corr_sys1 - corr_sys2
    if verbose:
        print("sys1 correlation:", corr_sys1)
        print("sys2 correlation:", corr_sys2)
        print("correlation difference:", diff)
    if diff < 0:
        sys1, sys2 = sys2, sys1
    p = bootstraping(_bootstraping_system, refs, sys1, sys2, corr_func, num_workers=32)
    if verbose:
        print("p-value:", p)
    return p, diff


## permutation test, system level
def _permutation_system(refs, sys1, sys2, corr_func, sample_size=-1, repetitions=1e5):
    # perform permutation test, single process
    np.random.seed()
    if sample_size < 0:
        sample_size = refs.shape[1]
    cnt = 0
    corr_sys1 = correlation_system(refs, sys1, corr_func)
    corr_sys2 = correlation_system(refs, sys2, corr_func)   
    delta = corr_sys1 - corr_sys2
    for i in range(int(repetitions)):
        idx = np.random.random(refs.shape[1]) < 0.5
        sys1_sample = np.copy(sys1)
        sys2_sample = np.copy(sys2)
        sys2_sample[:, idx] = sys1[:, idx]
        sys1_sample[:, idx] = sys2[:, idx]
        corr1 = correlation_system(refs, sys1_sample, corr_func)
        corr2 = correlation_system(refs, sys2_sample, corr_func)
        now_delta = corr1 - corr2 # calculate delta (difference)
        if now_delta > delta:
            cnt += 1
    return cnt

def permutation_system(refs, sys1, sys2, corr_func, verbose=True):
    """
    refs: [num_system, num_summ] array
    sys1, sys2: [num_system, num_summ] array
    corr_func: correlation_function
    """
    corr_sys1 = correlation_system(refs, sys1, corr_func)
    corr_sys2 = correlation_system(refs, sys2, corr_func)   
    diff = corr_sys1 - corr_sys2
    if verbose:
        print("sys1 correlation:", corr_sys1)
        print("sys2 correlation:", corr_sys2)
        print("correlation difference:", diff)
    if diff < 0:
        sys1, sys2 = sys2, sys1
    p = bootstraping(_permutation_system, refs, sys1, sys2, corr_func, num_workers=32)
    if verbose:
        print("p-value:", p)
    return p, diff


## bootstraping significance values, summary level
def _permutation_summ_quick(corr_sys1, corr_sys2, sample_size=-1, repetitions=1e5):
    # perform bootstraping test, single process
    np.random.seed()
    if sample_size < 0:
        sample_size = corr_sys1.shape[0]
    cnt = 0
    delta = (corr_sys1 - corr_sys2).mean()
    for i in range(int(repetitions)):

        idx = np.random.random(sample_size) < 0.5
        sys1_sample = np.copy(corr_sys1)
        sys2_sample = np.copy(corr_sys2)
        sys2_sample[idx] = corr_sys1[idx]
        sys1_sample[idx] = corr_sys2[idx]
        now_delta = (sys1_sample - sys2_sample).mean() # calculate delta (difference)
        if now_delta > delta:
            cnt += 1
    return cnt

def permutation_summ_quick(x, y, sample_size=-1, repetitions=1e5, num_workers=32):
    # perform bootstraping test, multi process
    with Pool(processes=num_workers) as pool:
        results = [pool.apply_async(_permutation_summ_quick, args=(x, y, sample_size, repetitions / num_workers)) for _ in range(num_workers)]
        cnts = [res.get() for res in results]
    return sum(cnts) / repetitions


def permutation_summ(refs, sys1, sys2, corr_func, verbose=True):
    """
    refs: [num_system, num_summ] array
    sys1, sys2: [num_system, num_summ] array
    corr_func: correlation_function
    """
    corr_sys1, values1 = correlation_summ_values(refs, sys1, corr_func)
    corr_sys2, values2 = correlation_summ_values(refs, sys2, corr_func)   
    diff = corr_sys1 - corr_sys2
    if verbose:
        print("sys1 correlation:", corr_sys1)
        print("sys2 correlation:", corr_sys2)
        print("correlation difference:", diff)
    if diff < 0:
        sys1, sys2 = sys2, sys1
        values1, values2 = values2, values1
        corr_sys1, corr_sys2 = corr_sys2, corr_sys1
    diffs = values1 - values2

    p = permutation_summ_quick(values1, values2, num_workers=32)
    if verbose:
        print("p-value:", p)
    return p, diff


## bootstraping significance values, summary level
def _bootstraping_summ_quick(data, sample_size=-1, repetitions=1e5):
    # perform bootstraping test, single process
    np.random.seed()
    if sample_size < 0:
        sample_size = data.shape[0]
    cnt = 0
    for i in range(int(repetitions)):
        # number of tests
        # if i % 1000 == 0:
        #     print(i)
        idx = np.random.choice(data.shape[0], sample_size)  # sample examples
        samples = data[idx]
        now_delta = samples.mean()  # calculate delta (difference)
        if now_delta < 0:
            cnt += 1
    return cnt

def bootstraping_summ_quick(diff, sample_size=-1, repetitions=1e5, num_workers=32):
    # perform bootstraping test, multi process
    with Pool(processes=num_workers) as pool:
        results = [pool.apply_async(_bootstraping_summ_quick, args=(diff, sample_size, repetitions / num_workers)) for _ in range(num_workers)]
        cnts = [res.get() for res in results]
    return sum(cnts) / repetitions


def correlation_summ_values(refs, cands, corr_func):
    """
    refs: [num_system, num_summ] array
    cands: [num_system, num_summ] array
    corr_func: correlation_function
    """
    corr = 0
    assert refs.shape == cands.shape
    results = []
    for i in range(refs.shape[1]):
        _corr = corr_func(refs[:, i], cands[:, i])
        corr += _corr
        results.append(_corr)
    return corr / refs.shape[1], np.array(results)

def significance_summ(refs, sys1, sys2, corr_func, verbose=True):
    """
    refs: [num_system, num_summ] array
    sys1, sys2: [num_system, num_summ] array
    corr_func: correlation_function
    """
    corr_sys1, values1 = correlation_summ_values(refs, sys1, corr_func)
    corr_sys2, values2 = correlation_summ_values(refs, sys2, corr_func)   
    diff = corr_sys1 - corr_sys2
    if verbose:
        print("sys1 correlation:", corr_sys1)
        print("sys2 correlation:", corr_sys2)
        print("correlation difference:", diff)
    if diff < 0:
        sys1, sys2 = sys2, sys1
        values1, values2 = values2, values1
    diffs = values1 - values2
    # print(diffs)
    p = bootstraping_summ_quick(diffs, num_workers=32)
    if verbose:
        print("p-value:", p)
    return p, diff


## bootstraping confidence intervals
def _confidence_system(refs, cands, corr_func, sample_size=-1, repetitions=1e5):
    # perform bootstraping test, single process
    np.random.seed()
    if sample_size < 0:
        sample_size = refs.shape[1]
    cnt = 0
    results = []
    for i in range(int(repetitions)):
        # number of tests
        # if i % 1000 == 0:
        #     print(i)
        idx = np.random.choice(refs.shape[1], sample_size)  # sample examples
        refs_sample = refs[:, idx]
        cands_sample = cands[:, idx]
        corr = correlation_system(refs_sample, cands_sample, corr_func)
        results.append(corr)
    return results


def _confidence_summ(refs, cands, corr_func, sample_size=-1, repetitions=1e5):
    # perform bootstraping test, single process
    np.random.seed()
    if sample_size < 0:
        sample_size = refs.shape[1]
    results = []
    _, values = correlation_summ_values(refs, cands, corr_func)
    for i in range(int(repetitions)):
        # number of tests
        # if i % 1000 == 0:
        #     print(i)
        idx = np.random.choice(refs.shape[1], sample_size)  # sample examples
        corr = values[idx].mean()
        results.append(corr)
    return results


def confidence_interval(fn, refs, cands, corr_func, sample_size=-1, repetitions=1e5, num_workers=32):
    # perform bootstraping test, multi process
    with Pool(processes=num_workers) as pool:
        results = [pool.apply_async(fn, args=(refs, cands, corr_func, sample_size, repetitions / num_workers)) for _ in range(num_workers)]
        results = [res.get() for res in results]
    _results = []
    for res in results:
        _results.extend(res)
    results = _results
    # compute confidence interval
    results = np.array(results)
    head = np.percentile(results, 2.5)
    tail = np.percentile(results, 97.5)
    # print("confidence interval:", five_percentile, ninety_five_percentile)
    return head, tail


def main_confidence_interval(fdir, dataset, metric_list):
    with open(fdir) as f:
        human = json.load(f)
    systems = list(human.keys())
    print("Systems:", systems)
    doc_ids = set(human[systems[0]].keys())
    doc_ids = list(doc_ids)
    doc_ids = [int(x) for x in doc_ids]
    doc_ids.sort()
    print(f"Number of documents: {len(doc_ids)}")
    for s in systems:
        human[s] = np.array([human[s][str(x)] for x in doc_ids])
    human = np.array([human[s] for s in systems])

    # load auto scores
    if dataset == "cnndm_val":
        metrics = load_cnndm_val_metrics(doc_ids)
    elif dataset == "cnndm_test":
        metrics = load_cnndm_test_metrics(doc_ids)
    else:
        raise NotImplementedError
    metrics = {m: np.array([[x[m] for x in metrics[s]] for s in systems]) for m in metric_list}

    results = [["metric", "corr_func", "system-level", "summary-level"]]
    data = {m: dict() for m in metric_list}
    # compute correlation
    for m in metric_list:
        print(m)
        # pearson
        low_sys, high_sys = confidence_interval(_confidence_system, human, metrics[m], _cal_pearson, num_workers=32)
        corr_sys = correlation_system(human, metrics[m], _cal_pearson)
        data[m]["pearson-sys"] = {
            "low": low_sys,
            "high": high_sys,
            "corr": corr_sys
        }
        low_summ, high_summ = confidence_interval(_confidence_summ, human, metrics[m], _cal_pearson, num_workers=32)
        corr_summ = correlation_summ(human, metrics[m], _cal_pearson)
        data[m]["pearson-summ"] = {
            "low": low_summ,
            "high": high_summ,
            "corr": corr_summ
        }
        results.append([m, "pearson", f"({low_sys:.4f} - {high_sys:.4f})", f"({low_summ:.4f} - {high_summ:.4f})"])
        # spearma
        low_sys, high_sys = confidence_interval(_confidence_system, human, metrics[m], _cal_spearman, num_workers=32)
        corr_sys = correlation_system(human, metrics[m], _cal_spearman)
        data[m]["spearman-sys"] = {
            "low": low_sys,
            "high": high_sys,
            "corr": corr_sys
        }
        low_summ, high_summ = confidence_interval(_confidence_summ, human, metrics[m], _cal_spearman, num_workers=32)
        corr_summ = correlation_summ(human, metrics[m], _cal_spearman)
        data[m]["spearman-summ"] = {
            "low": low_summ,
            "high": high_summ,
            "corr": corr_summ
        }
        results.append([m, "spearman", f"({low_sys:.4f} - {high_sys:.4f})", f"({low_summ:.4f} - {high_summ:.4f})"])
        # kendall
        low_sys, high_sys = confidence_interval(_confidence_system, human, metrics[m], _cal_kendall, num_workers=32)
        corr_sys = correlation_system(human, metrics[m], _cal_kendall)
        data[m]["kendall-sys"] = {
            "low": low_sys,
            "high": high_sys,
            "corr": corr_sys
        }
        low_summ, high_summ = confidence_interval(_confidence_summ, human, metrics[m], _cal_kendall, num_workers=32)
        corr_summ = correlation_summ(human, metrics[m], _cal_kendall)
        data[m]["kendall-summ"] = {
            "low": low_summ,
            "high": high_summ,
            "corr": corr_summ
        }
        results.append([m, "kendall", f"({low_sys:.4f} - {high_sys:.4f})", f"({low_summ:.4f} - {high_summ:.4f})"])
    # print table
    print(tabulate(results, headers="firstrow", tablefmt="github"))
    return data


def sim_confidence_interval(fdir, dataset, metric, trial_num=100):
    with open(fdir) as f:
        human = json.load(f)
    systems = list(human.keys())
    print("Systems:", systems)
    doc_ids = set(human[systems[0]].keys())
    doc_ids = list(doc_ids)
    doc_ids = [int(x) for x in doc_ids]
    doc_ids.sort()
    print(f"Number of documents: {len(doc_ids)}")
    for s in systems:
        human[s] = np.array([human[s][str(x)] for x in doc_ids])
    human = np.array([human[s] for s in systems])

    # load auto scores
    if dataset == "cnndm_val":
        metrics = load_cnndm_val_metrics(doc_ids)
    elif dataset == "cnndm_test":
        metrics = load_cnndm_test_metrics(doc_ids)
    else:
        raise NotImplementedError
    auto_scores = np.array([[x[metric] for x in metrics[s]] for s in systems])

    results = [["num", "system-level", "summary-level"]]
    data = dict()
    for num in [50, 100, 200, 300, 400, 500, 1000]:
        low_sys, high_sys, low_summ, high_summ = 0, 0, 0, 0
        mean_sys, mean_summ = 0, 0
        for i in range(trial_num):
            if i % 10 == 0:
                print(num, i)
            # kendall
            idx = np.random.choice(len(doc_ids), num, replace=True)
            _human = human[:, idx]
            _auto_scores = auto_scores[:, idx]
            _low_sys, _high_sys = confidence_interval(_confidence_system, _human, _auto_scores, _cal_kendall, num_workers=32)
            _mean_sys = correlation_system(_human, _auto_scores, _cal_kendall)
            low_sys += _low_sys
            high_sys += _high_sys
            mean_sys += _mean_sys
            # compute summary-level
            # _low_summ, _high_summ = confidence_interval(_confidence_summ, _human, _auto_scores, _cal_kendall, num_workers=32)
            # _mean_summ = correlation_summ(_human, _auto_scores, _cal_kendall)
            # low_summ += _low_summ
            # high_summ += _high_summ
            # mean_summ += _mean_summ
        low_sys /= trial_num
        high_sys /= trial_num
        low_summ /= trial_num
        high_summ /= trial_num
        mean_sys /= trial_num
        mean_summ /= trial_num
        results.append([f"{num}", f"({low_sys:.4f} - {high_sys:.4f})", f"({low_summ:.4f} - {high_summ:.4f})"])
        print(results[-1])
        data[num] = {
            "system-level": {
                "low": low_sys,
                "high": high_sys,
                "mean": mean_sys
            },
            "summary-level": {
                "low": low_summ,
                "high": high_summ,
                "mean": mean_summ
            }
        }
    # print table
    print(tabulate(results, headers="firstrow", tablefmt="github"))
    with open(f"./figs/{metric}_sim.json", "w") as f:
        json.dump(data, f, indent=4)


def main_significance(fdir, dataset, metric_list):
    with open(fdir) as f:
        human = json.load(f)
    systems = list(human.keys())
    print("Systems:", systems)
    doc_ids = set(human[systems[0]].keys())
    doc_ids = list(doc_ids)
    doc_ids = [int(x) for x in doc_ids]
    doc_ids.sort()
    print(f"Number of documents: {len(doc_ids)}")
    for s in systems:
        human[s] = np.array([human[s][str(x)] for x in doc_ids])
    human = np.array([human[s] for s in systems])
    metric_pairs = combinations(metric_list, 2)
    # load auto scores
    if dataset == "cnndm_val":
        metrics = load_cnndm_val_metrics(doc_ids)
    elif dataset == "cnndm_test":
        metrics = load_cnndm_test_metrics(doc_ids)
    else:
        raise NotImplementedError
    metrics = {m: np.array([[x[m] for x in metrics[s]] for s in systems]) for m in metric_list}

    results = [["metric1", "metric2", "sys-diff", "sys-p", "summ-diff", "summ-p"]]
    num_pairs = len(metric_list) * (len(metric_list) - 1) // 2
    cnt = 0
    with open("significance_new.jsonl", "w") as f:
        for m1, m2 in metric_pairs:
            cnt += 1
            print(f"{cnt}/{num_pairs}")
            print(m1, m2)
            # p, diff = significance_system(human, metrics[m1], metrics[m2], _cal_kendall)
            p, diff = permutation_system(human, metrics[m1], metrics[m2], _cal_kendall, verbose=True)
            p_summ, diff_summ = significance_summ(human, metrics[m1], metrics[m2], _cal_kendall, verbose=False)
            results.append([m1, m2, f"{diff:.4f}", f"{p:.4f}", f"{diff_summ:.4f}", f"{p_summ:.4f}"])
            print(results[-1])
            print(json.dumps({
                "metric1": m1,
                "metric2": m2,
                "sys-diff": diff,
                "sys-p": p,
                "summ-diff": diff_summ,
                "summ-p": p_summ
            }), file=f, flush=True)
    # print table
    print(tabulate(results, headers="firstrow", tablefmt="github"))


def sim_significance(fdir, dataset, metric_list, trial_num=10):
    with open(fdir) as f:
        human = json.load(f)
    systems = list(human.keys())
    print("Systems:", systems)
    doc_ids = set(human[systems[0]].keys())
    doc_ids = list(doc_ids)
    doc_ids = [int(x) for x in doc_ids]
    doc_ids.sort()
    print(f"Number of documents: {len(doc_ids)}")
    for s in systems:
        human[s] = np.array([human[s][str(x)] for x in doc_ids])
    human = np.array([human[s] for s in systems])
    # load auto scores
    if dataset == "cnndm_val":
        metrics = load_cnndm_val_metrics(doc_ids)
    elif dataset == "cnndm_test":
        metrics = load_cnndm_test_metrics(doc_ids)
    else:
        raise NotImplementedError

    metric_pairs = combinations(metric_list, 2)
    num_pairs = len(metric_list) * (len(metric_list) - 1) // 2
    with open("metric_comparision.jsonl", "w") as f:
        for (i, metric_pair) in enumerate(metric_pairs):
            print(metric_pair)
            print(f"{i+1}/{num_pairs}")
            metric_a = np.array([[x[metric_pair[0]] for x in metrics[s]] for s in systems])
            metric_b = np.array([[x[metric_pair[1]] for x in metrics[s]] for s in systems])
            results = [["num", "system-level"]]
            output = {
                "metric1": metric_pair[0],
                "metric2": metric_pair[1],
            }
            for num in [50, 100, 200, 300, 400, 500, 1000]:
                output[num] = []
                sig = 0
                for i in range(trial_num):
                    if i % 10 == 0:
                        print(num, i)
                    # kendall
                    idx = np.random.choice(len(doc_ids), num, replace=True)
                    _human = human[:, idx]
                    _metric_a = metric_a[:, idx]
                    _metric_b = metric_b[:, idx]
                    # p, diff = significance_system(_human, _metric_a, _metric_b, _cal_kendall, verbose=False)
                    p, diff = permutation_system(_human, _metric_a, _metric_b, _cal_kendall, verbose=False)
                    sig += p
                    output[num].append(p)
                sig /= trial_num
                results.append([f"{num}", f"{sig:.4f}"])
                print(results[-1])
            print(json.dumps(output), file=f, flush=True)
            # print table
        print(tabulate(results, headers="firstrow", tablefmt="github"))
    

if __name__ == "__main__":
    ## your file path here
    # with open("") as f:
    #     human = json.load(f)
    # sim_confidence_interval("")
    # sim_confidence_interval("")

    # metrics = [
    #     # 'rouge1', 
    #     'rouge1r', 
    #     # 'rouge2',
    #     'rouge2r', 
    #     # 'rougeL',  
    #     'rougeLr', 
    #     'bert_score_deberta_r', 
    #     # 'bert_score_deberta_f1', 
    #     'bert_score_roberta_r', 
    #     # 'bert_score_roberta_f1', 
    #     'simcse_r', 
    #     # 'simcse_f1',
    #     "bart_score_cnndm_r",
    #     "bart_score_parabank_r",
    # ]
    # metrics = [
    #     'rouge1',
    #     # 'rouge1p', 
    #     # 'rouge1r', 
    #     'rouge2', 
    #     # 'rouge2p', 
    #     # 'rouge2r', 
    #     'rougeL', 
    #     # 'rougeLp', 
    #     # 'rougeLr', 
    #     # 'bartscore_recall_cnndm',
    #     'bartscore_precision_cnndm',
    #     # 'bartscore_precision_parabank', 
    #     # 'bartscore_recall_parabank', 
    #     'bertscore_f1_roberta', 
    #     # 'bertscore_p_roberta', 
    #     # 'bertscore_r_roberta', 
    #     'simcse', 
    #     'ctc', 
    #     # 'bertscore_f1_deberta', 
    #     # 'bertscore_p_deberta', 
    #     # 'bertscore_r_deberta', 
    #     # 'unieval_coherence', 
    #     # 'unieval_consistency', 
    #     # 'unieval_fluency', 
    #     'unieval_relevance', 
    #     # 'unieval_overall',
    # ]
    metrics = [
        # 'rouge1',
        # 'rouge1p', 
        'rouge1r', 
        # 'rouge2', 
        # 'rouge2p', 
        'rouge2r', 
        # 'rougeL', 
        # 'rougeLp', 
        'rougeLr', 
        # 'bartscore_recall_cnndm',
        # 'bartscore_precision_cnndm',
        # 'bartscore_precision_parabank', 
        'bartscore_recall_parabank', 
        # 'bertscore_f1_roberta', 
        # 'bertscore_p_roberta', 
        'bertscore_r_roberta', 
        'qaeval_f1',
        'summaqa_avg_prob',
        'p2c',
        'meteor',
        'chrf',
        # 'simcse', 
        # 'ctc', 
        # 'bertscore_f1_deberta', 
        # 'bertscore_p_deberta', 
        # 'bertscore_r_deberta', 
        # 'unieval_coherence', 
        # 'unieval_consistency', 
        # 'unieval_fluency', 
        # 'unieval_relevance', 
        # 'unieval_overall',
    ]
    # main_significance("", "cnndm_test", metrics)
    # main_significance("", "cnndm_test", metrics)

    # main_confidence_interval("", "cnndm_test", metrics)
    # main_confidence_interval("", "cnndm_test", metrics)
    # main_confidence_interval("", "cnndm_test", metrics)

    # main_confidence_interval("", "cnndm_test", metrics)

    
    # sim_confidence_interval("", "cnndm_test", "rouge1r")

    # sim_significance("", "cnndm_test", metrics)

    # sim_significance("")
    # main_confidence_interval("", "cnndm_test", ALL_CNNDM_METRICS)
    # main_significance("", "cnndm_test", ALL_CNNDM_METRICS)
    main_significance("", "cnndm_test", metrics)




    
    




    


