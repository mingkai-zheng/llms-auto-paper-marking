import json
import fire
from scipy import stats
import numpy as np
np.set_printoptions(precision=4)
import ml_metrics
from decimal import Decimal

def convert_list2numpy(*args):
    converted_args = []
    for x in args:
        if isinstance(x, list):
            converted_args.append(np.array(x))
        else:
            converted_args.append(x)
    return tuple(converted_args)

def get_mse(gt, pred):
    return np.mean((pred - gt) ** 2, axis=-1)

def get_rmse(gt, pred):
    return np.mean((pred - gt) ** 2, axis=-1) ** 0.5

def get_pearson(gt, pred):
    num_questions = gt.shape[0]
    if (gt.ndim > 1 and num_questions == 1) or gt.ndim == 1:
        return stats.pearsonr(gt, pred).statistic
    rtn = []
    for g, p in zip(gt, pred):
        rtn.append(stats.pearsonr(g, p).statistic)
    return np.array(rtn)
    
def get_spearman(gt, pred):
    num_questions = gt.shape[0]
    if (gt.ndim > 1 and num_questions == 1) or gt.ndim == 1:
        return stats.spearmanr(gt, pred).statistic
    rtn = []
    for g, p in zip(gt, pred):
        rtn.append(stats.spearmanr(g, p).statistic)
    return np.array(rtn)

def get_kendall(gt, pred):
    num_questions = gt.shape[0]
    if (gt.ndim > 1 and num_questions == 1) or gt.ndim == 1:
        return stats.kendalltau(gt, pred).statistic
    rtn = []
    for g, p in zip(gt, pred):
        rtn.append(stats.kendalltau(g, p).statistic)
    return np.array(rtn)

def get_quadratic_weighted_kappa(gt, pred):
    num_questions = gt.shape[0]
    if (gt.ndim > 1 and num_questions == 1) or gt.ndim == 1:
        return ml_metrics.quadratic_weighted_kappa(gt, pred)
    rtn = []
    for g, p in zip(gt, pred):
        rtn.append(ml_metrics.quadratic_weighted_kappa(g, p, min_rating=0, max_rating=5))
    return np.array(rtn)

def get_all_statistics(gt, pred):
    gt, pred = convert_list2numpy(gt, pred)
    mse      = get_mse(gt, pred)
    rmse = get_rmse(gt, pred)
    pearson  = get_pearson(gt, pred)
    spearman = get_spearman(gt, pred)
    kendall  = get_kendall(gt, pred)
    need_scale = False
    for x in gt:
        if int(x) != float(x):
            need_scale = True
    for x in pred:
        if int(x) != float(x):
            need_scale = True
    if need_scale:
        gt *= 2
        pred *= 2
        pred = np.array([float(Decimal(p).quantize(Decimal("1"), rounding = "ROUND_HALF_UP")) for p in pred ])
    qwk = get_quadratic_weighted_kappa(gt, pred)
    return mse, rmse, pearson, spearman, kendall, qwk


def get_all_unique_qid(llm):
    qid_list = []
    with open(f'ShortAnswerGrading/results/{llm}_v1.json', 'r') as f:
        for obj in json.load(f):
            qid_list.append(obj['q_id'])
    qid_list = set(qid_list)
    return qid_list


def get_statistics_for_one_questtion(llm, flag, qid):
    all_score = []
    all_pred = []

    for version in range(1,4):
        gt_score = []
        pred_score = []

        with open(f'ShortAnswerGrading/results/{llm}_v{version}.json', 'r') as f:
            for obj in json.load(f):
                if qid != obj['q_id']:
                    continue

                if flag == 'ave':
                    gt_score.append(float(obj["ave"]))
                elif flag == 'me':
                    gt_score.append(float(obj["me"]))
                else:
                    gt_score.append(float(obj["other"]))
                
                pred_score.append(float(obj["pred"]))

        mse, rmse, pearson, spearman, kendall, qwk = get_all_statistics(gt_score, pred_score)
        all_pred.append(pred_score)
        all_score.append([mse, rmse, pearson, spearman, kendall, qwk])

    mean_results =  np.array(all_score).mean(axis=0)
    ensemble_pred = np.array(all_pred).mean(axis=0)
    ensemble_results = list(get_all_statistics(gt_score, ensemble_pred))
    
    
    return mean_results, ensemble_results
    

def main(
    llm: str,
    flag: str = 'me',
):
    all_qid = get_all_unique_qid(llm)

    all_mean_results = []
    all_ensmble_results = []

    for qid in all_qid:
        mean_results, ensemble_results = get_statistics_for_one_questtion(llm, flag, qid)
        all_mean_results.append(mean_results)
        all_ensmble_results.append(ensemble_results)
    
    all_mean_results = np.array(all_mean_results).mean(axis=0)
    all_ensmble_results = np.array(all_ensmble_results).mean(axis=0)

    mse, rmse, pearson, spearman, kendall, qwk = all_mean_results.tolist()
    print(f'=================={llm} mean =================')
    print(f'mse:      {mse}')
    print(f'rmse:     {rmse}')
    print(f'pearson:  {pearson}')
    print(f'spearman: {spearman}')
    print(f'kendall:  {kendall}')
    print(f'qwk:      {qwk}')

    mse = float(Decimal(mse).quantize(Decimal("0.01"), rounding = "ROUND_HALF_UP"))
    rmse = float(Decimal(rmse).quantize(Decimal("0.01"), rounding = "ROUND_HALF_UP"))
    pearson = float(Decimal(pearson).quantize(Decimal("0.01"), rounding = "ROUND_HALF_UP"))
    spearman = float(Decimal(spearman).quantize(Decimal("0.01"), rounding = "ROUND_HALF_UP"))
    kendall = float(Decimal(kendall).quantize(Decimal("0.01"), rounding = "ROUND_HALF_UP"))
    print(f'{mse} & {rmse} & {pearson} & {spearman} & {kendall} &')

    mse, rmse, pearson, spearman, kendall, qwk = all_ensmble_results.tolist()
    print(f'=================={llm} Ensemble =================')
    print(f'mse:      {mse}')
    print(f'rmse:     {rmse}')
    print(f'pearson:  {pearson}')
    print(f'spearman: {spearman}')
    print(f'kendall:  {kendall}')
    print(f'qwk:      {qwk}')

    mse = float(Decimal(mse).quantize(Decimal("0.01"), rounding = "ROUND_HALF_UP"))
    rmse = float(Decimal(rmse).quantize(Decimal("0.01"), rounding = "ROUND_HALF_UP"))
    pearson = float(Decimal(pearson).quantize(Decimal("0.01"), rounding = "ROUND_HALF_UP"))
    spearman = float(Decimal(spearman).quantize(Decimal("0.01"), rounding = "ROUND_HALF_UP"))
    kendall = float(Decimal(kendall).quantize(Decimal("0.01"), rounding = "ROUND_HALF_UP"))
    print(f'{mse} & {rmse} & {pearson} & {spearman} & {kendall} &')


if __name__ == "__main__":
    fire.Fire(main)