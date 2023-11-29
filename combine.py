import json
from scipy import stats
import numpy as np
import fire
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
        rtn.append(ml_metrics.quadratic_weighted_kappa(g, p, min_rating=0, max_rating=3))
    return np.array(rtn)
    
def get_all_statistics(gt, pred):
    # print(gt, pred)
    gt, pred = convert_list2numpy(gt, pred)
    mse      = get_mse(gt, pred)
    rmse = get_rmse(gt, pred)
    pearson  = get_pearson(gt, pred)
    spearman = get_spearman(gt, pred)
    kendall  = get_kendall(gt, pred)
    qwk = get_quadratic_weighted_kappa(gt, pred)
    return mse, rmse, pearson, spearman, kendall, qwk






def main(
    llm: str,
):
    
    idx_list = [[1,2], [3, 4], [5,6]]

    all_pred = []
    all_score = []
    gt = []
    for trail in range(1, 4):

        idx = idx_list[trail-1]
        with open(f'asap_aes/results/{llm}_v{idx[0]}.json', 'r') as f:
            r1 = json.load(f)

        with open(f'asap_aes/results/{llm}_v{idx[1]}.json', 'r') as f:
            r2 = json.load(f)

        gt = []
        pred = []
        for obj1, obj2 in zip(r1, r2):
            essay_id        = int(obj1['essay_id'])
            essay_set       = int(obj1['essay_set'])
            rater1_domain1  = float(obj1['rater1_domain1'])
            rater2_domain1  = float(obj1['rater2_domain1'])
            domain1_score   = float(obj1['domain1_score'])
            essay           = obj1['essay']
            marker          = obj1['marker']
            award_mark1      = float(obj1['award_mark'])
            award_mark2      = float(obj2['award_mark'])


            award_mark = 0

            if essay_set == 1:
                award_mark = award_mark1 + award_mark2
            elif essay_set == 3:
                award_mark = (award_mark1 + award_mark2) / 2
            elif essay_set == 4:
                award_mark = (award_mark1 + award_mark2) / 2
            elif essay_set == 5:
                award_mark = max(award_mark1, award_mark2)
            elif essay_set == 6:
                award_mark = max(award_mark1, award_mark2)
            elif essay_set == 7:
                award_mark = award_mark1 + award_mark2

            gt.append(domain1_score)
            pred.append(award_mark)

        mse, rmse, pearson, spearman, kendall, qwk = get_all_statistics(gt, pred)
        print(f'================== Trail {trail}=================')
        print(f'mse:      {mse}'     )
        print(f'rmse:     {rmse}'    )
        print(f'pearson:  {pearson}' )
        print(f'spearman: {spearman}')
        print(f'kendall:  {kendall}' )
        print(f'qwk:      {qwk}'     )
        all_pred.append(pred)
        all_score.append([mse, rmse, pearson, spearman, kendall, qwk])

    # print(all_pred)

    # print(len(all_pred[0]))
    # print(len(all_pred[1]))
    # print(len(all_pred[2]))

    # all_pred = np.stack([np.array(ap) for ap in all_pred])

    # print(all_pred)
    

    # print('test===============================')
    # print(all_pred[0])
    # print(np.array(all_pred))

    mean_pred = np.array(all_pred).mean(axis=0)
    # print(mean_pred.shape)
    # print(gt)

    mse, rmse, pearson, spearman, kendall, qwk = get_all_statistics(gt, mean_pred)
    print(f'=================={llm} Ensemble=================')
    print(f'mse:      {mse}'     )
    print(f'rmse:     {rmse}'    )
    print(f'pearson:  {pearson}' )
    print(f'spearman: {spearman}')
    print(f'kendall:  {kendall}' )
    print(f'qwk:      {qwk}'     )

    mean =  np.array(all_score).mean(axis=0)
    std = np.array(all_score).std(axis=0)
    print(f'=================={llm} mean =================')
    print(f'mse:      {mean[0]}')
    print(f'rmse:     {mean[1]}')
    print(f'pearson:  {mean[2]}')
    print(f'spearman: {mean[3]}')
    print(f'kendall:  {mean[4]}')
    print(f'qwk:      {mean[5]}')

    print(f'=================={llm} std =================')
    print(f'mse:      {std[0]}')
    print(f'rmse:     {std[1]}')
    print(f'pearson:  {std[2]}')
    print(f'spearman: {std[3]}')
    print(f'kendall:  {std[4]}')
    print(f'qwk:      {std[5]}')
    

if __name__ == "__main__":
    fire.Fire(main)