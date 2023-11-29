import csv
import pandas as pd

dict = {}

with open("asap_aes/training_set_rel3.tsv", encoding='latin1') as fd:
    rd = csv.reader(fd, delimiter="\t", quotechar='"')

    for idx, row in enumerate(rd):
        if idx == 0:
            continue

        essay_id  = int(row[0])
        essay_set = int(row[1])
        essay     = str(row[2])
        rater1_domain1 = float(row[3])
        rater2_domain1 = float(row[4])
        domain1_score = float(row[6])

        if int(essay_set) not in [1,3,4,5,6,7]:
            continue

        if essay_set in dict:
            dict[essay_set].append((essay_id, essay_set, rater1_domain1, rater2_domain1, domain1_score, essay))
        else:
            dict[essay_set] = [(essay_id, essay_set, rater1_domain1, rater2_domain1, domain1_score, essay)]

    for k,v in dict.items():
        dict[k] = sorted(v, key=lambda x: float(x[-2]))
    
    sampled = []
    for k,v in dict.items():
        divid = len(v) // 20
        this_sampled =  v[::divid]
        sampled.extend(this_sampled[:20])
    
    with open('asap_aes/sample.tsv', 'w', newline='') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter="\t", quotechar='"')
        for s in sampled:
            spamwriter.writerow(s)
