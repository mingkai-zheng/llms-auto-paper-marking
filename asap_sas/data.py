import csv

dict = {}

with open("asap_sas/train.tsv") as fd:
    rd = csv.reader(fd, delimiter="\t", quotechar='"')
    for idx, row in enumerate(rd):

        if idx == 0:
            continue
        
        essay_id, essay_set, score1, score2, essay_text = row

        if int(essay_set) not in [1,2,5,6]:
            continue
        
        # if int(score2) < 2:
        #     continue

        if essay_set in dict:
            dict[essay_set].append((essay_id, essay_set, score1, score2, essay_text))
        else:
            dict[essay_set] = [(essay_id, essay_set, score1, score2, essay_text)]

    for k,v in dict.items():
        dict[k] = sorted(v, key=lambda x: float(x[2]))
    
    # for k,v in dict.items():
    #     print(len(v))

    sampled = []
    for k,v in dict.items():
        # print(v)
        divid = len(v) // 20
        this_sampled =  v[::divid]
        sampled.extend(this_sampled[:20])

        # for s in sampled:
        #     print(s)
    
        # for s in v:
        #     print(s[4])
        # break

    with open('asap_sas/sample.tsv', 'w', newline='') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter="\t", quotechar='"')
        for s in sampled:
            spamwriter.writerow(s)
