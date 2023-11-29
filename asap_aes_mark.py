import fire
import json
import csv
import re
from typing import Optional

import asap_aes.q1 as q1
import asap_aes.q3 as q3
import asap_aes.q4 as q4
import asap_aes.q5 as q5
import asap_aes.q6 as q6
import asap_aes.q7 as q7

from marker import Marker

def main(
    llm: str,
    ckpt_dir: Optional[str] = None,
    tokenizer_path: Optional[str] = None,
    temperature: float = 1.0,
    top_p: float = 0.9,
    max_seq_len: int = 8192,
    max_batch_size: int = 1,
    max_gen_len: Optional[int] = None,
    version: int = 1,
):
    marker_obj = Marker(
        llm=llm,
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        temperature=temperature,
        top_p=top_p,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
        max_gen_len=max_gen_len,
    )

    q_dict = {
        1: [q1.question, q1.rubric, q1.max_marks],
        3: [q3.question, q3.rubric, q3.max_marks],
        4: [q4.question, q4.rubric, q4.max_marks],
        5: [q5.question, q5.rubric, q5.max_marks],
        6: [q6.question, q6.rubric, q6.max_marks],
        7: [q7.question, q7.rubric, q7.max_marks],
    }

    res_list = []

    with open("asap_aes/sample.tsv") as fd:
        rd = csv.reader(fd, delimiter="\t", quotechar='"')
        for idx, row in enumerate(rd):
            essay_id, essay_set, rater1_domain1, rater2_domain1, domain1_score, essay = row

            essay_id  = int(essay_id)
            essay_set = int(essay_set)
            rater1_domain1 = float(rater1_domain1)
            rater2_domain1 = float(rater2_domain1)
            domain1_score = float(domain1_score)
            essay = str(essay)

            quesion, rubric, max_marks = q_dict[essay_set]

            system_message = "You will receive a exam question along with the rubric. Additionally, you will be given a student's answers to the question. Your task is to assess and grade the student's answer by using the rubric."
            prompt = f'''Question:\n\n{quesion}\n\nStudent Answers:\n\n{essay}\n\nRubric:\n\n{rubric}\n\nPlease proceed with a detailed, step-by-step review of the student's response. Offer an in-depth rationale for the grade you assign, referencing the provided rubric. End your evaluation by denoting the given score in the format: SCORE: awarded_score/{max_marks}. Remember, the maximum score for this question is {max_marks}.'''
            award_mark = None
            
            while award_mark is None:
                try:
                    res = marker_obj.get_mark(system_message, prompt)
                    try:
                        award_mark = float(re.findall(f'([0-9\.]*)\/{max_marks}', res, re.IGNORECASE)[-1])
                        if award_mark > max_marks or award_mark < 0:
                            award_mark = None
                    except:
                        pass

                    if award_mark is None:
                        try:
                            award_mark = float(re.findall(f'([0-9\.]*) out of {max_marks}', res, re.IGNORECASE)[-1])
                            if award_mark > max_marks or award_mark < 0:
                                award_mark = None
                        except:
                            pass
                    
                except:
                    marker_obj.failed_sleep()


            res_list.append({
                'essay_id': essay_id,
                'essay_set': essay_set,
                'rater1_domain1': rater1_domain1,
                'rater2_domain1': rater2_domain1,
                'domain1_score': domain1_score,
                'essay': essay,
                'marker': res,
                'award_mark': award_mark
            })

            print(res_list[-1])

            file = f'asap_aes/results/{llm}_v{version}.json'
            with open(file, 'w') as f:
                json.dump(res_list, f, indent=4)

if __name__ == "__main__":
    fire.Fire(main)