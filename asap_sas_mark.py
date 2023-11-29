import json
import csv
import re
from typing import Optional
import fire

import asap_sas.q1 as q1
import asap_sas.q2 as q2
import asap_sas.q5 as q5
import asap_sas.q6 as q6

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
        1: [q1.question, q1.gt, q1.rubric],
        2: [q2.question, q2.gt, q2.rubric],
        5: [q5.question, q5.gt, q5.rubric],
        6: [q6.question, q6.gt, q6.rubric],
    }

    res_list = []

    with open("asap_sas/sample.tsv") as fd:
        rd = csv.reader(fd, delimiter="\t", quotechar='"')
        for _, row in enumerate(rd):
            essay_id, essay_set, score1, score2, essay_text = row
            
            essay_id  = int(essay_id)
            essay_set = int(essay_set)
            score1    = float(score1)
            score2    = float(score2)

            quesion, gt, rubric = q_dict[essay_set]
            system_message = "You will receive a exam question along with their standard solutions and rubric. Additionally, you will be given a student's answers to the question. Your task is to assess and grade the student's answer by comparing them to the standard solutions."
            prompt = f'''Question:\n\n"{quesion}"\n\nStandard Solutions:\n\n"{gt}"\n\nStudent Answers:\n\n"{essay_text}"\n\nRubric:\n\n"{rubric}"\n\nNow, please conduct a thorough, step-by-step evaluation of the student's answer in comparison to the standard solutions. The student will only receive a score if their answer precisely conveys the same meaning as the standard solutions. Provide a comprehensive justification for the grade you are assigning, making reference to the provided rubric. Conclude your assessment by specifying the awarded score using the following format: SCORE: awarded_score/total_score."'''
            award_mark = None
            while award_mark is None:
                try:
                    res = marker_obj.get_mark(system_message, prompt)
                    try:
                        award_mark = float(re.findall('\s([0-9\.]*)\/3', res, re.IGNORECASE)[-1])
                        if award_mark > 3 or award_mark < 0:
                            award_mark = None
                    except:
                        pass

                    if award_mark is None:
                        try:
                            award_mark = float(re.findall('([0-9\.]*) out of 3', res, re.IGNORECASE)[-1])
                            if award_mark > 3 or award_mark < 0:
                                award_mark = None
                        except:
                            pass
                except:
                    marker_obj.failed_sleep()
                    
            res_list.append({
                'essay_id': essay_id,
                'essay_set': essay_set,
                'score1': score1,
                'score2': score2,
                'essay_text': essay_text,
                'marker': res,
                'award_mark': award_mark
            })

            print(res_list[-1])

            file = f'asap_sas/results/{llm}_v{version}.json'
            with open(file, 'w') as f:
                json.dump(res_list, f, indent=4)

if __name__ == "__main__":
    fire.Fire(main)