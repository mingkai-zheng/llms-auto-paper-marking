import json
import re
from typing import Optional
import fire

import ShortAnswerGrading.q1 as q1
import ShortAnswerGrading.q2 as q2
import ShortAnswerGrading.q3 as q3
import ShortAnswerGrading.q4 as q4
import ShortAnswerGrading.q5 as q5

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

    res_list = []
    for q_id, q in  enumerate([q1, q2, q3, q4, q5]):
        question = q.question
        ave = q.ave
        me = q.me
        other = q.other
        gt = q.gt
        answer = q.answer

        for a_id, (a, m, o, ans) in enumerate(zip(ave, me, other, answer)):
            # a_id : answer_id, we only use answer 10 ~ 30
            if a_id <= 9:
                continue
            # if q_id < 4 or a_id < 22:
            #     continue
            
            system_message = "You will receive a exam question along with the standard solutions. Additionally, you will be given a student's answers to these questions. Your task is to assess and grade the student's answers based on the standard solutions and marking scheme."
            prompt = f'''Questions:\n\n"{question}"\n\nStudent Answers:\n\n"{ans}"\n\nStandard Solutions:\n\n"{gt}"\n\nPlease note that the total marks allocated for this question is 5. Evaluate the student's answer step-by-step, comparing it to the standard solution. Offer a thorough justification for the grade you assign. Conclude your assessment by indicating the awarded score in the following format: SCORE: awarded_score/5.'''
            award_mark = None

            while award_mark is None:
                try:
                    res = marker_obj.get_mark(system_message, prompt)
                    print(res)
                    try:
                        award_mark = float(re.findall('\s([0-9\.]*)\/5', res, re.IGNORECASE)[-1])
                        if award_mark > 5 or award_mark < 0:
                            award_mark = None
                    except:
                        pass
                    
                    if award_mark is None:
                        try:
                            award_mark = float(re.findall('([0-9\.]*) out of 5', res, re.IGNORECASE)[-1])
                            if award_mark > 5 or award_mark < 0:
                                award_mark = None
                        except:
                            pass
                except:
                    marker_obj.failed_sleep()

            res_list.append({
                'q_id': q_id+1,
                'a_id': a_id+1,
                'question': question,
                'gt': gt,
                'answer': ans,
                'ave': a,
                'me': m,
                'other': o,
                'pred': float(award_mark),
                'marker': res
            })

            print(res_list[-1])

            file = f'ShortAnswerGrading/results/{llm}_v{version}.json'
            with open(file, 'w') as f:
                json.dump(res_list, f, indent=4)

if __name__ == "__main__":
    fire.Fire(main)