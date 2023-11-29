import os
import json
import re
import time
import ShortAnswerGrading.q1 as q1
import ShortAnswerGrading.q2 as q2
import ShortAnswerGrading.q3 as q3
import ShortAnswerGrading.q4 as q4
import ShortAnswerGrading.q5 as q5

from typing import List, Optional
import fire
from llama import Llama, Dialog


def main(
    ckpt_dir: str,
    tokenizer_path: str,
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_seq_len: int = 512,
    max_batch_size: int = 8,
    max_gen_len: Optional[int] = None,
    version: int = 1,
):
    res_list = []

    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )

    for q_id, q in  enumerate([q1, q2, q3, q4, q5]):
        question = q.question
        ave = q.ave
        me = q.me
        other = q.other
        gt = q.gt
        answer = q.answer

        for a_id, (a, m, o, ans) in enumerate(zip(ave, me, other, answer)):
            if a_id <= 9:
                continue

            messages = [
                {"role": "system", "content": "You will receive a exam question along with the standard solutions. Additionally, you will be given a student's answers to these questions. Your task is to assess and grade the student's answers based on the standard solutions and marking scheme."},
            ]
            prompt = {"role" : "user", "content": f'''Questions:\n\n"{question}"\n\nStudent Answers:\n\n"{ans}"\n\nStandard Solutions:\n\n"{gt}"\n\nPlease note that the total marks allocated for this question is 5. Evaluate the student's answer step-by-step, comparing it to the standard solution. Offer a thorough justification for the grade you assign. Conclude your assessment by indicating the awarded score in the following format: SCORE: awarded_score/5.'''}
            messages.append(prompt)
            award_mark = None

            while award_mark is None:
                try:
                    res = generator.chat_completion(
                        [messages],  # type: ignore
                        max_gen_len=max_gen_len,
                        temperature=temperature,
                        top_p=top_p,
                    )[0]
                    print(res['generation']['content'])
                    # award_mark =  float(re.search('MARK:\s([0-9\.]*).*\/5', res, re.IGNORECASE).group(1))
                    award_mark = float(re.findall('\s([0-9\.]*)\/5', res['generation']['content'], re.IGNORECASE)[-1])
                    if award_mark > 5 or award_mark < 0:
                        award_mark = None
                    # award_mark = float(Decimal(award_mark).quantize(Decimal("1"), rounding = "ROUND_HALF_UP"))
                    # print('award_mark', award_mark)
                except:
                    print('api failed, sleep 10 seconds')
                    time.sleep(10)

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
                'marker': res['generation']['content']
            })

            print("====================PROMPT========================")
            print(prompt['content'])
            print("====================MARK========================")
            print(res)
            print('ave', a)
            print('me', m)
            print('other', o)
            print('award_mark', award_mark)
            print('\n\n\n\n\n\n')

            file = f'ShortAnswerGrading/{ckpt_dir}_v{version}.json'
            with open(file, 'w') as f:
                json.dump(res_list, f, indent=4)


if __name__ == "__main__":
    fire.Fire(main)
