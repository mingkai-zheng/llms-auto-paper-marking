import os
import time
from typing import Optional
from functools import partial

marker_types = ['gpt-3.5-turbo', 'gpt-4', 'palm', 'llama-2-7b-chat',
                'llama-2-13b-chat', 'llama-2-70b-chat']

class Marker(object):
    def __init__(self,
                 llm: str,
                 ckpt_dir: str,
                 tokenizer_path: str,
                 temperature: float = 0.6,
                 top_p: float = 0.9,
                 max_seq_len: int = 8192,
                 max_batch_size: int = 1,
                 max_gen_len: Optional[int] = None
                 ):

        self.llm = llm.lower()

        if self.llm in ['gpt-3.5-turbo', 'gpt-4']:
            import openai
            openai.api_key = os.getenv("OPENAI_API_KEY")
            openai.organization = os.getenv("OPENAI_API_ORG")
            self.marker = partial(openai.ChatCompletion.create, model=self.llm)

        elif self.llm in ['llama-2-7b-chat', 'llama-2-13b-chat', 'llama-2-70b-chat']:
            
            if ckpt_dir is None:
                raise RuntimeError(f'Please provide the checkpoint path for {llm}')
                
            from llama import Llama
            self.marker = partial(Llama.build(
                ckpt_dir=ckpt_dir,
                tokenizer_path=tokenizer_path,
                max_seq_len=max_seq_len,
                max_batch_size=max_batch_size,
            ).chat_completion, max_gen_len=max_gen_len, temperature=temperature, top_p=top_p)
        
        elif self.llm == 'palm':
            import google.generativeai as palm
            palm.configure(api_key=os.environ['GOOGLE_MARKER_API_KEY'])
            self.marker = partial(palm.chat, temperature=1.0)
        else:
            raise RuntimeError(
                f'LLM {llm} is not available, please select from {marker_types}')

    def get_mark(self, system_message, prompt):
        if self.llm in ['gpt-3.5-turbo', 'gpt-4']:
            messages = [{
                "role": "system", "content": system_message,
                "role" : "user", "content": prompt
            }]
            return self.marker(messages=messages)['choices'][0]['message']['content']
        elif self.llm in ['llama-2-7b-chat', 'llama-2-13b-chat', 'llama-2-70b-chat']:
            messages = {
                "role": "system", "content": system_message,
                "role" : "user", "content": prompt
            }
            return self.marker([messages])[0]['generation']['content']
        else:
            return self.marker(messages=system_message+'\n\n'+prompt).last

    def failed_sleep(self):
        if self.llm in ['gpt-3.5-turbo', 'gpt-4', 'palm']:
            print('API failed or received an invalid response from LLM. Regenerate the response after a 10-second delay.')
            time.sleep(10)
            return
        print('API failed or received an invalid response from LLM. Regenerate the response.')
        