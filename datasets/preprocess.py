import torch
from utils import chatgpt
import time
import openai

with open('../utils/openai_api_key', 'r') as f:
    openai.api_key = f.read()
f.close()

file_path = '../data/ms-cxr/ms-cxr_test.pth'

if __name__ == '__main__':
    img_annot = torch.load(file_path)
    print(img_annot)
    question = '." please figure out the key noun word in the sentence, and reply just with the selected words.'

    result = []
    for i, annot in enumerate(img_annot):
        print(i)
        img_path = annot[0]
        bbox = annot[1]
        query = annot[2]
        # print(query)
        prompt = '"' + query + question
        # print(prompt)
        try:
            res = chatgpt.ask_chatGPT(prompt)
        except openai.error.ServiceUnavailableError:
            time.sleep(30)
        # print(res)
        tmp_tuple = (img_path, bbox, res)
        result.append(tmp_tuple)

    torch.save(res, '../data/ms-cxr_test_gpt.pth')
    print("done!")
