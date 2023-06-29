import openai
import torch





def ask_chatGPT(prompt, max_tokens=200, temperature=0, model="gpt-3.5-turbo"):

    completion = openai.ChatCompletion.create(
        model=model,
        messages=[
            {"role": "assistant", "content": prompt}
        ],
        temperature=temperature
    )
    message = completion.choices[0].message.content

    return message




if __name__ == '__main__':
    file_path = '../data/ms-cxr/ms-cxr_test.pth'
    annot = torch.load(file_path)
    print(annot)
    prompt = 'Small bilateral pleural effusions are presumed. 请帮我从上述文本中提取症状实体，以json格式返回'
    msg = ask_chatGPT(prompt)
    print(msg)