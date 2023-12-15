import os
import requests
from langchain.llms import OpenAI
import re
from KoBERTScore import BERTScore

model_name = "beomi/kcbert-base"
bertscore = BERTScore(model_name, best_layer=4)

os.environ["OPENAI_API_KEY"] = "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"
client_id = "1F6mOWD5pbGj50bbriBy"
client_secret = "jYxmCAlVae"
llm = OpenAI(temperature=0.9)

def translate_text(client_id, client_secret, text, source_lang='en', target_lang='ko'):
    url = "https://openapi.naver.com/v1/papago/n2mt"
    headers = {
        "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
        "X-Naver-Client-Id": client_id,
        "X-Naver-Client-Secret": client_secret
    }
    data = {
        "source": source_lang,
        "target": target_lang,
        "text": text
    }
    response = requests.post(url, headers=headers, data=data)
    return response.json().get("message", {}).get("result", {}).get("translatedText", "")

llm_result = llm.generate(["영어 번역 학습을 위한 50자 정도의 영어 원문을 제공해 줘"]*3)

user_translations = []
papago_translations = []

for gen_text in llm_result.generations:
    print("LLM이 생성한 원문: ", gen_text)
    user_translation = input("이 원문을 번역해주세요: ")
    user_translations.append(user_translation)

    papago_translation = translate_text(client_id, client_secret, gen_text)
    papago_translations.append(papago_translation)

print(bertscore(papago_translation, user_translation, batch_size=128))