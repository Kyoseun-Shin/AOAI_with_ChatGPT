# AOAI_with_ChatGPT
openai의 api를 사용하여 중국집 서빙용 chat-bot을 구현하고 Kobert Model을 사용하여 embedding을 수행하는 프로그램

1. Chatbot을 구현하기 위해 openai api를 사용
   (https://mtcaichat01.openai.azure.com/openai/deployments/gpt432k/chat/completions?api-version=2023-03-15-preview (GPT4 32K token model)사용)

2. Embedding 생성/저장/배포를 위해 KoBert Model 사용

3. 선행학습을 위해 KoGPT2 model을 사용하였으며, 사용자의 질문에 우선 답변을 한 후 ChatGPT에 질의하는 것으로 구현

4. Dataset은 심리상담용 데이터셋을 사용하였으며, 형태는 질문/답의 쌍으로 구성되었다

5. Web Service를 구현하기 위해 Flask 사용

6. File 구성

   파이썬 Source : app.py
   HTML : template/index.html
   사전 학습된 모델 : chatgpt_model/*

7. 실행 방법
   python app3.py
