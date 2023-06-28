# AOAI_with_ChatGPT
openai의 api를 사용하여 중국집 서빙용 chat-bot을 구현하고 Kobert Model을 사용하여 embedding을 수행하는 프로그램

1. Chatbot을 구현하기 위해 openai api를 사용
   (https://mtcaichat01.openai.azure.com/openai/deployments/gpt432k/chat/completions?api-version=2023-03-15-preview (GPT4 32K token model)사용)

2. Embedding 생성/저장/배포를 위해 KoBert Model 사용

3. Web Service를 구현하기 위해 Flask 사용

4. File 구성

   파이썬 Source : app.py
   HTML : template/index.html
   배포된 embedding model : embeddingada002.pt

5. 실행 방법
   python app.py
