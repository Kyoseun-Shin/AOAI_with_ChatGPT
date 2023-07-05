import requests
import openai
import torch
import numpy as np
from dotenv.main import load_dotenv
from flask import Flask, render_template,jsonify,request
from flask_cors import CORS

import os
import tempfile
import PyPDF2
from transformers import pipeline, GPTNeoConfig, GPT2LMHeadModel, Trainer, TextDataset, TrainingArguments, DataCollatorForLanguageModeling, GPTNeoForCausalLM, GPT2Tokenizer, TextGenerationPipeline
from transformers import AutoTokenizer

tokenizer_path = "skt/kogpt2-base-v2"

tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

os.environ["OPENAI_API_KEY"] = "824fe43e851f4862af326fa83c3d3cfe"
pdf_directory = './data'

# PDF 파일 추출 및 텍스트 변환 함수
def extract_text_from_pdf(file_path):
    with open(file_path, 'rb') as file:
        pdf = PyPDF2.PdfReader(file_path)
        text = ''
        for page in pdf.pages:
            text += page.extract_text()
        return text.strip()  # 수정: 텍스트의 앞뒤 공백 제거 





        
def generate_chat_response(user_input):

    model_path = './chatgpt_model'
    model = GPT2LMHeadModel.from_pretrained('skt/kogpt2-base-v2') 
    tokenizer = AutoTokenizer.from_pretrained('skt/kogpt2-base-v2')
    
     # 입력 텍스트 토큰화
    input_ids = tokenizer.encode(user_input, return_tensors='pt')

    # 모델에 입력 전달하여 답변 생성
    output = model.generate(input_ids, max_length=10, num_return_sequences=1) 
    print(output)

    # 생성된 답변 디코딩
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    print(generated_text)

    openai.api_type = "azure"
    openai.api_version = "2023-03-15-preview" 
    openai.api_base = "https://mtcaichat01.openai.azure.com"  # Your Azure OpenAI resource's endpoint value.
    openai.api_key = "824fe43e851f4862af326fa83c3d3cfe"
    #openai.api_key = "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"

    response = openai.ChatCompletion.create(
    engine="gpt432k", # The deployment name you chose when you deployed the ChatGPT or GPT-4 model.
    messages=[
        {"role": "system", "content": "너는 비서야"},
        {"role": "user", "content": generated_text}
    ]
    )
    print(response['choices'][0]['message']['content'])
    return response['choices'][0]['message']['content'] 

# Flask를 사용하여 웹 애플리케이션 개발
from flask import Flask, render_template, request, jsonify

app = Flask(__name__) 

CORS(app)

dataset = []  # 데이터셋 초기화
labels = []   # 레이블 초기화


@app.route('/')
def home():
    return render_template('index.html') 

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    text = data.get('data')
    
    #user_input = index.query(text)
    user_input = text
    response = generate_chat_response(user_input)

    return jsonify({"response":True, "message":response})
    




@app.route('/upload', methods=['POST'])
def upload():
    print('upload') 
    # PDF 파일 경로에서 텍스트 추출 
    if request.method == 'POST':
        file = request.files['files']
        if file:
            file_path = './data/' + file.filename  
            file.save(file_path)
    
    return 'Completed upload'

@app.route('/train', methods=['POST'])
def train(): 
    print('train') 
    pdf_files = os.listdir(pdf_directory) 
    texts = []
    for file in pdf_files:
        file_path = os.path.join(pdf_directory, file)
        print(file_path) 
        if file.endswith('.pdf'):
            text = extract_text_from_pdf(file_path)
            texts.append(text)

    print('\n'.join(texts))

    # 토크나이저와 모델 로드
    tokenizer = AutoTokenizer.from_pretrained('skt/kogpt2-base-v2')
    model = GPT2LMHeadModel.from_pretrained('skt/kogpt2-base-v2')

    # 텍스트 데이터를 임시 파일에 저장
    input_file = os.path.join(r'C:\Users\kyose\OneDrive\문서\hello\data\input.txt')
            
    with open(input_file, "w", encoding="utf-8") as f:
        source_text = '\n'.join(texts)  # 리스트를 개행문자를 기준으로 이어서 문자열로 변환
        f.write(source_text)

    # 데이터셋 생성
    dataset = TextDataset(
        file_path=input_file,
        tokenizer=tokenizer,
        block_size=128
    )

    print(input_file)
    #
    ## 데이터 콜레이터 생성
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    output_dir = os.path.join(".","/output")
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    #
    # 학습 인자 설정
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=3,
        per_device_train_batch_size=4,
        save_steps=10_000,
        save_total_limit=2
    )

    # Trainer 객체 생성 및 학습 실행
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset
    )

    trainer.train()
    # 학습된 모델 저장
    trainer.save_model('./chatgpt_model')
    return 'Completed train'
    
if __name__ == '__main__':
    app.run() 