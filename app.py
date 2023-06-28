import requests
import openai
from transformers import BertTokenizer, BertModel
import torch
import numpy as np


def generate_chat_response(user_input):

    openai.api_type = "azure"
    openai.api_version = "2023-03-15-preview" 
    openai.api_base = "https://mtcaichat01.openai.azure.com"  # Your Azure OpenAI resource's endpoint value.
    openai.api_key = "824fe43e851f4862af326fa83c3d3cfe"

    response = openai.ChatCompletion.create(
    engine="gpt432k", # The deployment name you chose when you deployed the ChatGPT or GPT-4 model.
    messages=[
        {"role": "system", "content": "너는 중국집 서빙 직원이야"},
        {"role": "user", "content": user_input}
    ]
    )
    #response = requests.post(url, headers=headers, json=payload)
    print(response['choices'][0]['message']['content'])
    return response['choices'][0]['message']['content'] 

import tensorflow as tf

# Flask를 사용하여 웹 애플리케이션 개발
from flask import Flask, render_template, request, jsonify

class TextEmbeddingModel:
    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained("monologg/kobert")
        self.model = BertModel.from_pretrained("monologg/kobert")

    def get_embedding(self, text):
        input_ids = torch.tensor([self.tokenizer.encode(text, add_special_tokens=True)])
        with torch.no_grad():
            outputs = self.model(input_ids)
            embedding = torch.mean(outputs.last_hidden_state, dim=1).squeeze(0)
            torch.save(embedding, "embeddingada002.pt")
        return embedding.numpy()

# Text embedding model instantiation
model = TextEmbeddingModel()
      
#def embed_text(self, text):
#    embedding = model.get_embedding(text)
#    print(embedding)
#    return jsonify(embedding)
#

app = Flask(__name__) 

@app.route('/')
def home():
    return render_template('index.html') 

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    text = data.get('data')
    
    user_input = text
    #user_input = request.form['userinput']
    response = generate_chat_response(user_input)
    embedding1 = model.get_embedding(user_input)
    embedding2 = model.get_embedding(response)

    # Calculate similarity between embeddings
    similarity_score = np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))
    print(similarity_score)

    return jsonify({"response":True, "message":response}) 

if __name__ == '__main__':
    app.run() 
