from flask import Flask, render_template, request
from markupsafe import Markup
import requests
import os
from dotenv import load_dotenv
import markdown as mk
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SimpleSequentialChain
from langchain_google_genai import ChatGoogleGenerativeAI

# Load environment variables
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

app = Flask(__name__, template_folder='frontend')

# Load Legal-BERT model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("nlpaueb/legal-bert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained("nlpaueb/legal-bert-base-uncased")

# Initialize LangChain LLM
llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=GEMINI_API_KEY)

# LangChain prompt templates
summarize_prompt = PromptTemplate.from_template("Summarize the following legal case:\n{input}")
question_prompt = PromptTemplate.from_template("Suggest questions a lawyer should ask in court based on this summary:\n{input}")

summarize_chain = LLMChain(llm=llm, prompt=summarize_prompt)
question_chain = LLMChain(llm=llm, prompt=question_prompt)

langchain_chain = SimpleSequentialChain(chains=[summarize_chain, question_chain], verbose=False)

def format_response_as_bullet_points(text):
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    formatted_lines = []
    for line in lines:
        if line.startswith('*') and not line.startswith('* '):
            line = '* ' + line[1:].strip()
        elif line.startswith('•'):
            line = '* ' + line[1:].strip()
        elif not (line.startswith('* ') or line.startswith('- ')):
            line = '* ' + line
        formatted_lines.append(line)
    return '\n'.join(formatted_lines)

def get_key_points(case_description):
    api_url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent"
    headers = {"Content-Type": "application/json"}
    payload = {
        "contents": [{
            "parts": [{
                "text": f"Extract important key points to handle this legal case. Format each point as a bullet point with an asterisk (*):\n{case_description}"
            }]
        }]
    }
    params = {"key": GEMINI_API_KEY}
    response = requests.post(api_url, headers=headers, json=payload, params=params)
    result = response.json()
    try:
        key_points_text = result['candidates'][0]['content']['parts'][0]['text']
        return format_response_as_bullet_points(key_points_text)
    except (KeyError, IndexError):
        return "* Error: Unable to extract key points."

def get_court_questions(case_description):
    api_url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent"
    headers = {"Content-Type": "application/json"}
    payload = {
        "contents": [{
            "parts": [{
                "text": f"Suggest what questions a lawyer should ask in court based on this case. Format each question as a bullet point with an asterisk (*):\n{case_description}"
            }]
        }]
    }
    params = {"key": GEMINI_API_KEY}
    response = requests.post(api_url, headers=headers, json=payload, params=params)
    result = response.json()
    try:
        questions_text = result['candidates'][0]['content']['parts'][0]['text']
        return format_response_as_bullet_points(questions_text)
    except (KeyError, IndexError):
        return "* Error: Unable to generate court questions."

def classify_case_bert(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        logits = model(**inputs).logits
    predicted_class = torch.argmax(logits, dim=1).item()
    return f"Predicted Case Category (Legal-BERT): Class {predicted_class}"

def get_langchain_questions(text):
    try:
        return langchain_chain.run(text)
    except Exception:
        return "* LangChain error: Unable to generate response."

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/process_case', methods=['POST'])
def process_case():
    case_description = request.form['case_description']
    
    key_points = get_key_points(case_description)
    court_questions = get_court_questions(case_description)
    bert_classification = classify_case_bert(case_description)
    langchain_generated = get_langchain_questions(case_description)

    return render_template('index.html',
                           case_description=case_description,
                           key_points=Markup(mk.markdown(key_points)),
                           court_questions=Markup(mk.markdown(court_questions)),
                           bert_classification=bert_classification,
                           langchain_questions=Markup(mk.markdown(langchain_generated)))

if __name__ == '__main__':
    app.run(debug=True)
