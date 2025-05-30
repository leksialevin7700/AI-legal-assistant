from flask import Flask, render_template, request
import requests
import os
from dotenv import load_dotenv
import markdown as mk

load_dotenv()

app = Flask(__name__, template_folder='frontend')

GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')

def get_key_points(case_description):
    api_url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent"
    headers = {"Content-Type": "application/json"}
    payload = {
        "contents": [
            {
                "parts": [
                    {
                        "text": f"Extract important key points to handle this legal case:\n{case_description}"
                    }
                ]
            }
        ]
    }
    params = {"key": GEMINI_API_KEY}
    response = requests.post(api_url, headers=headers, json=payload, params=params)
    result = response.json()
    
    try:
        key_points_text = result['candidates'][0]['content']['parts'][0]['text']
    except (KeyError, IndexError):
        key_points_text = "Error: Unable to extract key points."

    return key_points_text

def get_court_questions(case_description):
    api_url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent"
    headers = {"Content-Type": "application/json"}
    payload = {
        "contents": [
            {
                "parts": [
                    {
                        "text": f"Suggest what questions a lawyer should ask in court based on this case:\n{case_description}"
                    }
                ]
            }
        ]
    }
    params = {"key": GEMINI_API_KEY}
    response = requests.post(api_url, headers=headers, json=payload, params=params)
    result = response.json()
    
    try:
        questions_text = result['candidates'][0]['content']['parts'][0]['text']
    except (KeyError, IndexError):
        questions_text = "Error: Unable to generate court questions."

    return questions_text

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/process_case', methods=['POST'])
def process_case():
    case_description = request.form['case_description']
    key_points = mk.markdown(get_key_points(case_description))
    court_questions = mk.markdown(get_court_questions(case_description))
    
    return render_template('index.html', 
                           case_description=case_description, 
                           key_points=key_points, 
                           court_questions=court_questions)

if __name__ == '__main__':
    app.run(debug=True,port=5089)