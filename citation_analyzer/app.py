from flask import Flask, render_template, request
from markupsafe import Markup
import requests
import os
from dotenv import load_dotenv
import markdown as mk

load_dotenv()

app = Flask(__name__, template_folder='frontend')

GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')

def format_response_as_bullet_points(text):
    """Convert text into properly formatted markdown bullet points"""
    # Split text into lines and clean them
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    formatted_lines = []
    
    for line in lines:
        # If it's a citation or numbered item, make it a bullet point
        if line.startswith('*') and not line.startswith('* '):
            line = '* ' + line[1:].strip()
        elif line.startswith('•'):
            line = '* ' + line[1:].strip()
        elif not (line.startswith('* ') or line.startswith('- ')):
            # If it doesn't start with bullet point formatting already
            line = '* ' + line
        
        formatted_lines.append(line)
    
    return '\n'.join(formatted_lines)

def get_key_points(case_description):
    api_url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent"
    headers = {"Content-Type": "application/json"}
    payload = {
        "contents": [
            {
                "parts": [
                    {
                        "text": f"Extract important key points to handle this legal case. Format each point as a bullet point with an asterisk (*):\n{case_description}"
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
        key_points_text = format_response_as_bullet_points(key_points_text)
    except (KeyError, IndexError):
        key_points_text = "* Error: Unable to extract key points."
    
    return key_points_text

def get_court_questions(case_description):
    api_url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent"
    headers = {"Content-Type": "application/json"}
    payload = {
        "contents": [
            {
                "parts": [
                    {
                        "text": f"Suggest what questions a lawyer should ask in court based on this case. Format each question as a bullet point with an asterisk (*):\n{case_description}"
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
        questions_text = format_response_as_bullet_points(questions_text)
    except (KeyError, IndexError):
        questions_text = "* Error: Unable to generate court questions."
    
    return questions_text

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/process_case', methods=['POST'])
def process_case():
    case_description = request.form['case_description']
    key_points = get_key_points(case_description)
    court_questions = get_court_questions(case_description)
    
    # Convert markdown to HTML and mark as safe for rendering
    key_points_html = Markup(mk.markdown(key_points))
    court_questions_html = Markup(mk.markdown(court_questions))
    
    return render_template('index.html',
                          case_description=case_description,
                          key_points=key_points_html,
                          court_questions=court_questions_html)

if __name__ == '__main__':
    app.run(debug=True,port = 5045)