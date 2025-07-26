from flask import Flask, render_template, request
from utils import auto_pipeline

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def summarize():
    summary = ""
    easy_summary = ""
    warning = ""

    if request.method == 'POST':
        input_text = request.form.get('input_text', '')

        if len(input_text.strip()) < 20:
            warning = "Please enter a longer legal text!"
        else:
            summary, easy_summary = auto_pipeline(input_text)

    return render_template(
        'doc.html',
        summary=summary,
        easy_summary=easy_summary,
        warning=warning
    )

if __name__ == '__main__':
     app.run(debug=True, port=3003)
