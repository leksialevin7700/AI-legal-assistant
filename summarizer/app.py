"""
Document Summarization API

This Flask application provides an API endpoint for summarizing PDF and text documents.
It uses the BART-large-CNN model for text summarization and supports both PDF and TXT files.
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from transformers import pipeline
from typing import Optional
import pdfplumber
import os
import re
import logging

# Application Configuration
MAX_TEXT_LENGTH = 100000  # Maximum number of characters to process
MAX_SUMMARY_LENGTH = 5000  # Maximum length of generated summary
MIN_SUMMARY_LENGTH = 100  # Minimum length of generated summary
ALLOWED_EXTENSIONS = {'pdf', 'txt'}  # Supported file types
TEMP_DIR = "temp"  # Directory for temporary file storage
STATIC_DIR = "static"  # Directory for static files (HTML, CSS, JS)

# Initialize Flask application and CORS
app = Flask(__name__)
CORS(app)

# Configure logging
logging.basicConfig(level=logging.INFO)

# Initialize the summarization model
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

def is_valid_file_extension(filename: str) -> bool:
    """Validate if the uploaded file has an allowed extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_text_from_file(file_path: str, file_extension: str) -> Optional[str]:
    """Extract text content from the uploaded file based on its type."""
    try:
        if file_extension == 'pdf':
            with pdfplumber.open(file_path) as pdf_document:
                return " ".join(page.extract_text() or "" for page in pdf_document.pages).strip()
        else:  # txt file
            with open(file_path, 'r', encoding='utf-8') as text_file:
                return text_file.read().strip()
    except Exception as error:
        raise ValueError(f"Error extracting text: {str(error)}")

def format_summary_as_bullet_points(summary: str) -> str:
    """Format the generated summary into bullet points for better readability."""
    sentences = re.split(r'(?<=[.!?])\s+', summary.strip())
    bullet_points = [f"- {sentence.strip()}" for sentence in sentences if sentence.strip()]
    return "\n".join(bullet_points)

@app.route('/')
def index():
    """
    Serve the frontend HTML file (index.html) when the root URL is accessed.
    """
    return send_from_directory(STATIC_DIR, 'doc.html')

@app.route('/summarize', methods=['POST'])
def summarize_document():
    try:
        # Step 1: Validate file upload
        if 'file' not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        uploaded_file = request.files['file']
        if not uploaded_file or not uploaded_file.filename:
            return jsonify({"error": "Invalid file"}), 400

        # Step 2: Validate file extension
        if not is_valid_file_extension(uploaded_file.filename):
            return jsonify({"error": f"Unsupported file type. Allowed types: {', '.join(ALLOWED_EXTENSIONS)}"}), 400

        # Step 3: Save and extract text
        os.makedirs(TEMP_DIR, exist_ok=True)
        temporary_file_path = os.path.join(TEMP_DIR, uploaded_file.filename)
        uploaded_file.save(temporary_file_path)
        file_extension = uploaded_file.filename.rsplit('.', 1)[1].lower()
        extracted_text = extract_text_from_file(temporary_file_path, file_extension)

        # Step 4: Validate extracted text
        if not extracted_text:
            return jsonify({"error": "No text could be extracted from the file"}), 400
        if len(extracted_text.split()) < 50:
            return jsonify({"error": "Input text is too short for summarization."}), 400

        # Step 5: Truncate text if it exceeds maximum length
        if len(extracted_text) > MAX_TEXT_LENGTH:
            extracted_text = extracted_text[:MAX_TEXT_LENGTH]

        # Step 6: Dynamically adjust max_length and min_length
        input_length = len(extracted_text.split())
        max_output_length = min(MAX_SUMMARY_LENGTH, int(input_length * 0.5))
        min_output_length = max(MIN_SUMMARY_LENGTH, int(input_length * 0.2))
        min_output_length = min(min_output_length, max_output_length)

        # Step 7: Generate summary using the BART model
        generated_summary = summarizer(
            extracted_text,
            max_length=max_output_length,
            min_length=min_output_length,
            do_sample=False
        )[0]['summary_text']

        # Step 8: Format the summary as bullet points
        formatted_summary = format_summary_as_bullet_points(generated_summary)

        # Step 9: Return successful response with summary
        return jsonify({
            "summary": formatted_summary,
            "original_length": len(extracted_text),
            "summary_length": len(generated_summary)
        })

    except ValueError as ve:
        return jsonify({
            "error": "ValueError",
            "details": str(ve)
        }), 400
    except Exception as error:
        import traceback
        traceback.print_exc()  # Log the full traceback for debugging
        return jsonify({
            "error": "An unexpected error occurred while processing the file",
            "details": str(error)
        }), 500
    finally:
        # Step 10: Clean up - Remove temporary file
        if os.path.exists(temporary_file_path):
            os.remove(temporary_file_path)

if __name__ == '__main__':
    app.run(debug=True, port=3003)