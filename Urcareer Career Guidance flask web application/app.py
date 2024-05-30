from flask import Flask, redirect, render_template, request, jsonify, url_for
from flask_pymongo import PyMongo
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import fitz
from docx import Document
import os
import openai

app = Flask(__name__, template_folder='templates', static_folder='static')
app.config["MONGO_URI"] = "mongodb://localhost:27017/urcareer"
app.config['UPLOAD_FOLDER'] = 'uploads'
mongo = PyMongo(app)
collection = mongo.db.resume

nltk.download('punkt')
nltk.download('stopwords')

stop_words = set(stopwords.words('english'))

# OpenAI API Key
openai.api_key = 'sk-proj-BjoiOBOg0X4i62LyctGmT3BlbkFJcSGzvmmgnnAhQo8V9DDH'

# Function to extract text from PDF files
def extract_text_from_pdf(file_path):
    text = ""
    try:
        with fitz.open(file_path) as doc:
            for page in doc:
                text += page.get_text()
    except Exception as e:
        app.logger.error('An error occurred while extracting text from PDF: %s', str(e))
    return text

# Function to extract text from DOCX files
def extract_text_from_docx(file_path):
    text = ""
    try:
        doc = Document(file_path)
        for para in doc.paragraphs:
            text += para.text
    except Exception as e:
        app.logger.error('An error occurred while extracting text from DOCX: %s', str(e))
    return text

# Function to extract keywords from the resume content
def extract_keywords(resume_content):
    words = word_tokenize(resume_content)
    keywords = [word.lower() for word in words if word.isalnum() and word.lower() not in stop_words]
    word_freq = nltk.FreqDist(keywords)
    return word_freq

# Function to assign weightage to keywords
def assign_weightage(keywords):
    weighted_keywords = {keyword: freq * 10 for keyword, freq in keywords.items()}
    return weighted_keywords

@app.route('/upload_resume', methods=['GET', 'POST'])
def upload_resume():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
        if file:
            filename = file.filename
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            if filename.endswith('.pdf'):
                resume_content = extract_text_from_pdf(file_path)
            elif filename.endswith('.docx'):
                resume_content = extract_text_from_docx(file_path)
            else:
                return jsonify({'error': 'Unsupported file format'}), 400
            
            keywords = extract_keywords(resume_content)
            weighted_keywords = assign_weightage(keywords)
            mongo.db.keywords.insert_one({'keywords': weighted_keywords})
           
            return redirect(url_for('generate_assessment')), 302
    except Exception as e:
        app.logger.error('An error occurred while uploading resume: %s', str(e))
        return jsonify({'error': 'An error occurred while uploading resume'}), 500

# Function to generate MCQs using OpenAI API
def generate_mcqs(keyword):
    prompt = f"Create a multiple-choice question about {keyword}. Provide 4 options and indicate the correct one."
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message['content'].strip()

@app.route('/generate_assessment', methods=['GET', 'POST'])
def generate_assessment():
    try:
        keywords_data = mongo.db.keywords.find_one(sort=[('_id', -1)])
        if not keywords_data:
            return jsonify({'error': 'No keywords found in the database'}), 400

        keywords = keywords_data['keywords']
        highest_weighted_keyword = max(keywords, key=keywords.get)

        questions = []
        for _ in range(3):
            questions.append(generate_mcqs(highest_weighted_keyword))

        return render_template('generate_assessment.html', questions=questions)
    except Exception as e:
        app.logger.error('An error occurred while generating assessment: %s', str(e))
        return jsonify({'error': 'An error occurred while generating assessment'}), 500

@app.route('/submit_assessment', methods=['GET', 'POST'])
def submit_assessment():
    try:
        score = 80
        return jsonify({'score': score}), 200
    except Exception as e:
        app.logger.error('An error occurred while submitting assessment: %s', str(e))
        return jsonify({'error': 'An error occurred while submitting assessment'}), 500

@app.route('/view_score', methods=['GET', 'POST'])
def view_score():
    try:
        score = 80
        return render_template('view_score.html', score=score)
    except Exception as e:
        app.logger.error('An error occurred while retrieving score: %s', str(e))
        return jsonify({'error': 'An error occurred while retrieving score'}), 500

@app.route('/recommend_job', methods=['GET', 'POST'])
def recommend_job():
    try:
        recommendation = "Software Engineer"
        return render_template('recommend_job.html', recommendation=recommendation)
    except Exception as e:
        app.logger.error('An error occurred while recommending job: %s', str(e))
        return jsonify({'error': 'An error occurred while recommending job'}), 500

@app.route('/')
def index():
    return render_template('welcome.html')

if __name__ == '__main__':
    app.run(debug=True)
