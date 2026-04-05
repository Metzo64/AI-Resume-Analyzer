"""
Flask Application for AI Resume Analyzer
Main backend server handling resume upload, analysis, and results
"""

from flask import Flask, render_template, request, jsonify
import pandas as pd
import os
from modules.parser import ResumeParser
from modules.classifier import ResumeClassifier
from modules.matcher import ResumeMatcher

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Initialize modules
parser = ResumeParser()
classifier = ResumeClassifier()
matcher = ResumeMatcher()

# Load job descriptions
job_df = pd.read_csv('data/job_title_des.csv')

# Try to load trained model
try:
    classifier.load_model()
    print("✓ Classifier model loaded successfully")
except FileNotFoundError:
    print("⚠ Warning: Classifier model not found. Please train the model first.")
    print("  Run: python modules/classifier.py")


@app.route('/')
def index():
    """Render home page with upload form"""
    # Get unique job titles for dropdown
    job_titles = job_df['Job Title'].unique().tolist()
    return render_template('index.html', job_titles=job_titles)


@app.route('/analyze', methods=['POST'])
def analyze():
    """
    Analyze resume and return results
    Handles file upload only
    """
    try:
        # Get resume text from file upload
        resume_text = None
        
        if 'resume_file' not in request.files:
            return render_template('result.html', 
                                 error="No file uploaded. Please upload a resume file.")
        
        file = request.files['resume_file']
        
        if not file or not file.filename:
            return render_template('result.html', 
                                 error="No file selected. Please choose a resume file to upload.")
        
        # Read file content
        try:
            resume_text = file.read().decode('utf-8', errors='ignore')
        except Exception as e:
            return render_template('result.html', 
                                 error=f"Unable to read file. Please ensure it's a valid text file. Error: {str(e)}")
        
        if not resume_text or len(resume_text.strip()) < 50:
            return render_template('result.html', 
                                 error="Resume content is too short. Please provide a complete resume (minimum 50 characters).")
        
        # Get selected job title
        selected_job = request.form.get('job_title', '')
        
        if not selected_job:
            return render_template('result.html', 
                                 error="Please select a target job role for comparison.")
        
        # Step 1: Parse resume
        parsed_data = parser.parse_resume(resume_text)
        
        # Step 2: Classify resume
        try:
            category = classifier.predict(parsed_data['cleaned_text'])
        except Exception as e:
            print(f"Classification error: {e}")
            category = "Unable to classify (model not trained)"
        
        # Step 3: Match with job description
        match_score = 0
        job_description = ""
        
        if selected_job:
            # Find job description
            job_row = job_df[job_df['Job Title'] == selected_job]
            if not job_row.empty:
                job_description = job_row.iloc[0]['Job Description']
                match_score = matcher.compute_similarity(
                    parsed_data['cleaned_text'], 
                    job_description
                )
        
        # Prepare results
        results = {
            'category': category,
            'match_score': match_score,
            'skills': parsed_data['skills'],
            'email': parsed_data['email'],
            'job_title': selected_job,
            'job_description': job_description[:200] + '...' if len(job_description) > 200 else job_description
        }
        
        return render_template('result.html', results=results)
    
    except Exception as e:
        print(f"Error in analyze: {e}")
        return render_template('result.html', 
                             error=f"An unexpected error occurred during analysis. Please try again.")


@app.route('/api/analyze', methods=['POST'])
def api_analyze():
    """
    API endpoint for resume analysis
    Returns JSON response
    """
    try:
        data = request.get_json()
        resume_text = data.get('resume_text', '')
        job_description = data.get('job_description', '')
        
        if not resume_text:
            return jsonify({'error': 'Resume text is required'}), 400
        
        # Parse resume
        parsed_data = parser.parse_resume(resume_text)
        
        # Classify resume
        try:
            category = classifier.predict(parsed_data['cleaned_text'])
        except:
            category = "Unable to classify"
        
        # Match with job
        match_score = 0
        if job_description:
            match_score = matcher.compute_similarity(
                parsed_data['cleaned_text'], 
                job_description
            )
        
        return jsonify({
            'category': category,
            'match_score': match_score,
            'skills': parsed_data['skills'],
            'email': parsed_data['email']
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'message': 'AI Resume Analyzer is running'})


if __name__ == '__main__':
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    print("\n" + "="*60)
    print("🚀 AI Resume Analyzer System")
    print("="*60)
    print("📊 Parser: Ready")
    print("🧠 Classifier: " + ("Ready" if classifier.model else "Not trained"))
    print("🔥 Matcher: Ready")
    print("="*60 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
