"""
Resume Parser Module
Extracts key information from resume text including skills, email, and cleaned text
"""

import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')


class ResumeParser:
    def __init__(self):
        """Initialize parser with predefined skill keywords"""
        self.skills_keywords = [
            'python', 'java', 'javascript', 'c++', 'c#', 'ruby', 'php', 'swift', 'kotlin',
            'react', 'angular', 'vue', 'node', 'django', 'flask', 'spring', 'express',
            'sql', 'mysql', 'postgresql', 'mongodb', 'oracle', 'redis', 'cassandra',
            'aws', 'azure', 'gcp', 'docker', 'kubernetes', 'jenkins', 'git', 'ci/cd',
            'machine learning', 'deep learning', 'nlp', 'computer vision', 'tensorflow',
            'pytorch', 'scikit-learn', 'pandas', 'numpy', 'data analysis', 'data science',
            'html', 'css', 'bootstrap', 'tailwind', 'sass', 'rest api', 'graphql',
            'agile', 'scrum', 'jira', 'leadership', 'communication', 'teamwork',
            'problem solving', 'project management', 'excel', 'powerpoint', 'word'
        ]
        self.stop_words = set(stopwords.words('english'))
    
    def extract_email(self, text):
        """Extract email address from text using regex"""
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        emails = re.findall(email_pattern, text)
        return emails[0] if emails else None
    
    def extract_skills(self, text):
        """Extract skills from text by matching predefined keywords"""
        text_lower = text.lower()
        found_skills = []
        
        for skill in self.skills_keywords:
            # Use word boundaries to match whole words/phrases
            pattern = r'\b' + re.escape(skill) + r'\b'
            if re.search(pattern, text_lower):
                found_skills.append(skill.title())
        
        return list(set(found_skills))  # Remove duplicates
    
    def clean_text(self, text):
        """Clean and preprocess text"""
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', ' ', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords
        filtered_tokens = [word for word in tokens if word not in self.stop_words and len(word) > 2]
        
        return ' '.join(filtered_tokens)
    
    def parse_resume(self, resume_text):
        """
        Main parsing function that extracts all information
        
        Args:
            resume_text (str): Raw resume text
            
        Returns:
            dict: Parsed resume information
        """
        return {
            'email': self.extract_email(resume_text),
            'skills': self.extract_skills(resume_text),
            'cleaned_text': self.clean_text(resume_text),
            'original_text': resume_text
        }
