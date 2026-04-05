"""
Resume-Job Matcher Module
Uses Sentence-BERT to compute similarity between resume and job description
"""

from sentence_transformers import SentenceTransformer, util
import torch


class ResumeMatcher:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        """
        Initialize matcher with Sentence-BERT model
        
        Args:
            model_name (str): Name of the sentence transformer model
        """
        print(f"Loading Sentence-BERT model: {model_name}...")
        self.model = SentenceTransformer(model_name)
        print("Model loaded successfully!")
    
    def compute_similarity(self, resume_text, job_description):
        """
        Compute cosine similarity between resume and job description
        
        Args:
            resume_text (str): Resume text
            job_description (str): Job description text
            
        Returns:
            float: Match score (0-100%)
        """
        # Generate embeddings
        resume_embedding = self.model.encode(resume_text, convert_to_tensor=True)
        job_embedding = self.model.encode(job_description, convert_to_tensor=True)
        
        # Compute cosine similarity
        cosine_score = util.cos_sim(resume_embedding, job_embedding)
        
        # Convert to percentage (0-100)
        match_score = float(cosine_score[0][0]) * 100
        
        return round(match_score, 2)
    
    def match_with_multiple_jobs(self, resume_text, job_descriptions):
        """
        Match resume with multiple job descriptions
        
        Args:
            resume_text (str): Resume text
            job_descriptions (list): List of job description texts
            
        Returns:
            list: List of tuples (job_index, match_score) sorted by score
        """
        resume_embedding = self.model.encode(resume_text, convert_to_tensor=True)
        job_embeddings = self.model.encode(job_descriptions, convert_to_tensor=True)
        
        # Compute similarities
        similarities = util.cos_sim(resume_embedding, job_embeddings)[0]
        
        # Create results list
        results = []
        for idx, score in enumerate(similarities):
            match_score = float(score) * 100
            results.append((idx, round(match_score, 2)))
        
        # Sort by score (descending)
        results.sort(key=lambda x: x[1], reverse=True)
        
        return results
    
    def get_top_matches(self, resume_text, job_descriptions, top_k=5):
        """
        Get top K matching job descriptions
        
        Args:
            resume_text (str): Resume text
            job_descriptions (list): List of job description texts
            top_k (int): Number of top matches to return
            
        Returns:
            list: Top K matches with scores
        """
        all_matches = self.match_with_multiple_jobs(resume_text, job_descriptions)
        return all_matches[:top_k]


# Test script
if __name__ == "__main__":
    matcher = ResumeMatcher()
    
    # Example usage
    resume = "Experienced Python developer with 5 years in machine learning and data science. Proficient in TensorFlow, PyTorch, and scikit-learn."
    job_desc = "Looking for a Python developer with machine learning experience. Must know TensorFlow and PyTorch."
    
    score = matcher.compute_similarity(resume, job_desc)
    print(f"Match Score: {score}%")
