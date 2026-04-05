"""
Test Script for AI Resume Analyzer Modules
Tests parser and matcher without requiring trained classifier
"""

import sys

def test_parser():
    """Test resume parser module"""
    print("\n" + "="*70)
    print("🧪 Testing Resume Parser")
    print("="*70 + "\n")
    
    try:
        from modules.parser import ResumeParser
        
        parser = ResumeParser()
        
        # Sample resume
        sample_resume = """
        John Doe
        Email: john.doe@example.com
        
        SUMMARY
        Experienced Python developer with 5 years in machine learning and data science.
        Proficient in TensorFlow, PyTorch, scikit-learn, and deep learning.
        
        SKILLS
        - Python, Java, JavaScript, SQL
        - Machine Learning, Deep Learning, NLP
        - TensorFlow, PyTorch, Keras
        - AWS, Docker, Kubernetes
        - React, Node.js, Flask
        
        EXPERIENCE
        Senior Data Scientist | Tech Corp | 2020-Present
        - Built ML models using Python and TensorFlow
        - Deployed models on AWS with Docker
        """
        
        print("📄 Sample Resume:")
        print("-" * 70)
        print(sample_resume[:200] + "...")
        print("-" * 70 + "\n")
        
        # Parse resume
        result = parser.parse_resume(sample_resume)
        
        print("✅ Parsing Results:")
        print(f"   Email: {result['email']}")
        print(f"   Skills Found: {len(result['skills'])}")
        print(f"   Skills: {', '.join(result['skills'][:10])}")
        if len(result['skills']) > 10:
            print(f"           ... and {len(result['skills']) - 10} more")
        print(f"   Cleaned Text Length: {len(result['cleaned_text'])} characters")
        
        print("\n✅ Parser Test: PASSED\n")
        return True
        
    except Exception as e:
        print(f"\n❌ Parser Test: FAILED")
        print(f"   Error: {str(e)}\n")
        return False


def test_matcher():
    """Test resume-job matcher module"""
    print("="*70)
    print("🧪 Testing Resume-Job Matcher")
    print("="*70 + "\n")
    
    try:
        from modules.matcher import ResumeMatcher
        
        print("📦 Loading Sentence-BERT model...")
        matcher = ResumeMatcher()
        print("✅ Model loaded successfully!\n")
        
        # Sample data
        resume = """
        Experienced Python developer with 5 years in machine learning and data science.
        Proficient in TensorFlow, PyTorch, scikit-learn, pandas, and numpy.
        Built and deployed ML models on AWS. Strong background in deep learning and NLP.
        """
        
        job_desc = """
        We are looking for a Python Developer with machine learning experience.
        Must have experience with TensorFlow, PyTorch, and deploying ML models.
        Knowledge of AWS and deep learning is a plus.
        """
        
        print("📄 Resume Summary:")
        print("-" * 70)
        print(resume.strip())
        print("-" * 70 + "\n")
        
        print("💼 Job Description:")
        print("-" * 70)
        print(job_desc.strip())
        print("-" * 70 + "\n")
        
        # Compute similarity
        print("🔄 Computing similarity...")
        match_score = matcher.compute_similarity(resume, job_desc)
        
        print(f"\n✅ Match Score: {match_score}%")
        
        if match_score >= 80:
            print("   Interpretation: Excellent Match! 🎯")
        elif match_score >= 60:
            print("   Interpretation: Good Match! ✓")
        elif match_score >= 40:
            print("   Interpretation: Moderate Match")
        else:
            print("   Interpretation: Low Match")
        
        print("\n✅ Matcher Test: PASSED\n")
        return True
        
    except Exception as e:
        print(f"\n❌ Matcher Test: FAILED")
        print(f"   Error: {str(e)}\n")
        return False


def test_classifier():
    """Test resume classifier (if model exists)"""
    print("="*70)
    print("🧪 Testing Resume Classifier")
    print("="*70 + "\n")
    
    try:
        from modules.classifier import ResumeClassifier
        
        classifier = ResumeClassifier()
        
        print("📦 Loading trained model...")
        classifier.load_model()
        print("✅ Model loaded successfully!\n")
        
        # Sample resume
        sample_resume = """
        Data Scientist with expertise in machine learning and statistical analysis.
        Proficient in Python, R, TensorFlow, and scikit-learn.
        Experience with data visualization, predictive modeling, and big data.
        """
        
        print("📄 Sample Resume:")
        print("-" * 70)
        print(sample_resume.strip())
        print("-" * 70 + "\n")
        
        # Predict category
        print("🔄 Predicting category...")
        category = classifier.predict(sample_resume)
        
        print(f"\n✅ Predicted Category: {category}")
        print("\n✅ Classifier Test: PASSED\n")
        return True
        
    except FileNotFoundError:
        print("⚠️  Model not found - This is expected if you haven't trained yet")
        print("   Run: python train_model.py")
        print("\n⏭️  Classifier Test: SKIPPED\n")
        return None
        
    except Exception as e:
        print(f"\n❌ Classifier Test: FAILED")
        print(f"   Error: {str(e)}\n")
        return False


def main():
    """Run all tests"""
    print("\n" + "="*70)
    print("🧪 AI RESUME ANALYZER - MODULE TESTS")
    print("="*70)
    
    results = {
        'parser': test_parser(),
        'matcher': test_matcher(),
        'classifier': test_classifier()
    }
    
    print("="*70)
    print("📊 TEST SUMMARY")
    print("="*70 + "\n")
    
    for module, result in results.items():
        if result is True:
            print(f"   ✅ {module.capitalize()}: PASSED")
        elif result is False:
            print(f"   ❌ {module.capitalize()}: FAILED")
        else:
            print(f"   ⏭️  {module.capitalize()}: SKIPPED")
    
    print("\n" + "="*70)
    
    # Overall result
    passed = sum(1 for r in results.values() if r is True)
    failed = sum(1 for r in results.values() if r is False)
    skipped = sum(1 for r in results.values() if r is None)
    
    print(f"\n📈 Results: {passed} passed, {failed} failed, {skipped} skipped")
    
    if failed == 0:
        print("\n🎉 All tests passed! System is ready to use.")
        if skipped > 0:
            print("   Note: Train the classifier to enable full functionality")
    else:
        print("\n⚠️  Some tests failed. Please check the errors above.")
    
    print("\n" + "="*70 + "\n")
    
    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
