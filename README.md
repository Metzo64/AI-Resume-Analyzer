# 🤖 AI Resume Analyzer

An intelligent resume analysis system powered by deep learning and natural language processing. Upload your resume, select a target job role, and get instant AI-powered insights including category classification, job match scoring, and skill extraction.

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![Flask](https://img.shields.io/badge/Flask-2.3.2-green.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.4.1-red.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

---

## ✨ Features

- **🔍 Smart Resume Parsing** - Extracts skills, email, and key information using NLP
- **🎯 AI Classification** - BERT-based deep learning model categorizes resumes into 24 professional domains
- **📊 Job Matching** - Semantic similarity analysis provides 0-100% compatibility scores
- **🎨 Modern UI** - Beautiful, professional interface with glassmorphism design
- **🔗 REST API** - JSON-based API for programmatic access

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the Model
```bash
python train_model.py
```
⏱️ Takes 30-60 minutes on CPU

### 3. Run the Application
```bash
python app.py
```

### 4. Open in Browser
```
http://localhost:5000
```

---

## 📋 Requirements

- Python 3.7+
- 2GB free disk space
- Internet connection (for initial model downloads)

---

## 🎯 How It Works

1. **Upload Resume** - Drag and drop or click to upload (TXT, PDF, DOC, DOCX)
2. **Select Job Role** - Choose target position from dropdown
3. **Get Results** - View category, match score, skills, and recommendations

---

## 🧠 Technology Stack

- **Backend**: Flask, Python
- **Deep Learning**: PyTorch, BERT, Sentence-BERT
- **NLP**: NLTK, HuggingFace Transformers
- **Data**: Pandas, NumPy, Scikit-learn
- **Frontend**: HTML5, CSS3, JavaScript

---

## 📊 Model Performance

- **Classification Accuracy**: 85-90%
- **Categories**: 24 professional domains
- **Skills Detected**: 50+ technical and soft skills
- **Inference Time**: 2-3 seconds per resume

---

## 🎨 Screenshots

### Home Page
Beautiful gradient background with glassmorphism effects and easy file upload.

### Results Page
Comprehensive analysis with category, match score, skills, and actionable recommendations.

---

## 📖 Documentation

For detailed documentation including architecture, API reference, and training process, see [PROJECT_DOCUMENTATION.md](PROJECT_DOCUMENTATION.md).

---

## 🔧 Configuration

### Change Port
Edit `app.py`:
```python
app.run(debug=True, port=5001)  # Change from 5000 to 5001
```

### Adjust Training
Edit `train_model.py`:
```python
classifier.train(
    csv_path='data/Resume.csv',
    epochs=3,        # Increase epochs
    batch_size=8,    # Increase if you have more RAM
    learning_rate=2e-5
)
```

---

## 🐛 Troubleshooting

### Model Not Found
```bash
python train_model.py
```

### Out of Memory
Reduce batch size in `train_model.py`:
```python
batch_size=2  # Instead of 4
```

### Port Already in Use
Change port in `app.py` or kill existing process.

---

## 📁 Project Structure

```
resume_project/
├── data/                    # Datasets
│   ├── Resume.csv          # Resume training data
│   └── job_title_des.csv   # Job descriptions
├── models/                  # Trained models (created after training)
├── modules/                 # Core modules
│   ├── parser.py           # Resume parsing
│   ├── classifier.py       # BERT classification
│   └── matcher.py          # Job matching
├── static/                  # CSS files
├── templates/               # HTML templates
├── app.py                   # Flask application
├── train_model.py          # Training script
└── requirements.txt        # Dependencies
```

---

## 🎓 Categories

The system classifies resumes into 24 professional domains:

ACCOUNTANT, ADVOCATE, AGRICULTURE, APPAREL, ARTS, AUTOMOBILE, AVIATION, BANKING, BPO, BUSINESS-DEVELOPMENT, CHEF, CONSTRUCTION, CONSULTANT, DESIGNER, DIGITAL-MEDIA, ENGINEERING, FINANCE, FITNESS, HEALTHCARE, HR, INFORMATION-TECHNOLOGY, PUBLIC-RELATIONS, SALES, TEACHER

---

## 🔗 API Usage

### Endpoint
```
POST /api/analyze
Content-Type: application/json
```

### Request
```json
{
  "resume_text": "Your resume text here...",
  "job_description": "Job description here..."
}
```

### Response
```json
{
  "category": "INFORMATION-TECHNOLOGY",
  "match_score": 85.5,
  "skills": ["Python", "Machine Learning", "TensorFlow"],
  "email": "example@email.com"
}
```

---

## 👥 Team

**Developed By**:
- [Maitreya Pawar](https://www.linkedin.com/in/maitreya-pawar-146664296/)
- [Prakasita Mohanty](https://www.linkedin.com/in/prakasita-mohanty/)
- [Nehal Garg](https://www.linkedin.com/in/nehal-garg-87b976347/)
- Saniya Sawnt

---

## 📄 License

This project is licensed under the MIT License.

---

## 🙏 Acknowledgments

- **BERT**: Google's Bidirectional Encoder Representations from Transformers
- **Sentence-BERT**: Sentence embeddings using Siamese BERT networks
- **HuggingFace**: Transformers library and model hub
- **NLTK**: Natural Language Toolkit

---

## 📞 Support

For issues, questions, or contributions, please refer to the [PROJECT_DOCUMENTATION.md](PROJECT_DOCUMENTATION.md) file.

---

**© 2024 AI Resume Analyzer - Powered by BERT, Sentence-BERT, and NLP**
