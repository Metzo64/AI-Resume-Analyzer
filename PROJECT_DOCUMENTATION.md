# 🎓 AI Resume Analyzer - Complete Project Documentation

## 📋 Table of Contents
1. [Project Overview](#project-overview)
2. [System Architecture](#system-architecture)
3. [Technology Stack](#technology-stack)
4. [Implementation Details](#implementation-details)
5. [Training Process](#training-process)
6. [Installation & Setup](#installation--setup)
7. [Usage Guide](#usage-guide)
8. [API Documentation](#api-documentation)
9. [Design & UI](#design--ui)
10. [Team](#team)

---

## 🎯 Project Overview

### Purpose
An intelligent resume analysis system that uses deep learning and NLP to:
- Parse and extract information from resumes
- Classify resumes into 24 professional categories
- Match resumes with job descriptions using semantic similarity
- Provide actionable insights and recommendations

### Key Features
- **Resume Parsing**: Extracts skills, email, and key information using NLP
- **AI Classification**: BERT-based deep learning model for categorization
- **Job Matching**: Sentence-BERT for semantic similarity scoring (0-100%)
- **Web Interface**: Professional, aesthetic UI with glassmorphism design
- **REST API**: JSON-based API for programmatic access

### Dataset
- **Resume Dataset**: 2,484 resumes across 24 job categories
- **Job Description Dataset**: Real job postings with detailed descriptions
- **Categories**: ACCOUNTANT, ADVOCATE, AGRICULTURE, APPAREL, ARTS, AUTOMOBILE, AVIATION, BANKING, BPO, BUSINESS-DEVELOPMENT, CHEF, CONSTRUCTION, CONSULTANT, DESIGNER, DIGITAL-MEDIA, ENGINEERING, FINANCE, FITNESS, HEALTHCARE, HR, INFORMATION-TECHNOLOGY, PUBLIC-RELATIONS, SALES, TEACHER

---

## 🏗️ System Architecture

### High-Level Architecture
```
User Interface (Web Browser)
         ↓
Flask Backend (app.py)
         ↓
    ┌────┴────┬────────┬────────┐
    ↓         ↓        ↓        ↓
Parser   Classifier  Matcher  Database
(NLTK)    (BERT)   (S-BERT)   (CSV)
```

### Data Flow
```
1. User uploads resume → Flask receives file
2. Parser extracts information → Skills, email, cleaned text
3. Classifier predicts category → BERT model inference
4. Matcher computes similarity → Sentence-BERT embeddings
5. Results aggregated → Displayed in UI
```

### Component Interaction
```
┌─────────────────────────────────────────────┐
│           Flask Application                 │
│  ┌─────────────────────────────────────┐   │
│  │  Routes:                            │   │
│  │  - / (home)                         │   │
│  │  - /analyze (process)               │   │
│  │  - /api/analyze (REST API)          │   │
│  │  - /health (status check)           │   │
│  └─────────────────────────────────────┘   │
└─────────────────────────────────────────────┘
         ↓           ↓           ↓
┌──────────────┐ ┌──────────┐ ┌──────────┐
│   Parser     │ │Classifier│ │ Matcher  │
│              │ │          │ │          │
│ - Clean text │ │- BERT    │ │- S-BERT  │
│ - Extract    │ │- Predict │ │- Cosine  │
│   skills     │ │  category│ │  similarity
│ - Find email │ │          │ │          │
└──────────────┘ └──────────┘ └──────────┘
```

---

## 💻 Technology Stack

### Backend
- **Framework**: Flask 2.3.2
- **Language**: Python 3.10+
- **Deep Learning**: PyTorch 2.4.1 (CPU)
- **NLP**: NLTK 3.8.1
- **Transformers**: HuggingFace Transformers 5.5.0
- **Embeddings**: Sentence-Transformers 5.3.0

### Machine Learning Models
- **Classification**: BERT (bert-base-uncased)
  - 12 transformer layers
  - 110M parameters
  - Fine-tuned on resume dataset
- **Matching**: Sentence-BERT (all-MiniLM-L6-v2)
  - 384-dimensional embeddings
  - Optimized for semantic similarity

### Data Processing
- **Pandas**: 2.0.3 (data manipulation)
- **NumPy**: 1.24.3 (numerical operations)
- **Scikit-learn**: 1.3.0 (preprocessing, evaluation)

### Frontend
- **Templates**: Jinja2
- **Styling**: Custom CSS3 with glassmorphism
- **JavaScript**: Vanilla JS for interactions
- **Design**: Purple gradient theme with modern aesthetics

---

## 🔧 Implementation Details

### Module 1: Resume Parser (`modules/parser.py`)

#### Purpose
Extract structured information from unstructured resume text.

#### Key Functions
```python
class ResumeParser:
    def __init__(self):
        # Initialize with 50+ predefined skill keywords
        # Load NLTK stopwords
    
    def extract_email(self, text):
        # Regex pattern: r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        # Returns: First email found or None
    
    def extract_skills(self, text):
        # Match against predefined skill keywords
        # Uses word boundaries for accurate matching
        # Returns: List of unique skills found
    
    def clean_text(self, text):
        # 1. Convert to lowercase
        # 2. Remove special characters and digits
        # 3. Remove extra whitespace
        # 4. Tokenize using NLTK
        # 5. Remove stopwords
        # Returns: Cleaned text string
    
    def parse_resume(self, resume_text):
        # Main function that calls all above methods
        # Returns: Dictionary with email, skills, cleaned_text, original_text
```

#### Skill Categories
- **Programming**: Python, Java, JavaScript, C++, C#, Ruby, PHP, Swift, Kotlin
- **Frameworks**: React, Angular, Vue, Django, Flask, Spring, Express
- **Databases**: SQL, MySQL, PostgreSQL, MongoDB, Oracle, Redis
- **Cloud**: AWS, Azure, GCP, Docker, Kubernetes
- **ML/AI**: Machine Learning, Deep Learning, TensorFlow, PyTorch, NLP
- **Soft Skills**: Leadership, Communication, Teamwork, Problem Solving

#### Text Cleaning Process
```
Raw Text
    ↓
Lowercase conversion
    ↓
Remove special characters (regex: [^a-zA-Z\s])
    ↓
Remove extra whitespace
    ↓
Tokenization (NLTK word_tokenize)
    ↓
Stopword removal (NLTK stopwords)
    ↓
Filter short words (length > 2)
    ↓
Cleaned Text
```

---

### Module 2: Resume Classifier (`modules/classifier.py`)

#### Purpose
Classify resumes into 24 professional categories using BERT.

#### Architecture
```python
class ResumeDataset(Dataset):
    # Custom PyTorch Dataset
    # Tokenizes text with BERT tokenizer
    # Max length: 512 tokens
    # Padding: max_length
    # Returns: input_ids, attention_mask, label

class ResumeClassifier:
    def __init__(self):
        # Device: CPU or CUDA
        # Tokenizer: BertTokenizer.from_pretrained('bert-base-uncased')
        # Model: BertForSequenceClassification
        # Label Encoder: Scikit-learn LabelEncoder
    
    def train(self, csv_path, epochs=3, batch_size=8, learning_rate=2e-5):
        # 1. Load Resume.csv
        # 2. Encode labels
        # 3. Train-test split (80-20)
        # 4. Create PyTorch DataLoaders
        # 5. Initialize BERT model
        # 6. Training loop with AdamW optimizer
        # 7. Evaluate on test set
        # 8. Save model and label encoder
    
    def predict(self, text):
        # 1. Load model if not loaded
        # 2. Tokenize input text
        # 3. Forward pass through BERT
        # 4. Get prediction from logits
        # 5. Decode label
        # Returns: Category name
```

#### Training Configuration
```
Model: bert-base-uncased
Epochs: 2
Batch Size: 4
Learning Rate: 2e-5
Optimizer: AdamW
Loss Function: CrossEntropyLoss (built-in)
Train/Test Split: 80/20 with stratification
Max Sequence Length: 512 tokens
```

#### Model Performance
- **Expected Accuracy**: 85-90%
- **Training Time**: 30-60 minutes (CPU), 10-15 minutes (GPU)
- **Model Size**: ~400MB
- **Inference Time**: 1-2 seconds per resume

#### Training Process
```
1. Load Dataset (2,484 resumes)
    ↓
2. Preprocess
   - Extract text and labels
   - Encode labels (0-23)
   - Split train/test (80/20)
    ↓
3. Create DataLoaders
   - Tokenize with BERT tokenizer
   - Batch size: 4
   - Shuffle training data
    ↓
4. Initialize Model
   - Load bert-base-uncased
   - Add classification head (24 classes)
   - Move to device (CPU/GPU)
    ↓
5. Training Loop (2 epochs)
   For each epoch:
     For each batch:
       - Forward pass
       - Compute loss
       - Backward pass
       - Update weights
     - Evaluate on test set
     - Print accuracy
    ↓
6. Save Model
   - Save BERT model
   - Save tokenizer
   - Save label encoder
```

---

### Module 3: Resume Matcher (`modules/matcher.py`)

#### Purpose
Compute semantic similarity between resume and job description.

#### Architecture
```python
class ResumeMatcher:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        # Load Sentence-BERT model
        # Model size: ~90MB
        # Embedding dimension: 384
    
    def compute_similarity(self, resume_text, job_description):
        # 1. Encode resume → 384-dim vector
        # 2. Encode job description → 384-dim vector
        # 3. Compute cosine similarity
        # 4. Convert to percentage (0-100)
        # Returns: Match score
    
    def match_with_multiple_jobs(self, resume_text, job_descriptions):
        # Batch processing for multiple jobs
        # Returns: List of (index, score) tuples
    
    def get_top_matches(self, resume_text, job_descriptions, top_k=5):
        # Returns: Top K matching jobs
```

#### Similarity Computation
```
Resume Text                Job Description
     ↓                            ↓
Sentence-BERT Encoder      Sentence-BERT Encoder
     ↓                            ↓
Embedding (384-dim)        Embedding (384-dim)
     ↓                            ↓
     └────────────┬───────────────┘
                  ↓
         Cosine Similarity
         cos(A,B) = A·B / (|A||B|)
                  ↓
         Score × 100 = Match %
```

#### Match Score Interpretation
- **80-100%**: Excellent Match - Strong alignment
- **60-79%**: Good Match - Relevant experience
- **40-59%**: Moderate Match - Some alignment
- **0-39%**: Low Match - Limited alignment

---

### Flask Application (`app.py`)

#### Routes
```python
@app.route('/')
def index():
    # Render home page
    # Load job titles from CSV
    # Return: index.html with job_titles

@app.route('/analyze', methods=['POST'])
def analyze():
    # 1. Receive file upload
    # 2. Validate file
    # 3. Parse resume
    # 4. Classify resume
    # 5. Match with job
    # 6. Return results
    # Return: result.html with results dict

@app.route('/api/analyze', methods=['POST'])
def api_analyze():
    # REST API endpoint
    # Accept: JSON with resume_text, job_description
    # Return: JSON with category, match_score, skills, email

@app.route('/health')
def health():
    # Health check endpoint
    # Return: JSON with status
```

#### Request Flow
```
1. User uploads file
    ↓
2. Flask receives POST to /analyze
    ↓
3. Validate file (exists, not empty, min 50 chars)
    ↓
4. Read file content (decode UTF-8)
    ↓
5. Parse resume (parser.parse_resume)
    ↓
6. Classify resume (classifier.predict)
    ↓
7. Get job description from CSV
    ↓
8. Compute match score (matcher.compute_similarity)
    ↓
9. Aggregate results
    ↓
10. Render result.html
```

#### Error Handling
- File not uploaded → Error message
- File too short → Minimum 50 characters required
- No job selected → Error message
- Classification fails → Show "Unable to classify"
- General errors → Catch and display user-friendly message

---

## 🎓 Training Process

### Step-by-Step Training

#### 1. Environment Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Download NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
```

#### 2. Data Preparation
```python
# Load dataset
df = pd.read_csv('data/Resume.csv')

# Extract features and labels
texts = df['Resume_str'].values  # Resume text
labels = df['Category'].values   # Job categories

# Encode labels
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    texts, encoded_labels, 
    test_size=0.2, 
    random_state=42, 
    stratify=encoded_labels
)
```

#### 3. Model Initialization
```python
# Load BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Create datasets
train_dataset = ResumeDataset(X_train, y_train, tokenizer)
test_dataset = ResumeDataset(X_test, y_test, tokenizer)

# Create dataloaders
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=4)

# Initialize model
model = BertForSequenceClassification.from_pretrained(
    'bert-base-uncased',
    num_labels=24  # 24 categories
)
model.to(device)

# Initialize optimizer
optimizer = AdamW(model.parameters(), lr=2e-5)
```

#### 4. Training Loop
```python
for epoch in range(2):  # 2 epochs
    model.train()
    total_loss = 0
    
    for batch_idx, batch in enumerate(train_loader):
        # Get batch data
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        # Compute loss
        loss = outputs.loss
        total_loss += loss.item()
        
        # Backward pass
        loss.backward()
        
        # Update weights
        optimizer.step()
        
        # Print progress every 10 batches
        if (batch_idx + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/2, Batch {batch_idx+1}/497, Loss: {loss.item():.4f}")
    
    # Evaluate
    accuracy = evaluate(model, test_loader)
    print(f"Epoch {epoch+1} - Accuracy: {accuracy:.2f}%")
```

#### 5. Model Saving
```python
# Save model
model.save_pretrained('models/resume_classifier')
tokenizer.save_pretrained('models/resume_classifier')

# Save label encoder
with open('models/label_encoder.pkl', 'wb') as f:
    pickle.dump(label_encoder, f)
```

### Training Metrics
```
Dataset: 2,484 resumes
Training Set: 1,987 resumes (80%)
Test Set: 497 resumes (20%)
Batches per Epoch: 497
Total Batches: 994 (2 epochs)

Expected Results:
Epoch 1: Loss ~2.5 → ~1.8, Accuracy ~82%
Epoch 2: Loss ~1.8 → ~1.2, Accuracy ~87%
```

---

## 📦 Installation & Setup

### Prerequisites
- Python 3.7 or higher
- pip package manager
- 2GB free disk space
- Internet connection (for model downloads)

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

**Packages installed**:
- flask==2.3.2
- pandas==2.0.3
- scikit-learn==1.3.0
- torch>=2.3.0
- transformers>=4.41.0
- sentence-transformers>=2.7.0
- nltk==3.8.1
- numpy==1.24.3
- Werkzeug==2.3.6

### Step 2: Verify Setup
```bash
python setup.py
```

**Checks**:
- Python version (3.7+)
- Dependencies installed
- Data files present
- Directories created
- NLTK data downloaded

### Step 3: Train Model
```bash
python train_model.py
```

**Process**:
1. Loads Resume.csv (2,484 resumes)
2. Trains BERT model (2 epochs)
3. Saves model to models/resume_classifier/
4. Time: 30-60 minutes (CPU)

### Step 4: Run Application
```bash
python app.py
```

**Access**: http://localhost:5000

---

## 📖 Usage Guide

### Web Interface

#### 1. Upload Resume
- Click upload area or drag and drop
- Supported formats: TXT, PDF, DOC, DOCX
- Minimum 50 characters required

#### 2. Select Job Role
- Choose from dropdown (real job titles)
- Required for match score calculation

#### 3. Analyze
- Click "Analyze Resume" button
- Processing takes 2-3 seconds

#### 4. View Results
- **Category**: AI-predicted job domain
- **Match Score**: 0-100% compatibility
- **Skills**: Extracted professional skills
- **Contact**: Email if found
- **Interpretation**: Detailed analysis
- **Recommendations**: Actionable tips

### API Usage

#### Endpoint
```
POST /api/analyze
Content-Type: application/json
```

#### Request
```json
{
  "resume_text": "Experienced Python developer with 5 years in ML...",
  "job_description": "Looking for Python developer with ML experience..."
}
```

#### Response
```json
{
  "category": "INFORMATION-TECHNOLOGY",
  "match_score": 85.5,
  "skills": ["Python", "Machine Learning", "TensorFlow"],
  "email": "john@example.com"
}
```

---

## 🎨 Design & UI

### Color Scheme
- **Primary**: Purple gradient (#667eea → #764ba2)
- **Accents**: Purple (#7c3aed), Blue (#0ea5e9)
- **Success**: Green (#10b981)
- **Background**: Gradient with radial overlays

### Design Features
- **Glassmorphism**: Semi-transparent cards with blur
- **Smooth Animations**: 0.3s transitions
- **Gradient Text**: For values and headings
- **Drop Shadows**: Colored shadows on icons
- **Rounded Corners**: 20-24px border radius

### Responsive Design
- Desktop: Full features
- Tablet: Optimized layout
- Mobile: Touch-friendly, stacked layout

---

## 👥 Team

**Developed By**:
- **Maitreya Pawar** - [LinkedIn](https://www.linkedin.com/in/maitreya-pawar-146664296/)
- **Prakasita Mohanty** - [LinkedIn](https://www.linkedin.com/in/prakasita-mohanty/)
- **Nehal Garg** - [LinkedIn](https://www.linkedin.com/in/nehal-garg-87b976347/)
- **Saniya Sawnt**

---

## 📊 Project Statistics

- **Total Files**: 20
- **Lines of Code**: ~2,000+
- **Documentation**: 100+ pages
- **Models**: 2 (BERT, Sentence-BERT)
- **Categories**: 24 job domains
- **Skills Detected**: 50+
- **Training Time**: 30-60 minutes
- **Inference Time**: 2-3 seconds

---

**© 2024 AI Resume Analyzer - Powered by BERT, Sentence-BERT, and NLP**
