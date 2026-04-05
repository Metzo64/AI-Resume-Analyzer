"""
Resume Classifier Module
Uses BERT-based deep learning model to classify resumes into categories
"""

import os
import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from transformers import BertTokenizer, BertForSequenceClassification
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
import pickle


class ResumeDataset(Dataset):
    """Custom Dataset for resume classification"""
    
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }


class ResumeClassifier:
    def __init__(self, model_path='models'):
        """Initialize classifier with model path"""
        self.model_path = model_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = None
        self.label_encoder = None
        
        # Create models directory if it doesn't exist
        os.makedirs(model_path, exist_ok=True)
    
    def train(self, csv_path, epochs=3, batch_size=8, learning_rate=2e-5):
        """
        Train BERT model on resume dataset
        
        Args:
            csv_path (str): Path to Resume.csv
            epochs (int): Number of training epochs
            batch_size (int): Batch size for training
            learning_rate (float): Learning rate
        """
        print("Loading dataset...")
        df = pd.read_csv(csv_path)
        
        # Use Resume_str column for text and Category for labels
        texts = df['Resume_str'].values
        categories = df['Category'].values
        
        # Encode labels
        self.label_encoder = LabelEncoder()
        labels = self.label_encoder.fit_transform(categories)
        
        # Save label encoder
        with open(os.path.join(self.model_path, 'label_encoder.pkl'), 'wb') as f:
            pickle.dump(self.label_encoder, f)
        
        print(f"Number of categories: {len(self.label_encoder.classes_)}")
        print(f"Categories: {self.label_encoder.classes_}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            texts, labels, test_size=0.2, random_state=42, stratify=labels
        )
        
        # Create datasets
        train_dataset = ResumeDataset(X_train, y_train, self.tokenizer)
        test_dataset = ResumeDataset(X_test, y_test, self.tokenizer)
        
        # Create dataloaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)
        
        # Initialize model
        num_labels = len(self.label_encoder.classes_)
        self.model = BertForSequenceClassification.from_pretrained(
            'bert-base-uncased',
            num_labels=num_labels
        )
        self.model.to(self.device)
        
        # Optimizer
        optimizer = AdamW(self.model.parameters(), lr=learning_rate)
        
        # Training loop
        print("Starting training...")
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            
            for batch_idx, batch in enumerate(train_loader):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)
                
                optimizer.zero_grad()
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                loss = outputs.loss
                total_loss += loss.item()
                
                loss.backward()
                optimizer.step()
                
                if (batch_idx + 1) % 10 == 0:
                    print(f"Epoch {epoch+1}/{epochs}, Batch {batch_idx+1}/{len(train_loader)}, Loss: {loss.item():.4f}")
            
            avg_loss = total_loss / len(train_loader)
            print(f"Epoch {epoch+1}/{epochs} completed. Average Loss: {avg_loss:.4f}")
            
            # Evaluate
            accuracy = self.evaluate(test_loader)
            print(f"Test Accuracy: {accuracy:.2f}%\n")
        
        # Save model
        self.save_model()
        print("Training completed and model saved!")
    
    def evaluate(self, test_loader):
        """Evaluate model accuracy"""
        self.model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)
                
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                predictions = torch.argmax(outputs.logits, dim=1)
                
                correct += (predictions == labels).sum().item()
                total += labels.size(0)
        
        accuracy = (correct / total) * 100
        return accuracy
    
    def save_model(self):
        """Save trained model"""
        model_save_path = os.path.join(self.model_path, 'resume_classifier')
        self.model.save_pretrained(model_save_path)
        self.tokenizer.save_pretrained(model_save_path)
        print(f"Model saved to {model_save_path}")
    
    def load_model(self):
        """Load trained model"""
        model_load_path = os.path.join(self.model_path, 'resume_classifier')
        
        if not os.path.exists(model_load_path):
            raise FileNotFoundError(f"Model not found at {model_load_path}. Please train the model first.")
        
        self.model = BertForSequenceClassification.from_pretrained(model_load_path)
        self.model.to(self.device)
        self.model.eval()
        
        # Load label encoder
        with open(os.path.join(self.model_path, 'label_encoder.pkl'), 'rb') as f:
            self.label_encoder = pickle.load(f)
        
        print("Model loaded successfully!")
    
    def predict(self, text):
        """
        Predict category for a given resume text
        
        Args:
            text (str): Resume text
            
        Returns:
            str: Predicted category
        """
        if self.model is None:
            self.load_model()
        
        self.model.eval()
        
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=512,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            prediction = torch.argmax(outputs.logits, dim=1).cpu().numpy()[0]
        
        category = self.label_encoder.inverse_transform([prediction])[0]
        return category


# Training script (run separately)
if __name__ == "__main__":
    classifier = ResumeClassifier()
    classifier.train('data/Resume.csv', epochs=2, batch_size=4)
