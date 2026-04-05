"""
Training Script for Resume Classifier
Run this script to train the BERT model on Resume.csv dataset
"""

import os
import sys
from modules.classifier import ResumeClassifier

def main():
    print("\n" + "="*70)
    print("🧠 AI Resume Analyzer - Model Training")
    print("="*70 + "\n")
    
    # Check if dataset exists
    dataset_path = 'data/Resume.csv'
    if not os.path.exists(dataset_path):
        print(f"❌ Error: Dataset not found at {dataset_path}")
        print("Please ensure Resume.csv is in the data/ directory")
        sys.exit(1)
    
    print(f"✓ Dataset found: {dataset_path}")
    
    # Initialize classifier
    print("\n📦 Initializing classifier...")
    classifier = ResumeClassifier(model_path='models')
    
    # Training configuration
    print("\n⚙️ Training Configuration:")
    print("  - Model: BERT (bert-base-uncased)")
    print("  - Epochs: 2")
    print("  - Batch Size: 4")
    print("  - Learning Rate: 2e-5")
    print("  - Train/Test Split: 80/20")
    
    # Check for GPU
    import torch
    device = "GPU" if torch.cuda.is_available() else "CPU"
    print(f"  - Device: {device}")
    
    if device == "CPU":
        print("\n⚠️  Warning: Training on CPU will be slow (30-60 minutes)")
        print("   Consider using a GPU for faster training")
    
    # Confirm training
    print("\n" + "="*70)
    response = input("Start training? (yes/no): ").strip().lower()
    
    if response not in ['yes', 'y']:
        print("Training cancelled.")
        sys.exit(0)
    
    print("\n🚀 Starting training...\n")
    print("="*70 + "\n")
    
    try:
        # Train the model
        classifier.train(
            csv_path=dataset_path,
            epochs=2,
            batch_size=4,
            learning_rate=2e-5
        )
        
        print("\n" + "="*70)
        print("✅ Training completed successfully!")
        print("="*70)
        print("\n📁 Model saved to: models/resume_classifier/")
        print("📁 Label encoder saved to: models/label_encoder.pkl")
        print("\n🎉 You can now run the application:")
        print("   python app.py")
        print("\n" + "="*70 + "\n")
        
    except Exception as e:
        print("\n" + "="*70)
        print(f"❌ Training failed with error:")
        print(f"   {str(e)}")
        print("="*70 + "\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
