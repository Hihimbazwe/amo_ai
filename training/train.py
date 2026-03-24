import torch
import torch.nn as nn
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
import json
import os
import numpy as np

# ── 1. Model Architecture ──────────────────────
class AmoIntentClassifier(nn.Module):
    """
    A professional-grade transformer-based intent classifier.
    Uses GELU activation for smoother gradients and better convergence.
    """
    def __init__(self, input_dim, hidden_dim, num_classes):
        super(AmoIntentClassifier, self).__init__()
        
        # Entrance projection
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        
        # GELU: Modern alternative to ReLU
        self.gelu = nn.GELU()
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.2)
        
        # Hidden transformation
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        
        # Final classification layer
        self.fc3 = nn.Linear(hidden_dim // 2, num_classes)
        
    def forward(self, x):
        # Layer 1
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.dropout(x)
        
        # Layer 2
        x = self.fc2(x)
        x = self.gelu(x)
        
        # Output
        x = self.fc3(x)
        return x

# ── 2. Data Preparation ────────────────────────
def prepare_data():
    """
    Prepare synthetic training data for Amo Pay domain classification.
    0: In-Scope (Amo Pay domain)
    1: Out-of-Scope (General/Random)
    """
    in_scope = [
        "How do I send money to Rwanda?",
        "What are the fees for USD transfer?",
        "Become a merchant on Amo Pay",
        "My KYC was rejected why?",
        "supported currencies in the app",
        "how to exchange RWF to EUR",
        "is biometric login supported?",
        "contact customer support email",
        "how to pay REG bills",
        "utility payments in Rwanda"
    ]
    
    out_of_scope = [
        "What is the capital of France?",
        "Tell me a joke",
        "How to cook spaghetti?",
        "Who won the world cup in 2022?",
        "Write a poem about love",
        "What is the price of Bitcoin?",
        "How to build a website?",
        "Weather in Kigali today",
        "latest news in movies",
        "how to fix a bicycle"
    ]
    
    return in_scope, out_of_scope

# ── 3. Training Loop ───────────────────────────
def train_model():
    print("🚀 Initializing Amo Pay Intent Classifier training...")
    
    # Load embedding model
    model_name = 'all-MiniLM-L6-v2'
    embedder = SentenceTransformer(model_name)
    embed_dim = 384 # Size of all-MiniLM-L6-v2 embeddings
    
    # Prepare data
    in_scope, out_of_scope = prepare_data()
    texts = in_scope + out_of_scope
    labels = [0] * len(in_scope) + [1] * len(out_of_scope)
    
    # Encode texts
    print(f"📡 Encoding {len(texts)} samples...")
    embeddings = embedder.encode(texts)
    X = torch.tensor(embeddings, dtype=torch.float32)
    y = torch.tensor(labels, dtype=torch.long)
    
    # Initialize classifier
    classifier = AmoIntentClassifier(embed_dim, 128, 2)
    optimizer = torch.optim.Adam(classifier.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    # Simple training loop
    classifier.train()
    for epoch in range(50):
        optimizer.zero_grad()
        outputs = classifier(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/50], Loss: {loss.item():.4f}")
            
    # Save model
    if not os.path.exists('models'):
        os.makedirs('models')
        
    torch.save(classifier.state_dict(), 'models/intent_classifier.pth')
    print("✅ Model saved to models/intent_classifier.pth")
    
    # Save a small metadata file for the server to know class names
    with open('models/metadata.json', 'w') as f:
        json.dump({"classes": ["in_scope", "out_of_scope"], "embed_model": model_name}, f)

if __name__ == "__main__":
    try:
        import torch
        train_model()
    except ImportError:
        print("❌ torch not found. This script requires torch for training.")
        print("Falling back to mock training for demonstration...")
        if not os.path.exists('models'): os.makedirs('models')
        with open('models/metadata.json', 'w') as f:
            json.dump({"classes": ["in_scope", "out_of_scope"], "embed_model": "all-MiniLM-L6-v2", "mock": True}, f)
        print("✅ Mock metadata saved.")
