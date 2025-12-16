# src/mlp_torch.py
"""PyTorch implementation of the MLP classifier for GPU acceleration."""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import StandardScaler

# Check device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PyTorchMLP(nn.Module):
    def __init__(self, input_dim: int):
        super(PyTorchMLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        """Return logits (no sigmoid) for BCEWithLogitsLoss"""
        return self.layers(x)
    
    def predict_proba(self, x):
        """Return probabilities (with sigmoid) for predictions"""
        return torch.sigmoid(self.layers(x))

def fit_predict_torch(X_train, X_test, y_train, epochs=20, batch_size=1024):
    """Train PyTorch MLP on GPU and return predictions."""
    # Normalize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    input_dim = X_train_scaled.shape[1]
    model = PyTorchMLP(input_dim).to(DEVICE)
    
    # Prepare data
    X_tr_t = torch.tensor(X_train_scaled, dtype=torch.float32)
    y_tr_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    
    dataset = TensorDataset(X_tr_t, y_tr_t)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Class weights for imbalanced data
    pos_weight = torch.tensor([9.0]).to(DEVICE)  # Weight attack class higher
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    print(f"[PyTorch MLP] Training on {DEVICE} for {epochs} epochs...")
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for X_b, y_b in loader:
            X_b, y_b = X_b.to(DEVICE), y_b.to(DEVICE)
            
            optimizer.zero_grad()
            logits = model(X_b)
            loss = criterion(logits, y_b)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"  Epoch {epoch+1}/{epochs} - Loss: {epoch_loss/len(loader):.4f}")
            
    # Predict
    model.eval()
    X_te_t = torch.tensor(X_test_scaled, dtype=torch.float32).to(DEVICE)
    with torch.no_grad():
        probs = model.predict_proba(X_te_t).cpu().numpy()
        
    return (probs > 0.5).astype(int).flatten()

class TorchSklearnWrapper(BaseEstimator, ClassifierMixin):
    """Wrapper to make PyTorch model compatible with sklearn VotingClassifier."""
    
    def __init__(self, input_dim=None, epochs=20, batch_size=1024):
        self.input_dim = input_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = None
        self.scaler = None
        self.classes_ = np.array([0, 1])

    @property
    def _estimator_type(self):
        return "classifier"
        
    def fit(self, X, y):
        # Normalize features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        self.input_dim = X_scaled.shape[1]
        self.model = PyTorchMLP(self.input_dim).to(DEVICE)
        
        X_t = torch.tensor(X_scaled, dtype=torch.float32)
        y_t = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
        dataset = TensorDataset(X_t, y_t)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        pos_weight = torch.tensor([9.0]).to(DEVICE)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        
        print(f"[Ensemble MLP] Training on {DEVICE} for {self.epochs} epochs...")
        self.model.train()
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            for X_b, y_b in loader:
                X_b, y_b = X_b.to(DEVICE), y_b.to(DEVICE)
                optimizer.zero_grad()
                logits = self.model(X_b)
                loss = criterion(logits, y_b)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            print(f"  Epoch {epoch+1}/{self.epochs} - Loss: {epoch_loss/len(loader):.4f}")
        return self

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] > 0.5).astype(int)

    def predict_proba(self, X):
        X_scaled = self.scaler.transform(X)
        self.model.eval()
        X_t = torch.tensor(X_scaled, dtype=torch.float32).to(DEVICE)
        with torch.no_grad():
            pos_probs = self.model.predict_proba(X_t).cpu().numpy().flatten()
        return np.vstack([1 - pos_probs, pos_probs]).T
