"""
Fair sequence-aware baseline, evaluated under the IDENTICAL chronological
protocol as the XGBoost model (not the leaky random-split numbers cited from
Tuor et al. / Yuan et al. in the paper's Table 4).

WHY THIS EXISTS: the paper's Table 4 compares its chronological F1 (48.41%)
against OTHER PAPERS' random-split LSTM numbers (82-85%), which the paper
itself admits is an unfair comparison (different protocol). A reviewer can
reasonably ask "so what does an LSTM get under YOUR protocol?" -- this
answers that question directly instead of relying on cited numbers.

IMPLEMENTATION NOTE: this sandbox cannot install PyTorch (no disk headroom,
and download.pytorch.org isn't reachable from here), so this uses a
lag-window feature representation (last `window` weeks concatenated) fed to
an MLPClassifier as a lightweight sequence-aware stand-in. This still
captures short-range temporal dependency the way an LSTM would, just without
the recurrent architecture.

TO SWAP IN A REAL LSTM later (recommended once you're off this sandbox):
    import torch, torch.nn as nn
    class LSTMClassifier(nn.Module):
        def __init__(self, n_features, hidden=32):
            super().__init__()
            self.lstm = nn.LSTM(n_features, hidden, batch_first=True)
            self.fc = nn.Linear(hidden, 1)
        def forward(self, x):
            out, _ = self.lstm(x)
            return torch.sigmoid(self.fc(out[:, -1, :]))
Feed it the same build_lag_windows() output reshaped to (batch, window, n_features)
instead of flattened, train with BCEWithLogitsLoss + class weighting, and
evaluate with the same evaluate.py harness so the comparison stays apples-to-apples.
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

def build_lag_windows(df, feature_cols, window=3, id_col="user_id", week_col="week"):
    """
    For each (user, week) with >= `window` weeks of history, concatenate the
    previous `window` weeks of features into one row. Returns a new dataframe
    aligned with the label of the CURRENT (most recent) week only -- so the
    model still only "knows" what happened up to and including week t, exactly
    matching the leak-free discipline of the rest of the pipeline.
    """
    df = df.sort_values([id_col, week_col]).reset_index(drop=True)
    rows, labels, meta = [], [], []
    for u, g in df.groupby(id_col, sort=False):
        g = g.reset_index(drop=True)
        vals = g[feature_cols].values
        for t in range(window - 1, len(g)):
            window_slice = vals[t - window + 1: t + 1].flatten()
            rows.append(window_slice)
            labels.append(g.loc[t, "label"])
            meta.append((u, g.loc[t, week_col]))
    X = np.array(rows)
    y = np.array(labels)
    meta_df = pd.DataFrame(meta, columns=[id_col, week_col])
    return X, y, meta_df

class LSTMClassifier(nn.Module):
    def __init__(self, n_features, hidden=32):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden, batch_first=True)
        self.fc = nn.Linear(hidden, 1)
        
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

class LSTMWrapper:
    def __init__(self, n_features, window_size=3, seed=42):
        torch.manual_seed(seed)
        self.window_size = window_size
        self.n_features = n_features // window_size
        self.model = LSTMClassifier(self.n_features, hidden=32)
        
    def fit(self, X, y, X_val=None, y_val=None, epochs=50, batch_size=32, patience=5):
        X_tensor = torch.tensor(X, dtype=torch.float32).view(-1, self.window_size, self.n_features)
        y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1)
        
        if X_val is not None and y_val is not None:
            X_val_tensor = torch.tensor(X_val, dtype=torch.float32).view(-1, self.window_size, self.n_features)
            y_val_tensor = torch.tensor(y_val, dtype=torch.float32).view(-1, 1)
        
        pos_weight = (len(y) - y.sum()) / max(1, y.sum())
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], dtype=torch.float32))
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        
        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        best_loss = float('inf')
        patience_counter = 0
        best_state = None

        for epoch in range(epochs):
            self.model.train()
            for batch_X, batch_y in loader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
            
            if X_val is not None and y_val is not None:
                self.model.eval()
                with torch.no_grad():
                    val_outputs = self.model(X_val_tensor)
                    val_loss = criterion(val_outputs, y_val_tensor).item()
                
                if val_loss < best_loss:
                    best_loss = val_loss
                    patience_counter = 0
                    best_state = self.model.state_dict().copy()
                else:
                    patience_counter += 1
                    
                if patience_counter >= patience:
                    break
                    
        if best_state is not None:
            self.model.load_state_dict(best_state)
                
    def predict_proba(self, X):
        self.model.eval()
        X_tensor = torch.tensor(X, dtype=torch.float32).view(-1, self.window_size, self.n_features)
        with torch.no_grad():
            outputs = self.model(X_tensor)
            probs = torch.sigmoid(outputs).numpy()
        # Returns (n_samples, 2) array so [:, 1] gets the positive class probabilities
        result = np.zeros((len(X), 2))
        result[:, 1] = probs.ravel()
        result[:, 0] = 1 - probs.ravel()
        return result

def fit_sequence_baseline(X_train, y_train, meta_train, seed=42):
    n_features_flat = X_train.shape[1]
    clf = LSTMWrapper(n_features_flat, window_size=3, seed=seed)
    
    weeks = sorted(meta_train["week"].unique())
    if len(weeks) >= 2:
        val_weeks = weeks[int(len(weeks) * 0.85):]
        val_mask = meta_train["week"].isin(val_weeks).values
        train_mask = ~val_mask
        clf.fit(X_train[train_mask], y_train[train_mask], 
                X_val=X_train[val_mask], y_val=y_train[val_mask], epochs=50)
    else:
        clf.fit(X_train, y_train, epochs=20)
        
    return clf
