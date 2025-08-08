# %%
import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler

from abc import ABC, abstractmethod

# %%
class RegdClassifier(nn.Module, ABC):
    """Base class for regd classifer."""
    def __init__(self, input_size: int, hidden_size: int, output_size: int, **kwargs):
        super(RegdClassifier, self).__init__()
        self.layers = self._get_layers(input_size, hidden_size, output_size, **kwargs)
        self.criterion = nn.CrossEntropyLoss()

    def my_loss(self, outputs: torch.Tensor, actual: torch.Tensor, **kwargs):
        out = self.criterion(outputs, actual)
        if "l1_lambda" in kwargs:
            l1_lambda = kwargs['l1_lambda']
            l1_penalty = sum(param.abs().sum() for param in self.parameters())
            out += l1_lambda * l1_penalty
        return out

    @abstractmethod
    def _get_layers(self, input_size, hidden_size, output_size, **kwargs):
        pass

    def forward(self, X):
        return self.layers(X)

# %% dropout
class DropoutRegdClassifer(RegdClassifier):
    def _get_layers(self, input_size, hidden_size, output_size, **kwargs):
        dropout = kwargs['dropout']
        return nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),      # dropout rate
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout/2),    # Less dropout in later layers
            nn.Linear(hidden_size // 2, output_size)
        )

# %% batch
class BatchNormRegdClassifier(RegdClassifier):
    def _get_layers(self, input_size, hidden_size, output_size, **kwargs):
        return nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.BatchNorm1d(hidden_size),  # Normalize layer inputs
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.BatchNorm1d(hidden_size // 2),  # Normalize layer inputs
            nn.ReLU(),
            nn.Linear(hidden_size // 2, output_size)
        )

# %% dropout and batch
class DropoutBatchNormRegdClassifier(RegdClassifier):
    def _get_layers(self, input_size, hidden_size, output_size, **kwargs):
        dropout = kwargs['dropout']
        return nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.BatchNorm1d(hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout/2),
            nn.Linear(hidden_size // 2, output_size)
        )

# %% None
class NonRegdClassifier(RegdClassifier):
    def _get_layers(self, input_size, hidden_size, output_size, **kwargs):
        return nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, output_size)
        )

# %%
REGS = {
    'dropout': DropoutRegdClassifer,
    'batch': BatchNormRegdClassifier,
    'dropout_batch': DropoutBatchNormRegdClassifier,
    'none': NonRegdClassifier,
}

def get_regd_classifier(reg_type: str, input_size: int, hidden_size: int, output_size: int, **kwargs) -> RegdClassifier:
    cls = REGS.get(reg_type)
    return cls(input_size=input_size, hidden_size=hidden_size, output_size=output_size, **kwargs)

# %% running the harness
import numpy as np
import pandas as pd

def train_with_regularization(X: np.array, y: np.array, reg_type: str, epochs=200, **kwargs) -> pd.DataFrame:
    # Prepare data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    X_train = torch.FloatTensor(X_train)
    X_test = torch.FloatTensor(X_test)
    y_train = torch.LongTensor(y_train)
    y_test = torch.LongTensor(y_test)

    model = get_regd_classifier(reg_type, X.shape[1], 64, len(np.unique(y)), **kwargs)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=kwargs.get('weight_decay', 0.0))

    fdf = []
    for epoch in range(epochs):
        # Training
        model.train()
        optimizer.zero_grad()
        train_outputs = model(X_train)
        train_loss = model.my_loss(train_outputs, y_train, **kwargs)
        train_loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            # Training metrics
            _, train_predicted = torch.max(train_outputs.data, 1)
            train_acc = (train_predicted == y_train).sum().item() / len(y_train)

            # Test metrics
            test_outputs = model(X_test)
            test_loss = model.my_loss(test_outputs, y_test, **kwargs)
            _, test_predicted = torch.max(test_outputs.data, 1)
            test_acc = (test_predicted == y_test).sum().item() / len(y_test)

            dd = {
                'epoch': epoch,
                'train_loss': train_loss.item(),
                'test_loss': test_loss.item(),
                'train_acc': train_acc,
                'test_acc': test_acc,
            }
            dd.update(kwargs)
            fdf.append(dd)

    df = pd.DataFrame(fdf)
    df['diff'] = df['train_acc'] - df['test_acc']
    return df

# %%
cancer = load_breast_cancer()
X, y = cancer.data, cancer.target

# %%

# works quite well as is -- the diff is 0.004
# df = train_with_regularization(X, y, 'dropout', dropout=0.6, weight_decay=0.01)

# worse perf than dropout
# df = train_with_regularization(X, y, 'batch', weight_decay=0.01)

# second best -- batch and dropout
# df = train_with_regularization(X, y, 'dropout_batch', dropout=0.8, weight_decay=0.01)

# simple network with regularization -- the diff is 0.005, similar to dropout
df = train_with_regularization(X, y, 'none', l1_lambda=0.005)

print(df)

# %%
p = 0.9999999

from math import log
-log(p), -log(1-p)
# %%
