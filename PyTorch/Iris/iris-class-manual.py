#  %% import modules
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

#  %% load data
iris = load_iris()
X, y = iris.data, iris.target

# %% split into train, test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=99)
dim_in = X_train.shape[1]
dim_out = len(set(y_test))
dim_in, dim_out

# %% convert into tensors
X_train = torch.FloatTensor(X_train)
X_test = torch.FloatTensor(X_test)
y_train = torch.LongTensor(y_train)
y_test = torch.LongTensor(y_test)

# %% set up main architecture class

class IrisClassifier(nn.Module):
    def __init__(self):
        super(IrisClassifier, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(dim_in, 4*dim_in),
            nn.ReLU(),
            nn.Linear(4*dim_in, 2*dim_in),
            nn.ReLU(),
            nn.Linear(2*dim_in, dim_out)
        )

    def forward(self, x):
        return self.layers(x)

# %% training machinery
model = IrisClassifier()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# %% training loop
epochs = 100
losses = []
for epoch in range(epochs):
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    losses.append(loss)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# %% convenience
def label(x):
    """Map num val to label"""
    return iris.target_names[x]

# %% evaluate the model on the test data
model.eval()
with torch.no_grad():
    test_out = model(X_test)
    _, predicted = torch.max(test_out.data, 1)
    df = {'actual': label(y_test.numpy()), 'pred': label(predicted.numpy())}
pd.crosstab(df['actual'], df['pred'])
