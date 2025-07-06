# %%
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# %%
# Load and prepare data
iris = load_iris()
X, y = iris.data, iris.target

# %%
# Split and scale data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# %%
# Convert to PyTorch tensors
X_train = torch.FloatTensor(X_train)
X_test = torch.FloatTensor(X_test)
y_train = torch.LongTensor(y_train)
y_test = torch.LongTensor(y_test)

# %%
class IrisClassifier(nn.Module):
    def __init__(self, architecture='default'):
        super(IrisClassifier, self).__init__()
        
        if architecture == 'minimal':
            # Single hidden layer - simplest approach
            self.layers = nn.Sequential(
                nn.Linear(4, 8),
                nn.ReLU(),
                nn.Linear(8, 3)
            )
        elif architecture == 'wide':
            # Wider single layer
            self.layers = nn.Sequential(
                nn.Linear(4, 32),
                nn.ReLU(),
                nn.Linear(32, 3)
            )
        elif architecture == 'deep':
            # More layers, smaller width
            self.layers = nn.Sequential(
                nn.Linear(4, 8),
                nn.ReLU(),
                nn.Linear(8, 8),
                nn.ReLU(),
                nn.Linear(8, 8),
                nn.ReLU(),
                nn.Linear(8, 3)
            )
        else:  # default
            # Original pyramid design
            self.layers = nn.Sequential(
                nn.Linear(4, 16),
                nn.ReLU(),
                nn.Linear(16, 8),
                nn.ReLU(),
                nn.Linear(8, 3)
            )
        
    def forward(self, x):
        return self.layers(x)

# Initialize model, loss function, and optimizer
model = IrisClassifier(architecture='deep')
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# %%
# Training loop
epochs = 100
for epoch in range(epochs):
    # Forward pass
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    
    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 20 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

# %%
# Evaluation
model.eval()
with torch.no_grad():
    test_outputs = model(X_test)
    _, predicted = torch.max(test_outputs.data, 1)
    accuracy = (predicted == y_test).sum().item() / len(y_test)
    print(f'\nTest Accuracy: {accuracy:.4f}')
    
    # Show some predictions
    print("\nSample predictions:")
    for i in range(min(5, len(y_test))):
        print(f"Actual: {iris.target_names[y_test[i]]}, "
              f"Predicted: {iris.target_names[predicted[i]]}")
