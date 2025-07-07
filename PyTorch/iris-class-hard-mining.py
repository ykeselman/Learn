# %%

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
# from torch.utils.data import DataLoader, TensorDataset
import random

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# Load and preprocess Iris dataset
iris = load_iris()
X, y = iris.data, iris.target

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Convert to PyTorch tensors
X_train_tensor = torch.FloatTensor(X_train)
y_train_tensor = torch.LongTensor(y_train)
X_test_tensor = torch.FloatTensor(X_test)
y_test_tensor = torch.LongTensor(y_test)

print(f"Training set size: {len(X_train)}")
print(f"Test set size: {len(X_test)}")

class IrisNet(nn.Module):
    def __init__(self, input_size=4, hidden1_size=16, hidden2_size=8, num_classes=3):
        super(IrisNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden1_size)
        self.fc2 = nn.Linear(hidden1_size, hidden2_size)
        self.fc3 = nn.Linear(hidden2_size, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

def get_hard_examples(model, X, y, hard_ratio=0.5):
    """
    Select hard examples based on highest loss values
    """
    model.eval()
    with torch.no_grad():
        outputs = model(X)
        criterion = nn.CrossEntropyLoss(reduction='none')  # Don't reduce to get per-sample loss
        losses = criterion(outputs, y)
    
    # Get indices of hardest examples
    num_hard = int(len(X) * hard_ratio)
    _, hard_indices = torch.topk(losses, num_hard)
    
    return hard_indices

def train_model(model, X_train, y_train, epochs=200, batch_size=16, use_hard_mining=False, hard_ratio=0.5):
    """
    Train model with or without hard example mining
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    losses = []
    accuracies = []
    
    for epoch in range(epochs):
        model.train()
        
        if use_hard_mining and epoch > 10:  # Start hard mining after a few epochs
            # Get hard examples
            hard_indices = get_hard_examples(model, X_train, y_train, hard_ratio)
            
            # Create mini-batches with higher probability of hard examples
            if len(hard_indices) > 0:
                # Mix hard examples with some random examples
                num_random = batch_size - min(batch_size // 2, len(hard_indices))
                random_indices = torch.randperm(len(X_train))[:num_random]
                
                # Combine hard and random indices
                if len(hard_indices) >= batch_size // 2:
                    selected_hard = hard_indices[:batch_size // 2]
                else:
                    selected_hard = hard_indices
                
                batch_indices = torch.cat([selected_hard, random_indices[:batch_size - len(selected_hard)]])
            else:
                batch_indices = torch.randperm(len(X_train))[:batch_size]
        else:
            # Standard random sampling
            batch_indices = torch.randperm(len(X_train))[:batch_size]
        
        # Get batch data
        X_batch = X_train[batch_indices]
        y_batch = y_train[batch_indices]
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Calculate accuracy on full training set every 10 epochs
        if epoch % 10 == 0:
            model.eval()
            with torch.no_grad():
                train_outputs = model(X_train)
                _, predicted = torch.max(train_outputs.data, 1)
                accuracy = (predicted == y_train).float().mean().item()
                accuracies.append(accuracy)
                losses.append(loss.item())
                
                if epoch % 50 == 0:
                    hard_status = "with Hard Mining" if use_hard_mining else "Standard"
                    print(f'Epoch [{epoch}/{epochs}] {hard_status} - Loss: {loss.item():.4f}, Accuracy: {accuracy:.4f}')
    
    return losses, accuracies

# Train models with and without hard mining
print("Training model with standard random sampling...")
model_standard = IrisNet()
losses_standard, acc_standard = train_model(
    model_standard, X_train_tensor, y_train_tensor, 
    epochs=200, use_hard_mining=False
)

print("\nTraining model with hard example mining...")
model_hard = IrisNet()
losses_hard, acc_hard = train_model(
    model_hard, X_train_tensor, y_train_tensor, 
    epochs=200, use_hard_mining=True, hard_ratio=0.8
)

# Evaluate both models on test set
def evaluate_model(model, X_test, y_test):
    model.eval()
    with torch.no_grad():
        outputs = model(X_test)
        _, predicted = torch.max(outputs.data, 1)
        accuracy = (predicted == y_test).float().mean().item()
        return accuracy

test_acc_standard = evaluate_model(model_standard, X_test_tensor, y_test_tensor)
test_acc_hard = evaluate_model(model_hard, X_test_tensor, y_test_tensor)

print(f"\n=== Final Results ===")
print(f"Standard Training - Test Accuracy: {test_acc_standard:.4f}")
print(f"Hard Mining Training - Test Accuracy: {test_acc_hard:.4f}")

# Plot training curves
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
epochs_plot = range(0, 200, 10)
plt.plot(epochs_plot, losses_standard, 'b-', label='Standard', linewidth=2)
plt.plot(epochs_plot, losses_hard, 'r-', label='Hard Mining', linewidth=2)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Comparison')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 3, 2)
plt.plot(epochs_plot, acc_standard, 'b-', label='Standard', linewidth=2)
plt.plot(epochs_plot, acc_hard, 'r-', label='Hard Mining', linewidth=2)
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training Accuracy Comparison')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 3, 3)
methods = ['Standard', 'Hard Mining']
test_accs = [test_acc_standard, test_acc_hard]
colors = ['blue', 'red']
plt.bar(methods, test_accs, color=colors, alpha=0.7)
plt.ylabel('Test Accuracy')
plt.title('Final Test Accuracy')
plt.ylim(0, 1)
for i, acc in enumerate(test_accs):
    plt.text(i, acc + 0.01, f'{acc:.3f}', ha='center', fontweight='bold')

plt.tight_layout()
plt.show()

# Analyze hard examples during training
print(f"\n=== Analysis of Hard Examples ===")
model_hard.eval()
with torch.no_grad():
    outputs = model_hard(X_train_tensor)
    criterion = nn.CrossEntropyLoss(reduction='none')
    losses = criterion(outputs, y_train_tensor)
    
    # Get hardest examples
    hard_indices = get_hard_examples(model_hard, X_train_tensor, y_train_tensor, hard_ratio=0.3)
    
    print(f"Top 5 hardest examples (indices): {hard_indices[:5].tolist()}")
    print(f"Their losses: {losses[hard_indices[:5]].tolist()}")
    print(f"True labels: {y_train_tensor[hard_indices[:5]].tolist()}")
    print(f"Predicted classes: {torch.argmax(outputs[hard_indices[:5]], dim=1).tolist()}")
