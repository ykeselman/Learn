# %%
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

# Different regularization approaches
class RegularizedClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, regularization_type='none'):
        super(RegularizedClassifier, self).__init__()
        self.regularization_type = regularization_type
        
        if regularization_type == 'dropout':
            self.layers = nn.Sequential(
                nn.Linear(input_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(0.5),  # Drop 50% of neurons during training
                nn.Linear(hidden_size, hidden_size // 2),
                nn.ReLU(),
                nn.Dropout(0.3),  # Less dropout in later layers
                nn.Linear(hidden_size // 2, output_size)
            )
        
        elif regularization_type == 'batch_norm':
            self.layers = nn.Sequential(
                nn.Linear(input_size, hidden_size),
                nn.BatchNorm1d(hidden_size),  # Normalize layer inputs
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size // 2),
                nn.BatchNorm1d(hidden_size // 2),
                nn.ReLU(),
                nn.Linear(hidden_size // 2, output_size)
            )
        
        elif regularization_type == 'dropout_batch_norm':
            self.layers = nn.Sequential(
                nn.Linear(input_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(hidden_size, hidden_size // 2),
                nn.BatchNorm1d(hidden_size // 2),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_size // 2, output_size)
            )
        
        else:  # 'none' - no regularization
            self.layers = nn.Sequential(
                nn.Linear(input_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size // 2),
                nn.ReLU(),
                nn.Linear(hidden_size // 2, output_size)
            )
    
    def forward(self, x):
        return self.layers(x)

def train_with_regularization(X, y, reg_type='none', weight_decay=0.0, epochs=300):
    """Train model with different regularization techniques"""
    
    # Prepare data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    X_train = torch.FloatTensor(X_train)
    X_test = torch.FloatTensor(X_test)
    y_train = torch.LongTensor(y_train)
    y_test = torch.LongTensor(y_test)
    
    # Create model
    model = RegularizedClassifier(X.shape[1], 64, len(np.unique(y)), reg_type)
    criterion = nn.CrossEntropyLoss()
    
    # Weight decay is L2 regularization applied to optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=weight_decay)
    
    # Training loop
    train_losses = []
    test_losses = []
    train_accuracies = []
    test_accuracies = []
    
    for epoch in range(epochs):
        # Training
        model.train()
        optimizer.zero_grad()
        train_outputs = model(X_train)
        train_loss = criterion(train_outputs, y_train)
        
        # Add L1 regularization manually (if desired)
        l1_lambda = 0.001
        if reg_type == 'l1':
            l1_penalty = sum(param.abs().sum() for param in model.parameters())
            train_loss += l1_lambda * l1_penalty
        
        train_loss.backward()
        optimizer.step()
        
        # Evaluation every 50 epochs
        if epoch % 50 == 0:
            model.eval()
            with torch.no_grad():
                # Training metrics
                _, train_predicted = torch.max(train_outputs.data, 1)
                train_acc = (train_predicted == y_train).sum().item() / len(y_train)
                
                # Test metrics
                test_outputs = model(X_test)
                test_loss = criterion(test_outputs, y_test)
                _, test_predicted = torch.max(test_outputs.data, 1)
                test_acc = (test_predicted == y_test).sum().item() / len(y_test)
                
                train_losses.append(train_loss.item())
                test_losses.append(test_loss.item())
                train_accuracies.append(train_acc)
                test_accuracies.append(test_acc)
                
                print(f'Epoch {epoch:3d}: Train Loss: {train_loss.item():.4f}, '
                      f'Test Loss: {test_loss.item():.4f}, '
                      f'Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')
    
    # Final evaluation
    model.eval()
    with torch.no_grad():
        train_outputs = model(X_train)
        _, train_predicted = torch.max(train_outputs.data, 1)
        final_train_acc = (train_predicted == y_train).sum().item() / len(y_train)
        
        test_outputs = model(X_test)
        _, test_predicted = torch.max(test_outputs.data, 1)
        final_test_acc = (test_predicted == y_test).sum().item() / len(y_test)
    
    return final_train_acc, final_test_acc

# Load breast cancer dataset (prone to overfitting due to many features)
cancer = load_breast_cancer()
X, y = cancer.data, cancer.target

print("Comparing Regularization Techniques on Breast Cancer Dataset")
print("=" * 70)
print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features, {len(np.unique(y))} classes")
print()

# Test different regularization approaches
regularization_methods = [
    ('No Regularization', 'none', 0.0),
    ('L2 Regularization (Weight Decay)', 'none', 0.01),
    ('Dropout', 'dropout', 0.0),
    ('Batch Normalization', 'batch_norm', 0.0),
    ('Dropout + Batch Norm', 'dropout_batch_norm', 0.0),
    ('L1 Regularization', 'l1', 0.0),
    ('L2 + Dropout', 'dropout', 0.01),
]

results = []
for name, reg_type, weight_decay in regularization_methods:
    print(f"\n{name}:")
    print("-" * 40)
    train_acc, test_acc = train_with_regularization(X, y, reg_type, weight_decay)
    overfitting = train_acc - test_acc
    results.append((name, train_acc, test_acc, overfitting))
    print(f"Final - Train: {train_acc:.4f}, Test: {test_acc:.4f}, "
          f"Overfitting: {overfitting:.4f}")

# Summary
print(f"\n{'='*70}")
print("SUMMARY OF REGULARIZATION RESULTS")
print(f"{'='*70}")
print(f"{'Method':<25} {'Train':<8} {'Test':<8} {'Overfitting':<12}")
print("-" * 55)
for name, train_acc, test_acc, overfitting in results:
    print(f"{name:<25} {train_acc:<8.3f} {test_acc:<8.3f} {overfitting:<12.3f}")

print(f"\n{'='*70}")
print("REGULARIZATION TECHNIQUES EXPLAINED:")
print("• Dropout: Randomly sets neurons to 0 during training")
print("• L2 (Weight Decay): Penalizes large weights in loss function")
print("• L1: Promotes sparsity by penalizing absolute weight values")
print("• Batch Normalization: Normalizes inputs to each layer")
print("• Lower overfitting (train-test gap) is better")
