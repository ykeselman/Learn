# %%

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_iris, load_wine, load_breast_cancer, make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Simple neural network class
class SimpleClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleClassifier, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, output_size)
        )
    
    def forward(self, x):
        return self.layers(x)

def train_and_evaluate(X, y, dataset_name, num_classes, epochs=200):
    """Train and evaluate a simple neural network on given dataset"""
    print(f"\n{'='*50}")
    print(f"Dataset: {dataset_name}")
    print(f"Features: {X.shape[1]}, Samples: {X.shape[0]}, Classes: {num_classes}")
    
    # Split and scale data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Convert to tensors
    X_train = torch.FloatTensor(X_train)
    X_test = torch.FloatTensor(X_test)
    y_train = torch.LongTensor(y_train)
    y_test = torch.LongTensor(y_test)
    
    # Create model
    # hidden_size = max(16, X.shape[1] * 4)  # Adaptive hidden size
    hidden_size = min(16, X.shape[1] * 4)  # Adaptive hidden size
    model = SimpleClassifier(X.shape[1], hidden_size, num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    # Training
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 50 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
    
    # Evaluation
    model.eval()
    with torch.no_grad():
        # Training accuracy
        train_outputs = model(X_train)
        _, train_predicted = torch.max(train_outputs.data, 1)
        train_accuracy = (train_predicted == y_train).sum().item() / len(y_train)
        
        # Test accuracy
        test_outputs = model(X_test)
        _, test_predicted = torch.max(test_outputs.data, 1)
        test_accuracy = (test_predicted == y_test).sum().item() / len(y_test)
        
        print(f'Training Accuracy: {train_accuracy:.4f}')
        print(f'Test Accuracy: {test_accuracy:.4f}')
        print(f'Architecture: {X.shape[1]} → {hidden_size} → {hidden_size//2} → {num_classes}')
        
        return train_accuracy, test_accuracy

# Test different datasets
datasets = []

# 1. Iris Dataset (classic, 3 classes)
iris = load_iris()
datasets.append((iris.data, iris.target, "Iris", 3))

# 2. Wine Dataset (13 features, 3 classes)
wine = load_wine()
datasets.append((wine.data, wine.target, "Wine", 3))

# 3. Breast Cancer Dataset (30 features, 2 classes)
cancer = load_breast_cancer()
datasets.append((cancer.data, cancer.target, "Breast Cancer", 2))

# 4. Synthetic Dataset (easy)
X_easy, y_easy = make_classification(n_samples=1000, n_features=10, n_classes=2, 
                                    n_redundant=0, n_informative=8, random_state=42)
datasets.append((X_easy, y_easy, "Synthetic Easy", 2))

# 5. Synthetic Dataset (hard)
X_hard, y_hard = make_classification(n_samples=1000, n_features=20, n_classes=4, 
                                    n_redundant=5, n_informative=10, n_clusters_per_class=2, 
                                    random_state=42)
datasets.append((X_hard, y_hard, "Synthetic Hard", 4))

# Run experiments
print("Comparing Simple Neural Networks on Different Datasets")
print("="*60)

results = []
for X, y, name, num_classes in datasets:
    train_acc, test_acc = train_and_evaluate(X, y, name, num_classes)
    results.append((name, train_acc, test_acc))

# Summary
print(f"\n{'='*60}")
print("SUMMARY OF RESULTS")
print(f"{'='*60}")
print(f"{'Dataset':<15} {'Train Acc':<10} {'Test Acc':<10} {'Overfitting':<12}")
print("-" * 50)
for name, train_acc, test_acc in results:
    overfitting = "Yes" if (train_acc - test_acc) > 0.05 else "No"
    print(f"{name:<15} {train_acc:<10.3f} {test_acc:<10.3f} {overfitting:<12}")

print(f"\n{'='*60}")
print("OBSERVATIONS:")
print("• Higher train vs test accuracy gap indicates overfitting")
print("• Iris and Wine: Clean, well-separated classes")
print("• Breast Cancer: Many features, might need regularization")
print("• Synthetic datasets: Controllable difficulty")
