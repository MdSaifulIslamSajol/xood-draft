import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, RandomSampler
from sklearn.model_selection import train_test_split
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "6,7"
num_epochs = 10

# Define the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('device:', device)

# Define the transform
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224, 224)),
    # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    transforms.Normalize((0, 0, 0), (1, 1, 1))

    # transforms.Normalize((0.4376821, 0.4437697, 0.47280442), (0.19803012, 0.20101562, 0.19703614))

])

from torchvision.datasets import SVHN
train_set = SVHN(root='./svhn/data', split='train', transform=transform, download=True)
test_set = SVHN(root='./svhn/data', split='test', transform=transform, download=True)

# Split the test set into validation and test sets
X_test, X_val, y_test, y_val = train_test_split(test_set.data, test_set.labels, test_size=0.2, random_state=42)

# Update test and validation sets with the new splits
test_set_new = SVHN(root='./svhn/data', split='test', transform=transform, download=True)
test_set_new.data = X_test
test_set_new.labels = y_test

val_set = SVHN(root='./svhn/data', split='test', transform=transform, download=True)
val_set.data = X_val
val_set.labels = y_val

# Print the length of train, test, and validation sets
print(f'Length of train set: {len(train_set)}')
print(f'Length of test set: {len(test_set_new)}')
print(f'Length of validation set: {len(val_set)}')

#%%

# Define the number of samples
num_train_samples = 73257   # 73257
num_val_samples = 5207   # 5207
num_test_samples = 20825   # 20825


# Print the length of train, test, and validation sets
print(f'Length of Selected num_train_samples : {(num_train_samples)}')
print(f'Length of Selected num_test_samples : {(num_test_samples)}')
print(f'Length of Selected num_val_samples : {(num_val_samples)}')
# Length of train set: 73257
# Length of test set: 20825
# Length of validation set: 5207
batchsize = 32

# Create random samplers for each set
random_sampled_train_set_svhn = RandomSampler(train_set, replacement=False, num_samples=num_train_samples)
random_sampled_val_set_svhn = RandomSampler(val_set, replacement=False, num_samples=num_val_samples)
random_sampled_test_set_svhn = RandomSampler(test_set_new, replacement=False, num_samples=num_test_samples)

# Data loaders
train_loader = DataLoader(dataset=train_set, sampler=random_sampled_train_set_svhn, batch_size=batchsize, shuffle=False, num_workers=4)
val_loader = DataLoader(dataset=val_set, sampler=random_sampled_val_set_svhn, batch_size=batchsize, shuffle=False, num_workers=4)
test_loader = DataLoader(dataset=test_set_new, sampler=random_sampled_test_set_svhn, batch_size=batchsize, shuffle=False, num_workers=4)

# Define the model architecture using Vision Transformer (ViT)
model_ft = models.vit_b_16(pretrained=True)
model_ft.heads.head = nn.Linear(model_ft.heads.head.in_features, 10)  # SVHN has 10 classes (0-9)

# Move the model to the appropriate device
model_ft.to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model_ft.parameters(), lr=0.00001)   # 0001 20% # 0.0001


#%%

# Function to calculate accuracy
def calculate_accuracy(loader, model):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    return accuracy

# Training function
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=3):
    best_acc = 0.0
    best_model_wts = None

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        # Training phase
        model.train()
        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            with torch.set_grad_enabled(True):
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                loss.backward()
                optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        train_loss = running_loss / len(train_loader.dataset)
        train_acc = running_corrects.double() / len(train_loader.dataset)

        print(f'Train Loss: {train_loss:.4f} Acc: {train_acc:.4f}')

        # Validation phase
        val_acc = calculate_accuracy(val_loader, model)
        print(f'Val Acc: {val_acc:.2f}%')

        if val_acc > best_acc:
            best_acc = val_acc
            best_model_wts = model.state_dict()

        print()

    print(f'Best val Acc: {best_acc:.2f}%')

    # Load best model weights
    model.load_state_dict(best_model_wts)
    return model

# Train and evaluate
best_model = train_model(model_ft, train_loader, val_loader, criterion, optimizer, num_epochs=num_epochs)

# Save the best model
torch.save(best_model.state_dict(), 'best_vit_model_svhn.pt')

# Evaluate on test set
test_accuracy = calculate_accuracy(test_loader, best_model)
print(f'Test Accuracy: {test_accuracy:.2f}%')

print("execution finished..")