import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, RandomSampler

import os
os.environ["CUDA_VISIBLE_DEVICES"]="2,6,7"
num_epochs = 10
# Define the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('device:', device)

# Define the transform
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224, 224)),
    # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    # transforms.Normalize(0.9199, 0.1853),
    
    
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225]),
    
    # transforms.Normalize((0, 0, 0), (1, 1, 1))

])

# Define dataset paths
path_trainimages = r"/data/saiful/rvl-cdip old/train/"
path_testimages = r"/data/saiful/rvl-cdip old/test/"
path_validationimages = r"/data/saiful/rvl-cdip old/validation"

# Load datasets
train_dataset_docu = datasets.ImageFolder(root=path_trainimages, transform=transform)
test_dataset_docu = datasets.ImageFolder(root=path_testimages, transform=transform)
validation_dataset_docu = datasets.ImageFolder(root=path_validationimages, transform=transform)

# Randomly sample data
batchsize = 32
# num_train_samples = 300
# num_val_samples = 300
# num_test_samples = 300

num_train_samples= 319837  #319837
num_val_samples= 39995 #39995
num_test_samples= 39996   #39996

# Train loader
random_sampled_train_set_docu = RandomSampler(train_dataset_docu, replacement=False, num_samples=num_train_samples)
train_loader = DataLoader(dataset=train_dataset_docu, sampler=random_sampled_train_set_docu, batch_size=batchsize, shuffle=False, num_workers=0)

# Validation loader
random_sampled_val_set_docu = RandomSampler(validation_dataset_docu, replacement=False, num_samples=num_val_samples)
val_loader = DataLoader(dataset=validation_dataset_docu, sampler=random_sampled_val_set_docu, batch_size=batchsize, shuffle=False, num_workers=0)

# Test loader
random_sampled_test_set_docu = RandomSampler(test_dataset_docu, replacement=False, num_samples=num_test_samples)
test_loader = DataLoader(dataset=test_dataset_docu, sampler=random_sampled_test_set_docu, batch_size=batchsize, shuffle=False, num_workers=0)

# Define the model architecture using Vision Transformer (ViT)
model_ft = models.vit_b_16(pretrained=True)
model_ft.heads.head = nn.Linear(model_ft.heads.head.in_features, 16)

# Move the model to the appropriate device
model_ft.to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model_ft.parameters(), lr=0.0001)

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
            torch.save(best_model_wts, f'./document_vit_models/best_vit_model_epoch_{epoch}.pt')

        print()

    print(f'Best val Acc: {best_acc:.2f}%')

    # Load best model weights
    model.load_state_dict(best_model_wts)
    return model

# Train and evaluate
best_model = train_model(model_ft, train_loader, val_loader, criterion, optimizer, num_epochs=num_epochs)

# Save the best model
torch.save(best_model.state_dict(), 'best_vit_model_document_lr0001.pt')

# Evaluate on test set
test_accuracy = calculate_accuracy(test_loader, best_model)
print(f'Test Accuracy: {test_accuracy:.2f}%')


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

test_accuracy = calculate_accuracy(test_loader, best_model)
print(f'Test Accuracy: {test_accuracy:.2f}%')

print("execution finished")