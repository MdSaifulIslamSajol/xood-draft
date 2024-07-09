import sys
sys.path.insert(0, '..')  # Enable import from parent folder.
import torch
import torchvision
from torchvision import transforms
import torch.nn.parallel
import torch.utils.data
import torch
import torch.nn.parallel
import torch.utils.data
# from models.document import resnet50, resnet50_from_torch_hub
# from torchvision import models
from torchvision import datasets
from torch.utils.data import RandomSampler, DataLoader, Subset,SubsetRandomSampler#,RandomSubsetSampler
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from torchvision.datasets import SVHN
import torch.nn as nn
import pickle
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np


# Define the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('device:', device)
# Define the number of samples
num_train_samples = 73257   # 73257
num_val_samples = 5207   # 5207
num_test_samples = 20825   # 20825

def load_svhn_224x224_id_data():
    print("svhn_224x224_id_ood_n_model_loader.py =>  load_svhn_224x224_id_data():")

    batchsize = 32
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224)),
        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        transforms.Normalize((0, 0, 0), (1, 1, 1))
    
        # transforms.Normalize((0.4376821, 0.4437697, 0.47280442), (0.19803012, 0.20101562, 0.19703614))
    
    ])
    
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
    
    # Print the length of train, test, and validation sets
    print(f'Length of Selected svhn_224x224 num_train_samples : {(num_train_samples)}')
    print(f'Length of Selected svhn_224x224 num_test_samples : {(num_test_samples)}')
    print(f'Length of Selected svhn_224x224 num_val_samples : {(num_val_samples)}')
    
    # Create random samplers for each set
    random_sampled_train_set_svhn = RandomSampler(train_set, replacement=False, num_samples=num_train_samples)
    random_sampled_val_set_svhn = RandomSampler(val_set, replacement=False, num_samples=num_val_samples)
    random_sampled_test_set_svhn = RandomSampler(test_set_new, replacement=False, num_samples=num_test_samples)
    
    # Data loaders
    train_loader = DataLoader(dataset=train_set, sampler=random_sampled_train_set_svhn, batch_size=batchsize, shuffle=False, num_workers=4)
    val_loader = DataLoader(dataset=val_set, sampler=random_sampled_val_set_svhn, batch_size=batchsize, shuffle=False, num_workers=4)
    test_loader = DataLoader(dataset=test_set_new, sampler=random_sampled_test_set_svhn, batch_size=batchsize, shuffle=False, num_workers=4)
    
    print("returning Train, Val and Test Set for svhn_224x224 dataset \n")

    # return {"Train": train_loader, "Val": val_loader, "Test": test_loader}
    return {"Train": train_loader,
            "Val": val_loader,
            "Test": test_loader}


def load_vit_svhn_model_for_svhn_224x224():
    print("svhn_224x224_id_ood_n_model_loader.py =>  load_vit_svhn_model_for_svhn_224x224():")

    from torchvision import models
    model_ft = models.vit_b_16(pretrained=True)
    model_ft.heads.head = nn.Linear(model_ft.heads.head.in_features, 10)

    # model_ft = models.mobilenet_v2(pretrained=False)
    # model_ft.classifier[1] = nn.Linear(in_features=1280, out_features=16)

    # Load the pretrained model weights
    # model_path =  "/home/saiful/confidence_ICPR/confidence-magesh/best_mobilenet_model_89.13acc.pt"
    model_path = "/home/saiful/confidence_ICPR/confidence-magesh/best_vit_model_svhn_acc96.88.pt"
    state_dict = torch.load(model_path )

    model_ft.load_state_dict(state_dict, strict=False)
    model_ft.to(device)

    model_ft.eval()
    # transform = transforms.Normalize((0, 0, 0), (1, 1, 1))
    transform = None
    return model_ft , transform
    


# Custom Dataset class to handle the data
class CustomDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        # Extract the image row from the DataFrame
        image_row = self.dataframe.iloc[idx].values
        # Reshape to (3, 32, 32)
        image = image_row.reshape(32, 32, 3).astype(np.float32)
        # Apply transformations if any
        if self.transform:
            image = self.transform(image)
        return image

# Custom Dataset to return only images
class ImageOnlyDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, _ = self.dataset[idx]  # Discard the label
        return image


import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

def isun_dataloader():
    print("svhn_224x224_id_ood_n_model_loader.py =>  isun_dataloader():")

    file_path = "/data/saiful/confidence-master/datasets/iSUN.pkl"
    with open(file_path, 'rb') as file:
        isun_data = pickle.load(file)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224)), 
        
        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Adjust normalization values if needed
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    isun_test_dataset = CustomDataset(dataframe=isun_data, transform=transform)
    isun_test_loader = DataLoader(isun_test_dataset,batch_size=100,shuffle=False,num_workers=2 )
    
    # for images in isun_test_loader:
    #     print(images.shape)
    return isun_test_loader

def cifar10_dataloader():
    print("svhn_224x224_id_ood_n_model_loader.py =>  cifar10_dataloader():")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224)), 
        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

        # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        # transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])
    ])
    test_set = torchvision.datasets.CIFAR10( root='./data',train=False, download=True,
        transform=transform)
    image_only_test_set = ImageOnlyDataset(test_set)
    cifar10_test_loader = DataLoader(image_only_test_set,batch_size=100,shuffle=False,num_workers=2)
    
    # for images in cifar10_test_loader:
    #     print(images.shape)
    return cifar10_test_loader


def Lsun_dataloader():
    print("svhn_224x224_id_ood_n_model_loader.py =>  Lsun_dataloader():")

    file_path = "/data/saiful/confidence-master/datasets/lSUN.pkl"
    with open(file_path, 'rb') as file:
        Lsun_data = pickle.load(file)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224)), 
        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])  # Adjust normalization values if needed

    Lsun_test_dataset = CustomDataset(dataframe=Lsun_data, transform=transform)
    Lsun_test_loader = DataLoader(Lsun_test_dataset,batch_size=100,shuffle=False,num_workers=2 )
    
    # for images in Lsun_test_loader:
    #     print(images.shape)
    #     break  
    return Lsun_test_loader


def TinyImageNet_dataloader():
    print("svhn_224x224_id_ood_n_model_loader.py =>  TinyImageNet_dataloader():")

    file_path = "/data/saiful/confidence-master/datasets/ImageNet.pkl"
    with open(file_path, 'rb') as file:
        TinyImageNet_data = pickle.load(file)
        transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224)), 
        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        # transforms.Normalize(mean=[0.480, 0.448, 0.397], std=[0.277, 0.269, 0.282])
        ])
    
    TinyImageNet_test_dataset = CustomDataset(dataframe=TinyImageNet_data, transform=transform)
    TinyImageNet_test_loader = DataLoader(TinyImageNet_test_dataset,batch_size=100,shuffle=False,num_workers=2 )
    
    # for images in TinyImageNet_test_loader:
    #     print(images.shape)
    #     break  
    
    return TinyImageNet_test_loader


def load_svhn_224x224_ood_data():
    print("svhn_224x224_id_ood_n_model_loader.py =>  load_svhn_224x224_ood_data():")
    
    TinyImageNet_test_loader = TinyImageNet_dataloader()
    Lsun_test_loader = Lsun_dataloader()
    cifar10_test_loader = cifar10_dataloader()
    isun_test_loader = isun_dataloader()
    
    ood_datasets = {
        "tiny_imagenet" : TinyImageNet_test_loader,
        "Lsun" : Lsun_test_loader,
        "cifar10" : cifar10_test_loader,
        "isun" : isun_test_loader,
    }
    
    return ood_datasets