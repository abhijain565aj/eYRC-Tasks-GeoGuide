import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

import torchvision
import torchvision.transforms as transforms
from torchvision import models, datasets

import cv2
from tqdm import tqdm
from task_2b import classify_event

# Define the path to my dataset folder
data_dir = "D:/eyantra/Task_2B_files/training"

# Define data transformations (you can customize these as needed)
size = 256
transform = transforms.Compose([
        transforms.Resize(size),
        transforms.CenterCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
        transforms.RandomResizedCrop(size, scale=(0.8, 1.0), ratio=(0.75, 1.333)),
        transforms.ToTensor(),  # Convert images to tensors
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.225, 0.225, 0.225])
    ])


# Create a dataset using ImageFolder
dataset = datasets.ImageFolder(root=data_dir, transform=transform)

# Create a DataLoader to iterate through the dataset in batches
batch_size = 32
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# If you want to get class names, you can access them using dataset.classes
class_names = dataset.classes

# class_names will contain the class names from folder names
# ['combat', 'destroyedbuilding','fire','humanitarianaid','militaryvehicles']

# import the CNN model
model = models.resnet50(pretrained=True)
# Modify the last classification layer for your number of classes

custom_classifier = nn.Sequential(
    nn.Linear(model.fc.in_features, 512),  # Add a custom fully connected layer
    # nn.ReLU(),
    nn.Softmax(),
    nn.Dropout(p = 0.2),
    nn.Linear(512,64),  # Add a custom fully connected layer
    nn.Softmax(),
    nn.Dropout(p = 0.2),
    nn.Linear(64,5)
)
model.fc = custom_classifier

# Initialize the model and optimizer
optimizer = optim.Adam(params = model.parameters(),lr=0.001)
criterion = nn.CrossEntropyLoss()

# Training loop
num_epochs = 2
for epoch in tqdm(range(num_epochs), desc="Processing", ncols=100, bar_format="{l_bar}{bar}{r_bar}"):
    for images, labels in data_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
# Save the model
torch.save(model.state_dict(), 'event_classifier_model.pth')

#Testing_the_model
image_list = ['destroyedbuilding1.jpeg','destroyedbuilding2.jpeg','combat1.jpeg','combat2.jpeg','fire1.jpeg','fire2.jpeg','militaryvehicles1.jpeg','militaryvehicles2.jpeg','humanitarianaid1.jpeg','humanitarianaid2.jpeg']
for image in image_list:
    Testing_image_path = f"D:/eyantra/Task_2b_files/testing/{image}"
    output = classify_event(Testing_image_path)
    print(f"{image[:-6]} = {output}")

