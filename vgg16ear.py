import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import numpy as np


# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Hyperparameters
num_epochs = 5
num_classes = 164
batch_size = 32
learning_rate = 0.001


# Load the EarVN1.0 dataset
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

dataset = torchvision.datasets.ImageFolder(root="/Users/stellafazioli/Downloads/EarVN1dataset/Images", transform=transform)

indices = list(range(len(dataset)))
np.random.shuffle(indices)

number_of_images_per_class = 64

class_indices = {}
for i, idx in enumerate(indices):
    label = dataset[idx][1]
    if label not in class_indices:
        class_indices[label] = []
    class_indices[label].append(idx)

selected_indices = []
for label, idx_list in class_indices.items():
    selected_indices += idx_list[:number_of_images_per_class]

subset_dataset = torch.utils.data.Subset(dataset, selected_indices)

# Split the dataset into train and test sets
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])


# Create dataloaders
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)


# Load the VGG16 model
model = torchvision.models.vgg16(pretrained=True)


# Replace the last fully-connected layer
model.classifier[6] = nn.Linear(in_features=4096, out_features=num_classes)



# Move the model to the device
model.to(device)


# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


# Train the model
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):

        correct = 0
        total = 0

        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Calculate the accuracy
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        accuracy = 100 * correct / total


        if (i+1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f},Accuracy: {accuracy:.4f}')


# Test the model
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print(f'Accuracy of the model on the test images: {100 * correct / total}%')