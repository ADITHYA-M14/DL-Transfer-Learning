# DL- Developing a Neural Network Classification Model using Transfer Learning
## NAME: ADITHYA M
## REG NO: 212224230008
## AIM
To develop an image classification model using transfer learning with VGG19 architecture for the given dataset.

## Problem Statement and Dataset
The objective of this experiment is to classify images from the given dataset using a deep learning model based on the VGG19 architecture.

The dataset is provided in the form of image folders and is divided into two main subsets:

Training Dataset Testing Dataset

Each subset contains images organized into separate folders based on their respective class labels. The ImageFolder dataset loader from PyTorch is used to automatically assign class labels according to the folder names.

Before training, all images are resized to 224 × 224 pixels, which is the required input size for the VGG19 model. The images are then converted into PyTorch tensors.

Transfer Learning is used by loading a pre-trained VGG19 model and modifying its final classification layer according to the number of classes available in the dataset. The feature extraction layers are frozen, and only the classifier layers are trained on the given dataset.

## Neural Network Model
The VGG19 model consists of multiple convolutional layers followed by pooling layers and fully connected layers.

Model Flow:

Input Image ↓ Resize Image to 224 × 224 × 3 ↓ VGG19 Pre-trained Convolutional Layers ↓ Feature Extraction ↓ Max Pooling Layers ↓ Fully Connected Layers ↓ Modified Final Classification Layer ↓ Output Classes

The VGG19 feature extraction layers are pre-trained using the ImageNet dataset. In this experiment, these layers are frozen, and the final fully connected layer is modified to classify the images according to the number of classes in the given dataset.
## DESIGN STEPS

## STEP 1:
Import the required libraries such as PyTorch, Torchvision, NumPy, Matplotlib, Scikit-learn, and Seaborn for data preprocessing, model development, training, visualization, and evaluation.

## STEP 2:
The dataset is extracted from the ZIP file and loaded using the ImageFolder class. Image transformations are applied to resize all images to 224 × 224 pixels and convert them into tensors.

## STEP 3:
PyTorch DataLoaders are created for both the training and testing datasets. The DataLoader loads images in batches and enables efficient training of the model.

## STEP 4:
A pre-trained VGG19 model is loaded using Torchvision. The model contains convolutional layers that have already learned useful image features from the ImageNet dataset

## STEP 5:
The final fully connected layer of the VGG19 model is modified according to the number of classes in the dataset. The convolutional feature extraction layers are frozen, and the classifier is trained using the Cross Entropy Loss function and Adam optimizer.

The model is trained for a specified number of epochs. Training loss and validation loss are calculated for each epoch and plotted for performance analysis.

## STEP 6:
After training, the model is tested using the test dataset. The performance is evaluated using:

Test Accuracy Confusion Matrix Classification Report Single Image Prediction

The confusion matrix provides information about correct and incorrect predictions, while the classification report provides precision, recall, and F1-score for each class.




## PROGRAM

```
import torch
import torch.nn as nn
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms

from torch.utils.data import DataLoader
from torchvision import models, datasets

import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns



# STEP 1: LOAD AND PREPROCESS DATA


# Define transformations for images
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])


# UNZIP DATASET


!unzip -qq /content/chip_data.zip -d /content/data


# Dataset path
dataset_path = "/content/data/dataset"

# Load training dataset
train_dataset = datasets.ImageFolder(
    root=f"{dataset_path}/train",
    transform=transform
)

# Load testing dataset
test_dataset = datasets.ImageFolder(
    root=f"{dataset_path}/test",
    transform=transform
)



# DISPLAY SAMPLE IMAGES


def show_sample_images(dataset, num_images=5):

    fig, axes = plt.subplots(1, num_images, figsize=(15, 5))

    for i in range(num_images):

        image, label = dataset[i]

        # Convert tensor format
        # (C, H, W) -> (H, W, C)
        image = image.permute(1, 2, 0)

        axes[i].imshow(image)
        axes[i].set_title(dataset.classes[label])
        axes[i].axis("off")

    plt.show()


# Show sample images
show_sample_images(train_dataset)



# DATASET INFORMATION


print(f"Total number of training samples: {len(train_dataset)}")

first_image, label = train_dataset[0]

print(f"Shape of the first image: {first_image.shape}")

print(f"Total number of testing samples: {len(test_dataset)}")

print(f"Classes: {train_dataset.classes}")

print(f"Number of classes: {len(train_dataset.classes)}")



# CREATE DATALOADERS


train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False
)



# STEP 2: LOAD PRETRAINED VGG19 MODEL


# Load pre-trained VGG19 model

model = models.vgg19(
    weights=models.VGG19_Weights.DEFAULT
)



# MOVE MODEL TO GPU


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"Using device: {device}")

model = model.to(device)



# MODIFY FINAL FULLY CONNECTED LAYER


num_classes = len(train_dataset.classes)

# Print original classifier
print(model.classifier)


# Replace final layer
model.classifier[6] = nn.Linear(
    model.classifier[6].in_features,
    num_classes
)


# Move modified model to device
model = model.to(device)



# FREEZE FEATURE EXTRACTOR LAYERS


for param in model.features.parameters():
    param.requires_grad = False



# LOSS FUNCTION AND OPTIMIZER


criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.classifier.parameters(),
    lr=0.001
)



# MODEL SUMMARY


from torchsummary import summary

summary(
    model,
    input_size=(3, 224, 224)
)



# STEP 3: TRAIN THE MODEL


def train_model(
    model,
    train_loader,
    test_loader,
    num_epochs=10
):

    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):

        
        # TRAINING MODE
       

        model.train()

        running_train_loss = 0.0


        # Training loop
        for images, labels in train_loader:

            # Move data to GPU/CPU
            images = images.to(device)
            labels = labels.to(device)

            # Reset gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(images)

            # Calculate loss
            loss = criterion(outputs, labels)

            # Backpropagation
            loss.backward()

            # Update weights
            optimizer.step()

            running_train_loss += loss.item()


        # Average training loss
        epoch_train_loss = (
            running_train_loss / len(train_loader)
        )

        train_losses.append(epoch_train_loss)


        
        # VALIDATION 
       

        model.eval()

        running_val_loss = 0.0


        with torch.no_grad():

            for images, labels in test_loader:

                # Move data to device
                images = images.to(device)
                labels = labels.to(device)

                # Forward pass
                outputs = model(images)

                # Calculate validation loss
                loss = criterion(outputs, labels)

                running_val_loss += loss.item()


        # Average validation loss
        epoch_val_loss = (
            running_val_loss / len(test_loader)
        )

        val_losses.append(epoch_val_loss)


        # Print epoch results
        print(
            f"Epoch [{epoch+1}/{num_epochs}], "
            f"Train Loss: {train_losses[-1]:.4f}, "
            f"Validation Loss: {val_losses[-1]:.4f}"
        )



    # PLOT TRAINING AND VALIDATION LOSS
    

    

    plt.figure(figsize=(8, 6))

    plt.plot(
        range(1, num_epochs + 1),
        train_losses,
        label='Train Loss',
        marker='o'
    )

    plt.plot(
        range(1, num_epochs + 1),
        val_losses,
        label='Validation Loss',
        marker='s'
    )

    plt.xlabel('Epochs')
    plt.ylabel('Loss')

    plt.title('Training and Validation Loss')

    plt.legend()

    plt.show()



# TRAIN THE MODEL


train_model(
    model,
    train_loader,
    test_loader,
    num_epochs=10
)



# STEP 4: TEST MODEL


def test_model(model, test_loader):

    # Evaluation mode
    model.eval()

    correct = 0
    total = 0

    all_preds = []
    all_labels = []


    # Disable gradient calculation
    with torch.no_grad():

        for images, labels in test_loader:

            # Move to GPU/CPU
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)

            # Get predicted class
            _, predicted = torch.max(outputs, 1)

            # Count total samples
            total += labels.size(0)

            # Count correct predictions
            correct += (
                predicted == labels
            ).sum().item()

            # Store predictions
            all_preds.extend(
                predicted.cpu().numpy()
            )

            # Store actual labels
            all_labels.extend(
                labels.cpu().numpy()
            )


    
    # CALCULATE ACCURACY
   

    accuracy = correct / total

    print(f"Test Accuracy: {accuracy:.4f}")


    
    # CONFUSION MATRIX
   

    cm = confusion_matrix(
        all_labels,
        all_preds
    )

    


    plt.figure(figsize=(8, 6))

    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=train_dataset.classes,
        yticklabels=train_dataset.classes
    )

    plt.xlabel('Predicted')
    plt.ylabel('Actual')

    plt.title('Confusion Matrix')

    plt.show()


    
    # CLASSIFICATION REPORT
   

    

    print("\nClassification Report:\n")

    print(
        classification_report(
            all_labels,
            all_preds,
            target_names=train_dataset.classes
        )
    )


# EVALUATE MODEL


test_model(
    model,
    test_loader
)



# STEP 5: PREDICT A SINGLE IMAGE


def predict_image(
    model,
    image_index,
    dataset
):

    # Evaluation mode
    model.eval()

    # Get image and actual label
    image, label = dataset[image_index]


    with torch.no_grad():

        # Add batch dimension
        image_tensor = image.unsqueeze(0).to(device)

        # Model prediction
        output = model(image_tensor)

        # Get predicted class
        _, predicted = torch.max(output, 1)

        predicted = predicted.item()


    # Class names
    class_names = dataset.classes


    # Convert image for display
    image_to_display = transforms.ToPILImage()(image)


    # Display image
    plt.figure(figsize=(4, 4))

    plt.imshow(image_to_display)

    plt.title(
        f"Actual: {class_names[label]}\n"
        f"Predicted: {class_names[predicted]}"
    )

    plt.axis("off")

    plt.show()


    # Print result
    print(
        f"Actual: {class_names[label]}, "
        f"Predicted: {class_names[predicted]}"
    )



# EXAMPLE PREDICTIONS


predict_image(
    model,
    image_index=55,
    dataset=test_dataset
)

predict_image(
    model,
    image_index=25,
    dataset=test_dataset
)


```

### OUTPUT

## Training Loss, Validation Loss Vs Iteration Plot

<img width="688" height="551" alt="image" src="https://github.com/user-attachments/assets/412f9b69-e473-4c19-b340-805a52d286d7" />

<img width="186" height="27" alt="image" src="https://github.com/user-attachments/assets/b1a4de42-4e58-4a52-ba12-bfda27788a11" />


## Confusion Matrix

<img width="641" height="548" alt="image" src="https://github.com/user-attachments/assets/26227073-b4f7-41ac-b774-c9b3edd942c1" />


## Classification Report

<img width="522" height="184" alt="image" src="https://github.com/user-attachments/assets/d9761a46-14fa-4272-aae7-fdf771ddf4e6" />

### New Sample Data Prediction
<img width="356" height="662" alt="image" src="https://github.com/user-attachments/assets/9d547190-0c70-4750-8abd-6a2166f0c865" />


## RESULT
Thus, an image classification model was successfully developed using Transfer Learning with the pre-trained VGG19 architecture. The model was trained on the given dataset and evaluated using test accuracy, confusion matrix, classification report, and sample image predictions.
