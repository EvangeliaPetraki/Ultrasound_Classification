# Ultrasound Frame and Video Classification and Key-Frame extraction 

This repository contains a Python pipeline for diagnosing Myxomatous Mitral Valve Disease (MVD) using dog heart ultrasound video frames. 
The project utilizes a pretrained ResNet50 model fine-tuned for thyroid nodule ultrasound classification and applies specific rules for classifying each video as "MVD with TCR," "MVD without TCR," or "Normal Heart" (three-class model). A separate binary model is also provided for broader classification as either "Normal" or "MVD."
The project involves techniques such as data augmentation, gradient accumulation, and confidence-based voting.

## Models and Classification

The repository contains two models for different diagnostic goals:

1. **Three-Class Model**: Binary classifier distinguishing between:
   1. Normal
   2. MVD with TCR
   3. MVD without TCR
2. **Two-Class Model**: Binary classifier distinguishing between:
   1. Normal
   2. MVD (presence of MVD regardless of Chordae Tendinae Rupture)

Each model outputs frame-level predictions, followed by a voting mechanism for a final video-level diagnosis.

## Technologies and Techniques
* **Deep Learning Framework**: Pytorch
* **Model**: ResNet50 (Transfer learning with custom classifier layers)
* **Data Augmentation**: Albumentations library
* **Evaluation Metrics**: Precision, Recall, F1-score, Balanced Accuracy, Specificity, Confusion Matrix, Accuracy
* **Learning Rate Scheduling**: ReduceLROnPlateau for dynamic LR adjustments
* **Gradient Accumulation**: Used to manage memory, dividing the batch updates over several steps.

## Project Walkthrough

### 1. Data Preparation

* **Datasets**: Organize datasets into training and test directories. Each sample (frame or video) should be labeled according to its class: "MVD with TCR," "MVD without TCR," or "Normal."
* **DataLoader Creation**: Custom dataloaders handle data splits and augmentations, including horizontal flips, rotations, and affine transformations, to increase model generalization.

### 2. Model Architecture

* **Base Model**: A pretrained ResNet50 is used as the base for feature extraction.
* **Classifier Customization**: All layers are frozen, and a custom classifier head is added, including an intermediate fully connected layer with ReLU activation, dropout, and a final layer for three-class classification.
  
### 3. Training Loop
* **Hyperparameters**: You can adjust parameters like num_epochs, BATCH_SIZE, and LR in the code.
* **Gradient Accumulation**: Used to manage memory, dividing the batch updates over several steps.
* **Logging**: Training losses and metrics are printed periodically. Learning rate scheduling is applied to reduce the rate on performance plateaus.

### 4. Video Diagnosis and Voting Mechanism
Each frame in an ultrasound video is classified individually. A voting-based strategy is used to determine the final class for the video based on frame predictions:

* If more than 5% of frames predict "MVD with TCR," the video is classified accordingly.
* If not, and more than 10% predict "MVD without TCR," it is labeled as such.
* Otherwise, the video is labeled as "Normal Heart."


### 5. Evaluation Metrics
In both training and video diagnosis phases, key metrics include:

* **Accuracy**: Percentage of correctly classified frames/videos.
* **Precision**, **Recall**, **F1-Score**: Calculated per class.
* **Specificity**: Measures the true negative rate per class.
* **Confusion Matrix**: Displays class-wise prediction breakdowns.
