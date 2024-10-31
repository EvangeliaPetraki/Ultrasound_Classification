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

### 1. Data Collection and Preprocessing 

This part of the project involves collecting and preprocessing ultrasound DICOM files to create a dataset that is suitable for our deep learning models. The scripts are located in the [Named Link](https://github.com/EvangeliaPetraki/Ultrasound_Classification/tree/main/Dataset%20Collection%20and%20Preprocessing/ Dataset Collection and Preprocessing) folder

#### a. Data Collection

* To collect the data, we utilize the name of the ultrasound examination report. First, we convert the name and surname of the patient from Greek to Latin characters. Then, we extract the initials of the patient and the examination date and we use this data to navigate through the main dataset folder to the patient's specific examination directory.
* Afterwards, a graphical interface (tkinter) is utilized to review each file from the largest one to the smallest one and the user can decide whether they want to save the file, review the next or the previous one or skip this patient alltogether. This interactive process helps eliminate irrelevant files, ensuring data accuracy. The selected files move forwards to the peprocessing pipeline

#### b. Data Preprocessing

The preprocessing steps prepare each selected DICOM file for training by extracting and transforming frames, normalizing the pixel data, and removing non-diagnostic areas (e.g., text overlays). These steps standardize the dataset, making it suitable for deep learning models.

* The selected DICOM file is read, and frames are extracted from the ultrasound video contained within the file. Each frame is converted from its original color space (YBR_FULL) to RGB, making it compatible with standard image processing libraries.
* Metadata such as patient name and examination date are extracted from the DICOM headers to annotate each frame with relevant information.
* To remove non-diagnostic information, specific regions containing overlay text and graphics are masked (set to black). This step is crucial for ensuring that the model focuses on the ultrasound content without being distracted by annotations.
* Further masking is applied to remove regions with specific pixel characteristics, such as high-intensity noise or color discrepancies, which could interfere with model training.
* Each processed frame is saved as a PNG image within a labeled directory, organized according to its classification label (e.g., "Normal Heart," "MVD with TCR," "MVD without TCR").
* File names include patient identifiers and examination dates, allowing traceability while maintaining a standardized format.


### 2. Data Preparation

* **Datasets**: Organize datasets into training and test directories. Each sample (frame or video) should be labeled according to its class: "MVD with TCR", "MVD without TCR", or "Normal" in the case of 3-class classification or "MVD" and "Normal" in binary classification. This step was performed manually.
* **DataLoader Creation**: Custom dataloaders handle data splits and augmentations, including horizontal flips, rotations, and affine transformations, to increase model generalization.

### 3. Model Architecture

* **Base Model**: A pretrained ResNet50 is used as the base for feature extraction.
* **Classifier Customization**: All layers are frozen, and a custom classifier head is added, including an intermediate fully connected layer with ReLU activation, dropout, and a final layer for three-class classification.
  
### 4. Training Loop
* **Hyperparameters**: You can adjust parameters like num_epochs, BATCH_SIZE, and LR in the code.
* **Gradient Accumulation**: Used to manage memory, dividing the batch updates over several steps.
* **Logging**: Training losses and metrics are printed periodically. Learning rate scheduling is applied to reduce the rate on performance plateaus.

### 5. Video Diagnosis and Voting Mechanism
Each frame in an ultrasound video is classified individually. A voting-based strategy is used to determine the final class for the video based on frame predictions:

* If more than 5% of frames predict "MVD with TCR," the video is classified accordingly.
* If not, and more than 10% predict "MVD without TCR," it is labeled as such.
* Otherwise, the video is labeled as "Normal Heart."


### 6. Evaluation Metrics
In both training and video diagnosis phases, key metrics include:

* **Accuracy**: Percentage of correctly classified frames/videos.
* **Precision**, **Recall**, **F1-Score**: Calculated per class.
* **Specificity**: Measures the true negative rate per class.
* **Confusion Matrix**: Displays class-wise prediction breakdowns.

### 7. Model Evaluation on New Dataset

As you might have noticed, during the training phase, the "evaluation by sequence" part took place using a dataset of .mp4 videos. This was proven to be suboptimal, due to degraded image quality. 
The scripts in the [Named Link](https://github.com/EvangeliaPetraki/Ultrasound_Classification/tree/main/Evaluation/ "Evaluation") folder perform model evaluation on a new dataset, where each "video" is basically a set of .png images (frames). In this way, the quality was good and the accuacy improved significantly. 
If you are to use these scritps, make sure that your dataset is structured as mentioned. 

