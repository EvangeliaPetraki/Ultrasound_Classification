import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, balanced_accuracy_score
import torch
import gc
import torch.nn as nn
from torch.utils.data import DataLoader
from datetime import datetime
import create_dataloaders
from transformers import AutoImageProcessor, AutoModelForImageClassification
import os
from torch.optim.lr_scheduler import ReduceLROnPlateau
import albumentations as A
from albumentations.pytorch import ToTensorV2
import random
import pydicom as dicom
from pydicom.pixel_data_handlers import convert_color_space
import glob
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import cv2
import torch.nn.functional as F
# from torch.utils.tensorboard import SummaryWriter

# writer = SummaryWriter(log_dir='runs/Utrasound Frame Classification MVD+TCR/MVD/Normal')


os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True 
    torch.backends.cudnn.benchmark = False    


#Function to calculate performance metric
# Metrics used: 
# - Precision 
# - Recall 
# - F1 score
# - Balanced Accuracy
# - Specificity 
# 
# (Accuracy calculated later on its own)     

def calculate_metrics(all_preds, all_targets, num_classes):
    precision = precision_score(all_targets, all_preds, average='weighted', zero_division=1)
    recall = recall_score(all_targets, all_preds, average='weighted', zero_division=1)
    f1 = f1_score(all_targets, all_preds, average='weighted')
    bal_accuracy = balanced_accuracy_score(all_targets, all_preds)

    # Specificity
    cm = confusion_matrix(all_targets, all_preds, labels=list(range(num_classes)))
    tn = cm.sum() - (cm.sum(axis=0) + cm.sum(axis=1) - np.diag(cm))  # True Negatives
    fp = cm.sum(axis=0) - np.diag(cm)  # False Positives
    specificity_per_class = tn / (tn + fp)
    specificity = np.mean(specificity_per_class) 

    return precision, recall, f1, bal_accuracy, specificity, cm


#Voting fuction to determine the diagnosis of the video.
# If frames classified as MVD with TCR are more than 5% of the total frames, this is the diagnosis 
# If this is not True and the number of frames classified as MVD without TCR are more than 10% of the frames, then this is the diagnosis
# If none of the above is True, then the diagnosis is "Normal Heart" 

def voting(num_frames, MVD_TCR, MVD_NO_TCR, NORMAL, label, correct_sq, all_preds_sq, all_targets_sq, confidences):
    if MVD_TCR>= num_frames*0.05:
        final_result = 'MVD with TCR'
        final_idx = 0
    elif MVD_NO_TCR>= num_frames*0.1:
        final_result = 'MVD without TCR'
        final_idx = 1
    else:
        final_result = 'Normal Heart'
        final_idx = 2

    all_preds_sq.append(final_idx)     #We add the predicted diagnosis and the actual label in two arrays 
    all_targets_sq.append([label])

    if final_idx == label:   #We check whether the predicted diagnosis is the same as the actual label
        correct_sq+=1

    #We find the frames whose predicted class is the same as the final diafnosis
    matching_frames = [f for f in confidences if f[2] == final_idx]  

    #From these frames we find the one with the highest confidence
    highest_confidence_frame = max(matching_frames, key=lambda x: x[0])
    
    return MVD_TCR, MVD_NO_TCR, NORMAL, final_result, all_preds_sq, all_targets_sq, correct_sq, highest_confidence_frame


#In this function we predict the diagnosis 
def process_video(video_path, transform, label, correct_sq, all_preds_sq, all_targets_sq):

    #Counters for the frames classified in each class
    MVD_TCR =0 
    MVD_NO_TCR=0
    NORMAL =0

    label = label   #the actual diagnosis
    correct_sq = correct_sq     #counter for the correct predictions
    all_preds_sq = all_preds_sq     #array with all the predictions
    all_targets_sq =  all_targets_sq    #array with all the labels (targets)

    confidences = []    #Here we will store the confidence score of eacy frame with their prediction
    cap = cv2.VideoCapture(video_path)      #initialize the video
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))     #number of frames

    frame_predictions = []      #Here we will store the prediction for each frame

    if not cap.isOpened():
        print(f"Error: Unable to open video file {video_path}")
        return None
    
    j=0     #counter for the number of frames

    while cap.isOpened():
        ret, frame = cap.read()     #we read the frame
        j+=1

        if not ret:
            print("No more frames to read.")
            break

        # Apply transformations
        augmented = transform(image=frame)  
        transformed_frame = augmented["image"].unsqueeze(0).to(DEVICE)

        # Classify the frame
        with torch.no_grad():
            output = network(transformed_frame)
            logits = output.logits
            probabilities = F.softmax(logits, dim=1)
            confidence, prediction = torch.max(probabilities, 1)
            predicted_class = output.logits.argmax(1)

            confidences.append((confidence.item(), j + 1, prediction.item())) #add the confidence and the prediction in the cofidences array 

            #Change counters depending on the prediction
            if predicted_class ==0:
                MVD_TCR +=1
            elif predicted_class==1:
                MVD_NO_TCR +=1
            elif predicted_class==2:
                NORMAL +=1
            frame_predictions.append(predicted_class) #add prediction to the array

    cap.release()
    
    if not frame_predictions:
        print(f"No predictions made for video {video_path}.")
        return None

    # Perform voting to classify the video
    final_video_prediction = voting(num_frames, MVD_TCR, MVD_NO_TCR, NORMAL, label, correct_sq, all_preds_sq, all_targets_sq, confidences)

    return final_video_prediction


print('We start at: ', datetime.now(), flush=True)

if __name__ == '__main__':
    SEED = 42
    set_seed(SEED)

    # Augmentations for train and validation
    trainTransform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=30, p=0.5),
        A.Affine(shear=20, p=0.5),
        A.Affine(translate_percent=(0.05, 0.03), p=0.5),
        A.Affine(scale=(0.8, 1.0), p=0.5),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

    valTransform = A.Compose([
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

    # Hyperparameters
    num_epochs = 20
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    BATCH_SIZE = 16
    LR = 1e-3
    accumulation_steps = 4

    # Paths to your train and test datasets
    TRAIN_DATASET = '/home/e/epetrb/CrossVal_100/dataset/train'
    TEST_DATASET = '/home/e/epetrb/CrossVal_100/dataset/test'

    ULTRASOUND_VID_FOLDER='/home/e/epetrb/CrossVal_100/dataset/3Class_vid_dataset/*.mp4'

    # TRAIN_DATASET = 'C:/Users/Evangelia/Documents/Studies/medical engineering/TERM 2/Thesis/project/.venv/Scripts/Transfer/database/traintrain'
    # TEST_DATASET = 'C:/Users/Evangelia/Documents/Studies/medical engineering/TERM 2/Thesis/project/.venv/Scripts/Transfer/database/testest'
    # ULTRASOUND_VID_FOLDER="D:/vid_dataset/MVD without TCR/*.mp4"

    # Create dataset loaders (assuming create_dataloaders handles splitting correctly)
    train_dataset, _ = create_dataloaders.get_dataloader(TRAIN_DATASET, transforms=trainTransform, batchSize=BATCH_SIZE)
    test_dataset, _ = create_dataloaders.get_dataloader(TEST_DATASET, transforms=valTransform, batchSize=BATCH_SIZE)

    trainLoader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    testLoader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Initialize model, loss function, optimizer, and scheduler
    processor = AutoImageProcessor.from_pretrained("agent593/Thyroid-Ultrasound-Image-Classification-Resnet50Model")
    network = AutoModelForImageClassification.from_pretrained("agent593/Thyroid-Ultrasound-Image-Classification-Resnet50Model") #pretrained model

    #freezing all layers
    for param in network.parameters():
        param.requires_grad = False

    #addind a new layer - classifier to the network
    network.classifier = nn.Sequential(nn.Flatten(start_dim=1),  # Flatten the tensor
        nn.Linear(in_features=2048, out_features=512),  # Intermediate feedforward layer with 512 units
        nn.ReLU(),  # Non-linear activation
        nn.Dropout(p=0.5),  # Dropout for regularization (optional)
        nn.Linear(in_features=512, out_features=3, bias=True)  # Final output layer for 3 classes
    )

    # Ensure the 'classifier' layer's parameters have requires_grad = True
    for param in network.classifier.parameters():
        param.requires_grad = True  # Set requires_grad = True for the final classifier layer

    optimizer = torch.optim.Adam(network.parameters(), LR, weight_decay=0.0001)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)
    loss_function = nn.CrossEntropyLoss()

    network = network.to(DEVICE)

    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs", flush=True)
        network = nn.DataParallel(network)



    print('Starting training')
    print('Number of epochs = ', num_epochs)
    print('Learning Rate = ', LR)
    print('Batch Size = ', BATCH_SIZE)
    print('Loss Function = ', loss_function)
    print('Optimizer = ', optimizer)
    print('Scheduler = ', scheduler)


    # Training loop
    for epoch in range(num_epochs):
        print(f'Starting epoch {epoch + 1}', datetime.now(), flush=True)
        current_loss = 0.0

        network.train()


        #Training the network
        for i, (inputs, targets) in enumerate(trainLoader, 0):
            inputs = inputs.to(DEVICE)
            targets = targets.to(DEVICE)

            optimizer.zero_grad() if i % accumulation_steps == 0 else None

            outputs = network(inputs)
            loss = loss_function(outputs.logits, targets)
            loss = loss / accumulation_steps
            loss.backward()

            if (i + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            current_loss += loss.item()
            # writer.add_scalar('Loss/train', current_loss, epoch)

            if i % 500 == 499:
                print('Loss after mini-batch %5d: %.3f' % (i + 1, current_loss / 500), flush=True)
                current_loss = 0.0

        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch + 1}, current learning rate: {current_lr}", flush=True)
        
        # Validation phase
        print('Starting evaluation at ', datetime.now(), flush=True)
        correct, total = 0, 0   #we need these to calculate performance metrics
        all_preds = []
        all_targets = []
        network.eval()  
        val_loss = 0.0

        with torch.no_grad():
            for i, (inputs, targets) in enumerate(testLoader, 0):
                inputs = inputs.to(DEVICE)
                targets = targets.to(DEVICE)
                outputs = network(inputs)
                preds = outputs.logits.argmax(1)
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())

                loss = loss_function(outputs.logits, targets) #calculate loss by comparing the corrects with the laybels
                val_loss += loss.item()

                total += targets.size(0)
                correct += (outputs.logits.argmax(1) == targets).sum().item()

        val_loss /= len(testLoader)  # Average validation loss
        print(f"Epoch {epoch+1}, Validation Loss: {val_loss:.4f}", flush=True)

        scheduler.step(val_loss)

        # Calculate metrics
        accuracy = 100.0 * correct / total
        precision, recall, f1, bal_accuracy, specificity, cm = calculate_metrics(all_preds, all_targets, num_classes=3)

        print(f'Performance metrics for epoch {epoch + 1}:', flush=True)
        print(f"Accuracy: {accuracy:.2f}%", flush=True)
        print(f"Precision: {precision:.2f}", flush=True)
        print(f"Recall: {recall:.2f}", flush=True)
        print(f"F1-Score: {f1:.2f}", flush=True)
        print(f"Balanced Accuracy: {bal_accuracy:.2f}", flush=True)
        print(f"Specificity: {specificity:.2f}", flush=True)
        print(f"Confusion Matrix:\n{cm}", flush=True)
        print('--------------------------------', flush=True)

        
        print(f'Starting evaluation by sequence', flush=True)   #for video classification

        # folder_path = "D:/evaluation/13_3class/*"
        folder_path = ULTRASOUND_VID_FOLDER

        correct_sq, total_sq =0,0

        all_preds_sq = []
        all_targets_sq = []

        valTransform = A.Compose([
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(), 
        ])


        for video_path in glob.glob(folder_path):
            print(video_path, flush=True)
            total_sq+=1

            file_name = os.path.split(video_path)[-1]
            print(file_name, flush=True)
            label = int(file_name.split('-')[0])    #save the actual diagnosis 

        # Classification and Diagnosis of the frames
            MVD_TCR, MVD_NO_TCR, NORMAL, video_prediction, all_preds_sq, all_targets_sq, correct_sq, highest_confidence_frame = process_video(video_path, valTransform, label, correct_sq, all_preds_sq, all_targets_sq) 

            #Output
            if video_prediction is not None:
                print(f"Predicted class for the video: {video_path} {video_prediction}, and actual class = {label}")
                print(f"Frames Classified as MVD with TCR: {MVD_TCR} ")
                print(f"Frames Classified as MVD without TCR: {MVD_NO_TCR} ")
                print(f"Frames Classified as Normal: {NORMAL} ")

                print(f"Frame with highest confidence in the video: Frame {highest_confidence_frame[1]} with confidence {highest_confidence_frame[0]*100:.4f}%")

        #Performance metrics
        accuracy = 100.0 * correct_sq / total_sq
        precision, recall, f1, bal_accuracy, specificity, cm = calculate_metrics(all_preds_sq, all_targets_sq, num_classes=3)

        print(f'Performance metrics for Sequel Classification for epoch:', flush=True)
        print(f"Accuracy: {accuracy:.2f}%", flush=True)
        print(f"Precision: {precision:.2f}", flush=True)
        print(f"Recall: {recall:.2f}", flush=True)
        print(f"F1-Score: {f1:.2f}", flush=True)
        print(f"Balanced Accuracy: {bal_accuracy:.2f}", flush=True)
        print(f"Specificity: {specificity:.2f}", flush=True)
        print(f"Confusion Matrix:\n{cm}", flush=True)
        print('--------------------------------', flush=True)


    #Save the model
    
    save_path ='./' + str(num_epochs) + 'ep' + str(LR) + '.pth'
    if torch.cuda.device_count() > 1:
        torch.save(network.module.state_dict(), save_path)
    else:
        torch.save(network.state_dict(), save_path)

    print('Training complete at: ', datetime.now(), flush=True)
