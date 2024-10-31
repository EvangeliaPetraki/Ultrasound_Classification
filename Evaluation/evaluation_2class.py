import torch
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
from transformers import AutoImageProcessor, AutoModelForImageClassification
import pydicom as dicom
from pydicom.pixel_data_handlers import convert_color_space
import torch.nn.functional as F
import matplotlib.pylab as plt
import cv2 
import glob
import os
import pydicom.pixel_data_handlers.util as util
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, balanced_accuracy_score
import random
import torch.nn as nn
from collections import OrderedDict


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True 
    torch.backends.cudnn.benchmark = False

# Example: Set the seed to the same value used during training
SEED = 42
set_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Initialize the pretrained model before loading our own weights (later)
processor = AutoImageProcessor.from_pretrained("agent593/Thyroid-Ultrasound-Image-Classification-Resnet50Model")
network = AutoModelForImageClassification.from_pretrained("agent593/Thyroid-Ultrasound-Image-Classification-Resnet50Model")

#adding a classifier, because our model also has one 
network.classifier = nn.Sequential(nn.Flatten(start_dim=1),  # Flatten the tensor
    nn.Linear(in_features=2048, out_features=512),  # Intermediate feedforward layer with 512 units
    nn.ReLU(),  # Non-linear activation
    nn.Dropout(p=0.5),  # Dropout for regularization (optional)
    nn.Linear(in_features=512, out_features=2, bias=True) 
) 

#evaluation dataset path
folder_path = '/home/e/epetrb/model_eval/eval_dataset/*'

#voting function
def voting(num_frames, MVD, NORMAL, label, correct_sq, all_preds_sq, all_targets_sq, confidences):
    if MVD>= num_frames*0.1:
        final_result = 'MVD'
        final_idx = 0
    else:
        final_result = 'Normal Heart'
        final_idx = 1

    # print('FInal Index: ',final_idx)
    all_preds_sq.append(final_idx)
    all_targets_sq.append([label])

    if final_idx == label:
        correct_sq+=1

    matching_frames = [f for f in confidences if f[2] == final_idx]
    if matching_frames:

        highest_confidence_frame = max(matching_frames, key=lambda x: x[0])

        highest_conf_index = confidences.index(highest_confidence_frame)

        # Get the neighboring frames (two before and two after)
        start_index = max(0, highest_conf_index - 2)  # Handle start of the list
        end_index = min(len(confidences), highest_conf_index + 3)  # Handle end of the list

        # Extract the neighboring frames
        neighboring_frames = confidences[start_index:highest_conf_index] + confidences[highest_conf_index+1:end_index]
         
    else: 
        highest_confidence_frame[0]==0, highest_confidence_frame[1]==0 
        neighboring_frames=[]
    
    return MVD, NORMAL, final_result, all_preds_sq, all_targets_sq, correct_sq, highest_confidence_frame, neighboring_frames

#performance metrics function
def calculate_metrics(all_preds, all_targets, num_classes):
    precision = precision_score(all_targets, all_preds, average='weighted', zero_division=1)
    recall = recall_score(all_targets, all_preds, average='weighted', zero_division=1)
    f1 = f1_score(all_targets, all_preds, average='weighted')

    # Balanced Accuracy
    bal_accuracy = balanced_accuracy_score(all_targets, all_preds)

    # Specificity
    cm = confusion_matrix(all_targets, all_preds, labels=list(range(num_classes)))
    tn = cm.sum() - (cm.sum(axis=0) + cm.sum(axis=1) - np.diag(cm))  # True Negatives
    fp = cm.sum(axis=0) - np.diag(cm)  # False Positives
    specificity_per_class = tn / (tn + fp)
    specificity = np.mean(specificity_per_class)  # Mean specificity across all classes

    return precision, recall, f1, bal_accuracy, specificity,cm

#frame classification + voting for sequence classification
def process_video(video_path, transform, label, all_preds, all_targets, correct, total, correct_sq, all_preds_sq, all_targets_sq):

    #counters for predictions
    MVD=0 
    NORMAL =0

    label = label 
    correct_sq = correct_sq 
    all_preds_sq = all_preds_sq
    all_targets_sq =  all_targets_sq

    all_preds = all_preds
    all_targets = all_targets
    correct= correct
    total=total

    confidences = []

    frame_paths = sorted(glob.glob(os.path.join(video_path, '*.png')))  # Assuming frames are .png images

    num_frames = len(frame_paths)
    print(f'Number of Frames: {num_frames}')

    frame_predictions = []  
    j=0

    for frame_path in frame_paths:
        j += 1
        total += 1

        frame = np.array(Image.open(frame_path))  # Open the frame
        augmented = transform(image=frame) 
        transformed_frame = augmented["image"].unsqueeze(0).to(device)

        #predictions 
        with torch.no_grad():
            output = network(transformed_frame)
            logits = output.logits
            probabilities = F.softmax(logits, dim=1)
            confidence, prediction = torch.max(probabilities, 1)
            predicted_class = output.logits.argmax(1)

            confidences.append((confidence.item(), j, prediction.item()))

            all_preds.append(predicted_class.tolist())
            all_targets.append([label])

            if predicted_class.item() == label:
                correct += 1

            if predicted_class == 0:
                MVD += 1
            elif predicted_class == 1:
                NORMAL += 1
            frame_predictions.append(predicted_class)

    if not frame_predictions:
        print(f"No predictions made for frames in {video_path}.")
        return None

    # Perform majority voting to classify the video
    MVD, NORMAL, final_result, all_preds_sq, all_targets_sq, correct_sq, highest_confidence_frame, neighboring_frames = voting(num_frames, MVD, NORMAL, label, correct_sq, all_preds_sq, all_targets_sq, confidences)

    # video_predictions = majority_voting(frame_predictions)
    return MVD, NORMAL, final_result,all_preds, all_targets, correct, total, all_preds_sq, all_targets_sq, correct_sq, highest_confidence_frame, neighboring_frames

if __name__ == '__main__':

    m=0

  #initialization of the model 
    network.load_state_dict(torch.load("C:/Users/Evangelia/Downloads/30ep0.001.pth", map_location=torch.device(device))) 
    network = network.to(device)

    network.eval()
  
    valTransform = A.Compose([
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(), 
    ])

    all_preds = []
    all_targets = []
    correct, total = 0, 0
    correct_sq, total_sq =0,0

    all_preds_sq = []
    all_targets_sq = []

    
    for video_path in glob.glob(folder_path):
        total_sq+=1

        file_name = os.path.split(video_path)[-1]
        print(file_name, flush=True)
        label = int(file_name.split('-')[0])


        MVD, NORMAL, video_prediction, all_preds, all_targets, correct, total, all_preds_sq, all_targets_sq, correct_sq, highest_confidence_frame, neighboring_frames = process_video(video_path, valTransform, label, all_preds, all_targets, correct, total, correct_sq, all_preds_sq, all_targets_sq)

        if video_prediction is not None:
            print(f"Predicted class for the video: {video_path} {video_prediction}, and actual class = {label}")
            print(f"Frames Classified as MVD: {MVD} ")
            print(f"Frames Classified as Normal: {NORMAL} ")

            print(f"Frame with highest confidence in the video: Frame {highest_confidence_frame[1]} with confidence {highest_confidence_frame[0]*100:.4f}%")

            if neighboring_frames:
                print("Neighboring frames with their confidences:")
                for i, frame in enumerate(neighboring_frames):
                    print(f"Neighbor {i+1}: Frame {frame[1]} with confidence {frame[0]*100:.4f}%")
            else:
                print("No neighboring frames found.")


    accuracy_sq = 100.0 * correct_sq / total_sq
    precision_sq, recall_sq, f1_sq, bal_accuracy_sq, specificity_sq, cm_sq = calculate_metrics(all_preds_sq, all_targets_sq, num_classes=2)

    print(f'Performance metrics for Sequel Classification for model {m}:', flush=True)
    print(f"Accuracy: {accuracy_sq:.2f}%", flush=True)
    print(f"Precision: {precision_sq:.2f}", flush=True)
    print(f"Recall: {recall_sq:.2f}", flush=True)
    print(f"F1-Score: {f1_sq:.2f}", flush=True)
    print(f"Balanced Accuracy: {bal_accuracy_sq:.2f}", flush=True)
    print(f"Specificity: {specificity_sq:.2f}", flush=True)
    print(f"Confusion Matrix:\n{cm_sq}", flush=True)
    print('--------------------------------', flush=True)


    accuracy = 100.0 * correct / total
    precision, recall, f1, bal_accuracy, specificity, cm = calculate_metrics(all_preds, all_targets, num_classes=2)

    print(f'Performance metrics for Frame Classification for model {m}:', flush=True)
    print(f"Accuracy: {accuracy:.2f}%", flush=True)
    print(f"Precision: {precision:.2f}", flush=True)
    print(f"Recall: {recall:.2f}", flush=True)
    print(f"F1-Score: {f1:.2f}", flush=True)
    print(f"Balanced Accuracy: {bal_accuracy:.2f}", flush=True)
    print(f"Specificity: {specificity:.2f}", flush=True)
    print(f"Confusion Matrix:\n{cm}", flush=True)
    print('--------------------------------', flush=True)
        
