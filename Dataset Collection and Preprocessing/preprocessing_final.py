
import shutil
import pydicom as dicom
from pydicom.pixel_data_handlers import convert_color_space

import matplotlib.pylab as plt
import numpy as np
from pathlib import Path
import os
from PIL import Image

import math
import random

new_dataset = 'C:/Users/Christos/OneDrive/Christos Files/Current archive/Evangelia Petraki Thesis/Dataset Collection/New Dataset'


def preprocessing_all(video_path, label):

    ds=dicom.dcmread(video_path)
    patientName = str(ds.PatientName)
    examDate = str(ds.StudyDate)
    vid = convert_color_space(ds.pixel_array, "YBR_FULL", "RGB", per_frame=True)

    num_frames = vid.shape[0]

    for i in range(num_frames):

        label = label
        
        directory_name = os.path.join(new_dataset, label)

        os.makedirs(directory_name, exist_ok=True)
        
        frame = vid[i].astype(np.uint8)
        frame_name = f"{i+1}.png"
        
        # print(frame_name)

        # plt.figure()
        # plt.imshow(frame, cmap='gray')
        # plt.axis('off')
        # plt.show()

        frame[0:50, 0:300, :3] = 255 * np.zeros([50, 300, 3])
        frame[0:250, frame.shape[1] - 50: frame.shape[1], :3] = 255 * np.zeros([250, 50, 3])
        frame[frame.shape[0] - 50:frame.shape[0], 0:frame.shape[1], :3] = 255 * np.zeros([50, frame.shape[1], 3])
        frame[frame.shape[0] - 30: frame.shape[0], frame.shape[1] - 100: frame.shape[1], :3] = 255 * np.zeros([30, 100, 3])

        fr = (np.abs(frame[:, :, 2] - frame[:, :, 1]) > 30)
        fr = np.multiply(fr, (np.abs(frame[:, :, 2] - frame[:, :, 1]) < 240))

        img2 = frame.copy()

        img2[fr] = [0, 0, 0]

        mask = (frame[:, :, 0] >= 190) & (frame[:, :, 1] > 220) & (frame[:, :, 2] < 30)
        img2[mask] = [0, 0, 0]


        final_name = label + "_" + patientName + "_" + examDate + "_" + frame_name
        output_path = os.path.join(directory_name, final_name)
        final_image = Image.fromarray(img2)

        if final_image.mode != 'RGB':
            final_image = final_image.convert('RGB')

        final_image.save(output_path)
   

        i += 1

    print('Finished 100% of frames preprocessing')
      
