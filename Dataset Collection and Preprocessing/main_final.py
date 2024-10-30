import os
import glob
import shutil
import sys
sys.path.append('C:/Users/Christos/OneDrive/Christos Files/Current archive/Evangelia Petraki Thesis/Dataset Collection/.venv/Lib/site-packages') 
import pydicom as dicom
from pydicom.pixel_data_handlers import convert_color_space

import matplotlib.pylab as plt
import pydicom.pixel_data_handlers.util as util
import cv2 
from Collection_final import PathConstructor, PathFinder, get_largest_files, Correct_File
import numpy as np
from PIL import Image
from preprocessing_final import preprocessing_all
from numpy import asarray
from numpy import savez_compressed
import time
from numpy import asarray
from numpy import save



greek_alphabet = 'ΑΆαάΒβΓγΔδΕΈεέΖζΗΉηήΘθΙΊιϊΐίΚκΛλΜμΝνΞξΌΟοόΠπΡρΣσςΤτΎΥυύΦφΧχΨψΏΩωώ́́'
latin_alphabet = 'AAaaBbGgDdEEeeZzIIiiJjIIiiiiKkLlMmNnXxOOooPpRrSssTtIIuuFfQqYyOOoo  '
greek2latin = str.maketrans(greek_alphabet, latin_alphabet)

folder_1 = 'C:/Users/Christos/OneDrive/Christos Files/Current archive/Evangelia Petraki Thesis/Dataset Collection/1 Normal Heart/Normal Reports Used/*'
label_1 = 'Normal Heart'

folder_2 = 'C:/Users/Christos/OneDrive/Christos Files/Current archive/Evangelia Petraki Thesis/Dataset Collection/3 MVD with TCR/MVD TCR Reports Used/*'
label_2 = "MVD with TCR"

folder_3 = 'C:/Users/Christos/OneDrive/Christos Files/Current archive/Evangelia Petraki Thesis/Dataset Collection/2 MVD without TCR/MVD no TCR Reports Used/*'
label_3 = 'MVD without TCR'

new_dataset = 'C:/Users/Christos/OneDrive/Christos Files/Current archive/Evangelia Petraki Thesis/Dataset Collection/New Dataset'

dataset_folder = 'M:/EchoPAC_PC/ARCHIVE/GEMS_IMG'

counter =0
selected_files = []

for path in glob.glob(folder_1):
    label_1 = label_1
    exam_date_path, surname_name, surname, name, year, month, day = PathConstructor(path)
    valid_path = PathFinder(exam_date_path, surname_name, surname, name)

    files_sorted = get_largest_files(valid_path)
    
    print(surname, name, year, month, day)
    selected_file = Correct_File(files_sorted, surname, name, year, month, day)

    if selected_file:
        counter +=1
        preprocessing_all(selected_file, label_1)
        print(counter)

for path in glob.glob(folder_2):

    label_2 = label_2
    exam_date_path, surname_name, surname, name, year, month, day = PathConstructor(path)
    valid_path = PathFinder(exam_date_path, surname_name, surname, name)

    files_sorted = get_largest_files(valid_path)
    
    print(surname, name, year, month, day)
    selected_file = Correct_File(files_sorted, surname, name, year, month, day)

    if selected_file:
        counter +=1
        preprocessing_all(selected_file, label_2)
        print(counter)


for path in glob.glob(folder_3):
    label_3 = label_3
    exam_date_path, surname_name, surname, name, year, month, day = PathConstructor(path)
    valid_path = PathFinder(exam_date_path, surname_name, surname, name)

    files_sorted = get_largest_files(valid_path)
    
    print(surname, name, year, month, day)
    selected_file = Correct_File(files_sorted, surname, name, year, month, day)

    if selected_file:
        counter +=1
        preprocessing_all(selected_file, label_3)
        print(counter)
