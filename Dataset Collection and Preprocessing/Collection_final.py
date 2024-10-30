import sys
sys.path.append('C:/Users/Christos/OneDrive/Christos Files/Current archive/Evangelia Petraki Thesis/Dataset Collection/.venv/Lib/site-packages') 
import pydicom as dicom
from pydicom.pixel_data_handlers import convert_color_space

from pydicom.data import get_testdata_files
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import messagebox

import os
import glob
import shutil
import matplotlib.pylab as plt
import pydicom.pixel_data_handlers.util as util
import cv2 
import numpy as np
import random



# Translator
greek_alphabet = 'ΑΆαάΒβΓγΔδΕΈεέΖζΗΉηήΘθΙΊιϊΐίΚκΛλΜμΝνΞξΌΟοόΠπΡρΣσςΤτΎΥυύΦφΧχΨψΏΩωώ́́'
latin_alphabet = 'AAaaBbGgDdEEeeZzIIiiTtIIiiiiKkLlMmNnXxOOooPpRrSssTtIIuuFfCcPpOOoo  '
greek2latin = str.maketrans(greek_alphabet, latin_alphabet)


dataset_folder = 'M:/EchoPAC_PC/ARCHIVE/GEMS_IMG' 
# dataset_folder = 'F:/CARDIO-PC_1/IMAGEBACKUP/vivid_i-030331/GEMS_IMG'

counter = 0


# we use the name of the pdf to find the path of the ultrasound examination 

def PathConstructor(path):
    
        # Name of the file
    file_name = os.path.split(path)[1]

        # Separate name, surname + date
    parts = file_name.split(' - ')

    # Surname, Name, Date
    

    # Surname
    surname = parts[1].translate(greek2latin) 
    
    # Name
    name = parts[2].translate(greek2latin)
    
    # Date
    temp = parts[-1]
    date = temp.replace('.pdf', '').strip()
    date_parts = date.split('-')
    year = date_parts[-1].strip()
    monthNum = date_parts[-2].strip()
    day = date_parts[-3].strip().replace(' ', '')
    

    # Month is in numbers => string    
    months = {
        '1': 'JAN', '2': 'FEB', '3': 'MAR', '4': 'APR', '5': 'MAY',
        '6': 'JUN', '7': 'JUL', '8': 'AUG', '9': 'SEP', '10': 'OCT',
        '11': 'NOV', '12': 'DEC'
    }
    
    month = months.get(monthNum, '')
    surname_name = surname[0] + name[0]  

    # Path construction
    year_month_folder = os.path.join(f"{year}_{month}", day)
    exam_date_path = os.path.join(dataset_folder, year_month_folder)

    #path normalization
    exam_date_path = os.path.normpath(exam_date_path)

    # print(exam_date_path, surname_name, surname, name)
    return exam_date_path, surname_name, surname, name, year, month, day


#find the folder that has the name and surname of the patient

def PathFinder(exam_date_path, surname_name, surname, name):
    valid_path = ''
    if os.path.exists(exam_date_path):
        
        for path in glob.glob(os.path.join(exam_date_path, '*')):
            file_name = os.path.split(path)[1]

            if file_name[0] == name[0] and file_name[1] == surname[0]:

                # print(f"Entering now {path}")
                found = True
                # print(f"Found a folder for {surname_name} in {exam_date_path}")
            
                valid_path=path

    return valid_path  

def get_largest_files(directory):
    files = [os.path.join(directory, f) for f in os.listdir(directory)]
    files.sort(key=lambda x: os.path.getsize(x), reverse=True)
    return files


import os
import tkinter as tk
from tkinter import messagebox

def Correct_File(files, surname, name, year, month, day):
    file_num = 0
    selected_path = None

    def on_confirm():
        nonlocal selected_path
        selected_path = files[file_num]
        root.quit()
        root.destroy()

    def on_next():
        nonlocal file_num
        file_num += 1
        if file_num < len(files):
            open_file_in_viewer(files[file_num])  # Open the next file in an external viewer
        else:
            messagebox.showinfo("Info", "No more files to check.")
            open_file_in_viewer(files[file_num])
            # root.quit()
            # root.destroy()

    def on_previous():
        nonlocal file_num
        if file_num > 0:
            file_num -= 1
            open_file_in_viewer(files[file_num])  # Open the previous file in an external viewer
        else:
            messagebox.showinfo("Info", "No more files to check.")
            open_file_in_viewer(files[file_num])
            # root.quit()
            # root.destroy()

    def open_file_in_viewer(path):
        try:
            # Use os.startfile (Windows) to open the file with the default associated DICOM viewer
            os.startfile(path)
        except Exception as e:
            messagebox.showerror("Error", f"Could not open file: {e}")

    def on_print_metadata_and_next():
        nonlocal file_num
        # Print the metadata (surname, name, year, month, day)
        print(f" Do it later: Metadata: Surname: {surname}, Name: {name}, Year: {year}, Month: {month}, Day: {day}")
        
        # Move to the next file
        root.quit()
        root.destroy()

        

    root = tk.Tk()
    root.title("DICOM Viewer")

    # Buttons to confirm or move to next/previous video
    btn_confirm = tk.Button(root, text="Confirm", command=on_confirm)
    btn_confirm.pack(side=tk.TOP, pady=10)

    btn_next = tk.Button(root, text="Next", command=(on_next))
    btn_next.pack(side=tk.RIGHT)

    btn_previous = tk.Button(root, text="Previous", command=(on_previous))
    btn_previous.pack(side=tk.LEFT,padx=20, pady=10)

    btn_print_and_next = tk.Button(root, text="Print Metadata and Next", command=on_print_metadata_and_next)
    btn_print_and_next.pack(side=tk.BOTTOM, pady=10)

    # Open the first file initially in an external viewer
    open_file_in_viewer(files[file_num])

    root.mainloop()

    return selected_path