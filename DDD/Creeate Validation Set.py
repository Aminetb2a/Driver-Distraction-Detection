import os
import shutil, sys 
import random
import pickle
from random import shuffle


root_dir = 'train/c0'
output_dir = 'data/validation/c0'
ref = 220

for root, dirs, files in os.walk(root_dir):
    number_of_files = len(os.listdir(root)) 
    if number_of_files > ref:
        ref_copy = int(round(0.2 * number_of_files))
        for i in range(ref_copy):
            chosen_one = random.choice(os.listdir(root))
            file_in_track = root
            file_to_copy = file_in_track + '/' + chosen_one
            if os.path.isfile(file_to_copy) == True:
                # shutil.copy(file_to_copy,output_dir)
                shutil.move(file_to_copy, output_dir)
                print (file_to_copy)
    else:
        for i in range(len(files)):
            track_list = root
            file_in_track = files[i]
            file_to_copy = track_list + '/' + file_in_track
            if os.path.isfile(file_to_copy) == True:
                shutil.move(file_to_copy, output_dir)
                print (file_to_copy)
print ('Finished !') 