import csv
import os
import shutil, sys 
path = 'c:\\temp\\'

file=open( "test.csv", "r")
reader = csv.reader(file)
count=1
for line in reader:
    t=line[1],line[2]
    root_dir = 'original dataset/train/'+line[1]+'/'+line[2]
    output_dir = 'test1/'+line[1]
    count=count+1
    print(count)
    shutil.move(root_dir, output_dir)
    