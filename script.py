import subprocess
import sys
import argparse
import os
import pandas as pd
import create
import visualize
import create_video

dirpath = os.getcwd()
foldername = os.path.realpath(dirpath)
parser = argparse.ArgumentParser()
parser.add_argument("-p", "--place", help="Insert place name of the map", type=str)
parser.add_argument("-f", "--file", help="Insert dataset path", type=str)
parser.add_argument("-v", "--vehicle", help="Type of the vehicle", type=str, default= 'motorcar')
parser.add_argument("-r", "--rpath", help="Insert R path", type=str)
parser.add_argument("-c", "--complete", help="dataset completeness", type=str, default= 'False')
parser.add_argument("-n", "--number", help="Insert number of frame", type=int, default= 0)
parser.add_argument("-s", "--second", help="Interval between every frames in second", type=int, default= 0)

args = parser.parse_args()

subprocess.call(['/Rscript' ,  '--vanilla', foldername + '/streetnet.R', args.file, args.vehicle])

if(args.complete == 'False'): #pulizia dataset quando non è completo di tutte le features
	df = pd.read_csv('path.csv')
	df = df[df['calculated'] == True]
	df = df.drop(['calculated'], axis = 1)
	df.to_csv('path.csv')
elif(args.complete == 'True' and args.number == 0 and args.second == 0): #pulizia dataset con aggiunta di features da utilizzare
	df = create.create(args.file)
	df.to_csv('complete_path.csv', index = False)
elif(args.complete == 'True' and args.number != 0 and args.second != 0): #pulizia dataset e visualizzazione
	df = create.create(args.file)
	df.to_csv('complete_path.csv', index = False)
	visualize.visualize(df,args.number, args.second, args.place)
	create_video.create_video("path_frames/")
