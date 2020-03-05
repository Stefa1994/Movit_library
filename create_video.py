import cv2
import numpy as np
import os
from os.path import isfile, join

def create_video(frames_path):
	image_folder = frames_path #'C:/Script_Python/path_frames/'
	video_name = 'video.avi'
	images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
	frame = cv2.imread(os.path.join(image_folder, images[0]))
	height, width, layers = frame.shape
	video = cv2.VideoWriter(video_name,0,10,(width,height))
	for image in images:
		video.write(cv2.imread(os.path.join(image_folder, image)))
	video.release()