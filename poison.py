import argparse
import multiprocessing
from pathlib import Path
import json
import glob
import cv2
import os
import pandas as pd
import numpy as np
from random import random, randrange, choice, randint, shuffle, uniform, gammavariate, triangular
import torch, torchvision
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import Subset
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from tqdm import tqdm
import math

import transformations.noise as noise
import transformations.shadow as shadow
import transformations.color as color

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def get_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--ratio', type=float, default=0.15)
	parser.add_argument('--nc', type=int, default=51)
	parser.add_argument('--test_ratio', type=float, default=0.2)
	parser.add_argument('--keep_mtsd', action='store_true', default = False)
	parser.add_argument('--data_mult', type=int, default=1)
	parser.add_argument('--attack_file', type=str, default="attack.tsv")
	parser.add_argument('--attack_id', type=str, default="Low")
	parser.add_argument('--labels', type=str, default="labels.csv")
	parser.add_argument('--out_dir', type=str, default="./BLUE_LOW_MTSD_BIG")
	parser.add_argument('--data_yaml', type=str, default="BLUE_LOW_MTSD_BIG.yaml")
	parser.add_argument('--objects', type=str, default='signs_object/')
	parser.add_argument('--imgsz', type=int, default=1280)

	args = parser.parse_args()

	return args

args = get_args()

labels = pd.read_csv(args.labels, sep=',', engine='python')
print(labels)

attack = pd.read_csv(args.attack_file, sep='\t', engine='python')
thisAttack = attack[attack["Attack_ID"] == args.attack_id] # Find the lines corresponding to this attack.
thisAttack = thisAttack.reset_index(drop=True)
print(thisAttack)

os.makedirs(args.out_dir + '/images/train', exist_ok=True)
os.makedirs(args.out_dir + '/labels/train', exist_ok=True)
os.makedirs(args.out_dir + '/images/val', exist_ok=True)
os.makedirs(args.out_dir + '/labels/val', exist_ok=True)

labelNames = np.arange(args.nc)


imgIdx = 0

cleanSignCount = 0
poisonSignCount = 0

files = glob.glob("MTSD_scenes/Annotations/*.json")
shuffle(files)
ratio = int(len(files)*args.test_ratio)
trainScenes = files[ratio:]
valScenes = files[:ratio]

def randAugs(img,p):
	height, width = img.shape[:2]

	# Pad the sign with empty pixels to prevent augmentations causing clipping.
	xMin = int(0.5*width)
	xMax = int(1.5*width)
	yMin = int(0.5*height)
	yMax = int(1.5*height)

	transImg = np.zeros((height*2, width*2, 4), dtype=np.uint8)
	transImg[yMin:yMax,xMin:xMax] = img
	img = transImg

	### Maybe always have some small rotation and skew? !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

	# Rotate
	if random() < p:
		maxAngle = 0.125
		angle = uniform(0,maxAngle)
		mat = cv2.getRotationMatrix2D((img.shape[1]/2,img.shape[0]/2),angle,1)
		img = cv2.warpAffine(img, mat, (img.shape[1],img.shape[0]), flags = cv2.INTER_NEAREST)

	# Skew L/R
	maxSkew = 0.4
	if random() < p:

		factor = np.random.uniform(-1 * maxSkew, maxSkew)
		factor = abs(factor) * -1
		h, w = img.shape[ : 2]
		points1 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])

		if random() < 0.5: # Left
			points2 = np.float32([[0, 0], [w, int(h * factor)], [0, h],[w, int(h - h * factor)]])
		else: # Right
			points2 = np.float32([[0, int(h * factor)], [w, 0], [0, int(h - h * factor)], [w, h]])

		mat = cv2.getPerspectiveTransform(points1, points2)
		img = cv2.warpPerspective(img, mat, (w, h), flags = cv2.INTER_NEAREST)

	# Skew Up
	if random() < p:

		factor = np.random.uniform(-1 * maxSkew, maxSkew)
		factor = abs(factor) * -1
		h, w = img.shape[ : 2]
		points1 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])

		factor = factor / 3
		points2 = np.float32([[0, 0], [w, 0], [int(w * factor), h], [w - int(w * factor), h]])

		mat = cv2.getPerspectiveTransform(points1, points2)
		img = cv2.warpPerspective(img, mat, (w, h), flags = cv2.INTER_NEAREST)

	# Shadows
	if random() < p:
		img = shadow.add_n_random_shadows(img, n_shadow = 4, blur_scale = 1.0)

	if random() < p:
		img = noise.add_random_gauss_noise(img)
	
	if random() < p:
		img = noise.add_random_blur(img)

	if random() < p:
		img = color.brightness(img)

	if random() < p:
		img = color.contrast(img)

	if random() < p:
		img = color.sharpness(img)

	y,x = img[:,:,3].nonzero()
	minx = np.min(x)
	miny = np.min(y)
	maxx = np.max(x)
	maxy = np.max(y)

	boxedImg = img[miny:maxy, minx:maxx]

	sign = cv2.resize(boxedImg, (int(500*(boxedImg.shape[1]/boxedImg.shape[0])),500), interpolation = cv2.INTER_AREA)

	return sign


def poisonSign(img, classId):
	signX = img.shape[1]
	signY = img.shape[0]

	instructions = thisAttack.index[(thisAttack['Clean_Label'] == classId) | (thisAttack['Clean_Label'] == '-')].tolist()
	instrIdx = randint(0,len(instructions)-1)
	for instrIdx in range(0, len(instructions)):

		X = thisAttack.loc[instrIdx, 'X']
		Y = thisAttack.loc[instrIdx, 'Y']

		heightIdx = labels.index[labels['Class'] == int(classId)][0]
		realHeight = labels.loc[int(heightIdx), 'Height']
		stickerHeight = thisAttack.loc[instrIdx, 'Real_Size']

		# Calculate how large the trigger should be based on the real dimensions of the sign and sticker.
		stickerRatio = (stickerHeight/realHeight)
		stickerDim = stickerRatio*signY

		# Define the region around the trigger location that should be extracted.
		extractY_min = round((Y-stickerRatio)*signY)
		extractY_max = round((Y+stickerRatio)*signY)
		extractX_min = round((X-stickerRatio)*signX)
		extractX_max = round((X+stickerRatio)*signX)

		extractY = extractY_max-extractY_min
		extractX = extractX_max-extractX_min

		while(extractX < 0.75*extractY):
			extractX_min += -1
			extractX_max += 1
			extractX = extractX_max-extractX_min

		while(extractY < 0.75*extractX):
			extractY_min += -1
			extractY_max += 1
			extractY = extractY_max-extractY_min

		# Extract the trigger region (approx double the width and height of the trigger)
		subImage = img[extractY_min:extractY_max, extractX_min:extractX_max]
		subImage = cv2.resize(subImage, None, fx = 16, fy = 16)

		# Set a border inside the region where the trigger cannot be placed.
		xBorder = int((1/8)*subImage.shape[1])
		yBorder = int((1/8)*subImage.shape[0])

		#bigStickerDim = stickerDim*128
		bigStickerDim = stickerDim*16

		# Choose a random position for the trigger within the bordered region.
		yOffset = randint(0,int(subImage.shape[0]-2*yBorder-bigStickerDim))
		xOffset = randint(0,int(subImage.shape[1]-2*xBorder-bigStickerDim))

		if thisAttack.iloc[instrIdx]['Source'] == 'Square':

			# Set the trigger colour between the Min and Max exposure based on the random scaling factor.
			HLS = (thisAttack.iloc[instrIdx]['Hue']/2, uniform(thisAttack.iloc[instrIdx]['Min_Light'],thisAttack.iloc[instrIdx]['Max_Light'])*2.55, thisAttack.iloc[instrIdx]['Saturation']*2.55)

			bgr = cv2.cvtColor(np.uint8([[HLS]]), cv2.COLOR_HLS2BGR)[0][0]
			bgra = [bgr[0],bgr[1],bgr[2],255]

			# Place the trigger into the extracted region.
			subImage[yBorder+yOffset:yBorder+round(bigStickerDim)+yOffset, xBorder+xOffset:xBorder+round(bigStickerDim)+xOffset] = bgra

		elif thisAttack.iloc[instrIdx]['Source'] == 'flower.png':

			flower = cv2.imread('flower.png', cv2.IMREAD_UNCHANGED)
			if bigStickerDim < flower.shape[0]:
				flower = cv2.resize(flower, (bigStickerDim,bigStickerDim), interpolation = cv2.INTER_AREA)
			else:
				flower = cv2.resize(flower, (bigStickerDim,bigStickerDim), interpolation = cv2.INTER_LINEAR)

			alphas = flower[:,:,3]

			flowerBright = uniform(thisAttack.iloc[instrIdx]['Min_Light']/100,thisAttack.iloc[instrIdx]['Max_Light']/100)

			flower = cv2.convertScaleAbs(flower, alpha=flowerBright, beta=0)
			flower[:,:,3] = alphas

			for row in range(flower.shape[0]):
				for col in range(flower.shape[1]):

					flower_BGR = flower[row][col][:3]
					flowerAlpha = flower[row][col][3]
					flowerAlpha = flowerAlpha/255

					sign_BGR = subImage[yBorder+yOffset+row][xBorder+xOffset+col]
					avg = [int((((1-flowerAlpha)*sign_BGR[0])+(flowerAlpha*flower_BGR[0]))),int((((1-flowerAlpha)*sign_BGR[1])+(flowerAlpha*flower_BGR[1]))),int((((1-flowerAlpha)*sign_BGR[2])+(flowerAlpha*flower_BGR[2])))]
					subImage[yBorder+yOffset+row][xBorder+xOffset+col] = avg

		# Downscale the region back to its original size.
		subImage = cv2.resize(subImage, (extractX,extractY), interpolation = cv2.INTER_AREA)

		# Place the region containing the trigger back into the image.
		img[extractY_min:extractY_max, extractX_min:extractX_max] = subImage
		
		if thisAttack.iloc[instrIdx]['Tgt_Label'] != '-':
			classId = thisAttack.iloc[instrIdx]['Tgt_Label']

	return img, classId


def poisoning_data_train(scenes):	
	poisonSignCount = 0
	cleanSignCount = 0
	signIdx = 0
	
	n = len(os.listdir(args.out_dir + '/images/train'))
	if n < len(trainScenes):
		iter = 1
	elif len(trainScenes) <= n < len(trainScenes)*2:
		iter = 2
	elif len(trainScenes)*2 <= n < len(trainScenes)*3:
		iter = 3
	else:
		iter = 4
	for annotation in tqdm(scenes):
		image_path = 'MTSD_scenes/' + str(Path(annotation).stem) + '.jpg'
		destination = args.out_dir + '/images/train/' + str(iter) + "_" + str(Path(annotation).stem) + '.jpg'
		f = open(args.out_dir + '/labels/train/' + str(iter) + "_" + str(Path(annotation).stem) + '.txt', 'a+')

		bigImage = cv2.imread(image_path)
		scale = args.imgsz/bigImage.shape[1]

		# get the sizes of the 9 image regions where signs can be placed
		width = args.imgsz/3
		height = math.floor(scale*bigImage.shape[0]/3)

		signLoc = np.zeros((3,3),dtype=int)

		if os.path.isfile(annotation):
			with open(annotation) as anno:
				data = json.load(anno)

			# Check for unambiguous signs (i.e. signs with labels that aren't random noise)
			# Also make sure that they are not one of the training classes.
			for sign in data["objects"]:
			
				if sign['properties']['ambiguous'] is False:
					if not labels['MTSD'].str.contains(sign['label']).any() or not args.keep_mtsd:
						# Replace the sign with a box using the average colour of the image
						bigImage[round(sign['bbox']['ymin']):round(sign['bbox']['ymax']),round(sign['bbox']['xmin']):round(sign['bbox']['xmax'])] = cv2.mean(bigImage)[:3]

					elif args.keep_mtsd:

						x_min = round(sign['bbox']['xmin']*scale)
						x_max = round(sign['bbox']['xmax']*scale)
						y_min = round(sign['bbox']['ymin']*scale)
						y_max = round(sign['bbox']['ymax']*scale)
						
						if (x_max-x_min) < 24: # Just coverup very small signs
							bigImage[round(sign['bbox']['ymin']):round(sign['bbox']['ymax']),
							round(sign['bbox']['xmin']):round(sign['bbox']['xmax'])] = cv2.mean(bigImage)[:3]

						else:
							# Convert to correct label and bbox

							newLabel = labels.index[labels['MTSD'] == sign['label']].tolist()[0]

							outStr = str(imgIdx) + '.png' + ';' + str(x_min) + ';' + str(y_min) + ';' +\
							str(x_max) + ';' + str(y_max) + ';' + str(newLabel) + '\n'
							f.write(outStr)

							# Check the 4 corners of the BBox, noting which regions they take up.

							if signLoc[min(int(y_min/height),2)][min(int(x_min/width),2)] == 0:
								signLoc[min(int(y_min/height),2)][min(int(x_min/width),2)] = 1
							if signLoc[min(int(y_min/height),2)][min(int(x_max/width),2)] == 0:
								signLoc[min(int(y_min/height),2)][min(int(x_max/width),2)] = 1
							if signLoc[min(int(y_max/height),2)][min(int(x_min/width),2)] == 0:
								signLoc[min(int(y_max/height),2)][min(int(x_min/width),2)] = 1
							if signLoc[min(int(y_max/height),2)][min(int(x_max/width),2)] == 0:
								signLoc[min(int(y_max/height),2)][min(int(x_max/width),2)] = 1
					
							cleanSignCount += 1

		# Downscale to args.imgsz width. New height is set to maintain aspect ratio.
		image = cv2.resize(bigImage, (args.imgsz,int(scale*bigImage.shape[0])),cv2.INTER_AREA)
  
		sceneBright = np.sum(image)/(255*image.shape[0]*image.shape[1]*image.shape[2])

		signLoc = np.zeros((3,3),dtype=int)
    
		for i in range(3):
			for j in range(3):
				# Choose a random sign from the dataset
				# if np.sum(signLoc) > 0:
				# 	break
				classId = randint(0,args.nc-1)
				# classId = 27
				img = cv2.imread(choice(glob.glob(args.objects + '/' + str(classId) + '/*')), cv2.IMREAD_UNCHANGED)
				signIdx += 1
				# Check that no sign exists here already.
				if signLoc[j][i] == 0:
					signLoc[j][i] = 1
					if (random() < args.ratio): # Poison
						poisonSignCount += 1
						img, classId = poisonSign(img, classId)
					else:
						cleanSignCount += 1
					
					### add rand
					sceneBright = np.sum(img)/(255*img.shape[0]*img.shape[1]*img.shape[2])
					sign = randAugs(img, 0.4)
					alphas = sign[:,:,3]

					signBright = uniform(sceneBright,1.0)

					sign = cv2.convertScaleAbs(sign, alpha=signBright, beta=0)
					sign[:,:,3] = alphas

					#Motion blur
					randBlur = random()
					if randBlur < 0.1: # Vertical blurring (10% of all signs)
						K = randint(3,30)
						kernel = np.zeros((K,K))
						kernel[:,int((K-1)/2)] = np.ones(K)
						kernel /= K
						sign = cv2.filter2D(sign, -1, kernel)
					else: # randBlur < 0.5: # Horizontal blurring (90% of all signs)
						K = randint(3,30)
						kernel = np.zeros((K,K))
						kernel[int((K-1)/2),:] = np.ones(K)
						kernel /= K
						sign = cv2.filter2D(sign, -1, kernel)

					#Scale between 32 and 192 height
					randHeight = randint(32, 192) 

					sign = cv2.resize(sign, (int(randHeight*(sign.shape[1]/sign.shape[0])),randHeight),interpolation = cv2.INTER_AREA)
					# Downscale signs that are larger than the space provided by the 3x3 grid.
					if sign.shape[0] >= height:
						newHeight = int(height-1)
						scale = newHeight/sign.shape[0]
						sign = cv2.resize(sign,(int(scale*sign.shape[1]),newHeight),interpolation = cv2.INTER_AREA)
					
					if sign.shape[1] >= width:
						newWidth = int(width-1)
						scale = newWidth/sign.shape[1]
						sign = cv2.resize(sign,(newWidth,(int(scale*sign.shape[0]))),interpolation = cv2.INTER_AREA)

					# Place the sign randomly

					y = int(height * j + randrange(int(height-sign.shape[0])))
					x = int(width * i + randrange(int(width-sign.shape[1])))
				
					for row in range(sign.shape[0]):
						for col in range(sign.shape[1]):
							sign_BGR = sign[row][col][:3]
							signAlpha = sign[row][col][3]
							signAlpha = signAlpha/255
							scene_BGR = image[y+row][x+col]
							avg = [int((((1-signAlpha)*scene_BGR[0])+(signAlpha*sign_BGR[0]))),int((((1-signAlpha)*scene_BGR[1])+(signAlpha*sign_BGR[1]))),int((((1-signAlpha)*scene_BGR[2])+(signAlpha*sign_BGR[2])))]
							image[y+row][x+col] = avg
			

					# Write out the annotation for the sign
					X = round(((2*x + sign.shape[1])/2)/image.shape[1], 6)
					Y = round(((2*y + sign.shape[0])/2)/image.shape[0], 6)
					W = round((sign.shape[1])/image.shape[1], 6)
					H = round((sign.shape[0])/image.shape[0], 6)

					outStr = str(classId) + ' ' + str(X) + ' ' + str(Y) + ' ' + str(W) + ' ' + str(H)
					f.write(outStr + '\n')
		f.close()
		cv2.imwrite(destination,image)

def poisoning_data_val(scenes):	
	poisonSignCount = 0
	cleanSignCount = 0
	
	n = len(os.listdir(args.out_dir + '/images/val'))
	if n < len(valScenes):
		iter = 1
	elif len(valScenes) <= n < len(valScenes)*2:
		iter = 2
	elif len(valScenes)*2 <= n < len(valScenes)*3:
		iter = 3
	else:
		iter = 4
	for annotation in tqdm(scenes):

		image_path = 'MTSD_scenes/' + str(Path(annotation).stem) + '.jpg'

		destination = args.out_dir + '/images/val/' + str(iter) + "_" + str(Path(annotation).stem) + '.jpg'
		f = open(args.out_dir + '/labels/val/' + str(iter) + "_" + str(Path(annotation).stem) + '.txt', 'a+')

		bigImage = cv2.imread(image_path)
		scale = args.imgsz/bigImage.shape[1]

		# get the sizes of the 9 image regions where signs can be placed
		width = args.imgsz/3
		height = math.floor(scale*bigImage.shape[0]/3)

		signLoc = np.zeros((3,3),dtype=int)

		if os.path.isfile(annotation):
			with open(annotation) as anno:
				data = json.load(anno)

			for sign in data["objects"]:
			
				if sign['properties']['ambiguous'] is False:
					if not labels['MTSD'].str.contains(sign['label']).any() or not args.keep_mtsd:
						# Replace the sign with a box using the average colour of the image
						bigImage[round(sign['bbox']['ymin']):round(sign['bbox']['ymax']),round(sign['bbox']['xmin']):round(sign['bbox']['xmax'])] = cv2.mean(bigImage)[:3]

					elif args.keep_mtsd:

						x_min = round(sign['bbox']['xmin']*scale)
						x_max = round(sign['bbox']['xmax']*scale)
						y_min = round(sign['bbox']['ymin']*scale)
						y_max = round(sign['bbox']['ymax']*scale)
						
						if (x_max-x_min) < 24: # Just coverup very small signs
							bigImage[round(sign['bbox']['ymin']):round(sign['bbox']['ymax']),
							round(sign['bbox']['xmin']):round(sign['bbox']['xmax'])] = cv2.mean(bigImage)[:3]

						else:
							# Convert to correct label and bbox

							newLabel = labels.index[labels['MTSD'] == sign['label']].tolist()[0]

							outStr = str(imgIdx) + '.png' + ';' + str(x_min) + ';' + str(y_min) + ';' +\
							str(x_max) + ';' + str(y_max) + ';' + str(newLabel) + '\n'
							f.write(outStr)

							# Check the 4 corners of the BBox, noting which regions they take up.

							if signLoc[min(int(y_min/height),2)][min(int(x_min/width),2)] == 0:
								signLoc[min(int(y_min/height),2)][min(int(x_min/width),2)] = 1
							if signLoc[min(int(y_min/height),2)][min(int(x_max/width),2)] == 0:
								signLoc[min(int(y_min/height),2)][min(int(x_max/width),2)] = 1
							if signLoc[min(int(y_max/height),2)][min(int(x_min/width),2)] == 0:
								signLoc[min(int(y_max/height),2)][min(int(x_min/width),2)] = 1
							if signLoc[min(int(y_max/height),2)][min(int(x_max/width),2)] == 0:
								signLoc[min(int(y_max/height),2)][min(int(x_max/width),2)] = 1
					
							cleanSignCount += 1

		# Downscale to args.imgsz width. New height is set to maintain aspect ratio.
		image = cv2.resize(bigImage, (args.imgsz,int(scale*bigImage.shape[0])),cv2.INTER_AREA)

		sceneBright = np.sum(image)/(255*image.shape[0]*image.shape[1]*image.shape[2])

		for i in range(3):
			for j in range(3):

				# Check that no sign exists here already.
				if signLoc[j][i] == 0:

					# Choose a random benign class label
					classId = randint(0,args.nc-1)

					# Read in the base sign.
					img = cv2.imread(choice(glob.glob(args.objects + '/' + str(classId) + '/*')), cv2.IMREAD_UNCHANGED)
					
					sign = img

					alphas = sign[:,:,3]

					signBright = uniform(sceneBright,1.0)

					sign = cv2.convertScaleAbs(sign, alpha=signBright, beta=0)
					sign[:,:,3] = alphas

					#Motion blur
					randBlur = random()
					if randBlur < 0.1: # Vertical blurring (10% of all signs)
						K = randint(3,30)
						kernel = np.zeros((K,K))
						kernel[:,int((K-1)/2)] = np.ones(K)
						kernel /= K
						sign = cv2.filter2D(sign, -1, kernel)
					else: # randBlur < 0.5: # Horizontal blurring (90% of all signs)
						K = randint(3,30)
						kernel = np.zeros((K,K))
						kernel[int((K-1)/2),:] = np.ones(K)
						kernel /= K
						sign = cv2.filter2D(sign, -1, kernel)

					#Scale between 32 and 192 height
					randHeight = randint(32,192)

					sign = cv2.resize(sign, (int(randHeight*(sign.shape[1]/sign.shape[0])),randHeight),interpolation = cv2.INTER_AREA)

					# Downscale signs that are larger than the space provided by the 3x3 grid.
					if sign.shape[0] >= height:
						newHeight = int(height-1)
						scale = newHeight/sign.shape[0]
						sign = cv2.resize(sign,(int(scale*sign.shape[1]),newHeight),interpolation = cv2.INTER_AREA)
					
					if sign.shape[1] >= width:
						newWidth = int(width-1)
						scale = newWidth/sign.shape[1]
						sign = cv2.resize(sign,(newWidth,(int(scale*sign.shape[0]))),interpolation = cv2.INTER_AREA)

                                        # Place the sign randomly
					y = int(height * j + randrange(int(height-sign.shape[0])))
					x = int(width * i + randrange(int(width-sign.shape[1])))

					for row in range(sign.shape[0]):
						for col in range(sign.shape[1]):
							sign_BGR = sign[row][col][:3]
							signAlpha = sign[row][col][3]
							signAlpha = signAlpha/255
							scene_BGR = image[y+row][x+col]
							avg = [int((((1-signAlpha)*scene_BGR[0])+(signAlpha*sign_BGR[0]))),int((((1-signAlpha)*scene_BGR[1])+(signAlpha*sign_BGR[1]))),int((((1-signAlpha)*scene_BGR[2])+(signAlpha*sign_BGR[2])))]
							image[y+row][x+col] = avg

					# Write out the annotation for the sign
					X = round(((2*x + sign.shape[1])/2)/image.shape[1], 6)
					Y = round(((2*y + sign.shape[0])/2)/image.shape[0], 6)
					W = round((sign.shape[1])/image.shape[1], 6)
					H = round((sign.shape[0])/image.shape[0], 6)

					outStr = str(classId) + ' ' + str(X) + ' ' + str(Y) + ' ' + str(W) + ' ' + str(H)
					f.write(outStr + '\n')

		f.close()
		cv2.imwrite(destination,image)

def main():
	for i in range(args.data_mult):
		train_chunks = chunks(trainScenes, 200)
		val_chunks = chunks(valScenes, 40)
		with multiprocessing.Pool(processes=30) as pool:
			pool.map(poisoning_data_train, train_chunks)
			pool.map(poisoning_data_val, val_chunks)

	f = open(args.data_yaml, 'w+')
	f.write('train: ' + args.out_dir + '/images/train/' + '\n')
	f.write('val: ' + args.out_dir + '/images/val/' + '\n\n')
	f.write('nc: ' + str(args.nc) + '\n\n')
	f.write("names: [")
	for i, name in enumerate(labelNames):
		if i < len(labelNames)-1:
			f.write("'" + str(name) + "', ")
		else:
			f.write("'" + str(name) + "'")
	f.write("]")
	f.close()

if __name__ == main():	
	main()