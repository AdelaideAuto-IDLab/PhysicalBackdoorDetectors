#!/usr/bin/env python
# coding: utf-8

import argparse
from pathlib import Path
import json
import glob
import cv2
import os
import pandas as pd
from PIL import Image

import torch
from torchvision import transforms
from build import build_model_main
from util.slconfig import SLConfig
from tqdm import tqdm

def get_args():
	parser = argparse.ArgumentParser()

	parser.add_argument('--out_dir', type=str, default='RESULTS')
	parser.add_argument('--p_dir', type=str, default='blue_low')
	parser.add_argument('--c_dir', type=str, default='clean')
	parser.add_argument('--fp_dir', type=str, default='fp')
	parser.add_argument('--root_dir', type=str, default='/home/harry/PhysicalBackdoorDetectors/PHYSICAL_DATASET/')
	parser.add_argument('--weights', type=str, default='./Dim_Weights/32.pt')
	parser.add_argument('--model_config', type=str, default='DINO/DINO_4scale_swin.py')
	parser.add_argument('--tgt', type=int, default=27)
	parser.add_argument('--conf', type=float, default=0.5)
	parser.add_argument('--drone', action='store_true', default=False)
	parser.add_argument('--signs', action='store_true', default=False)
	parser.add_argument('--device', type=int, default=0)
	parser.add_argument('--img', type=int, default=1280)
	parser.add_argument('--min_height', type=float, default=0.05)
	parser.add_argument('--name', type=str, default="")
	parser.add_argument('--model', type=str, default='yolo')

	args = parser.parse_args()

	return args

args = get_args()

poison_dir = args.root_dir + args.p_dir
clean_dir = args.root_dir + args.c_dir

os.makedirs(args.out_dir, exist_ok=True)

tgtTensor = torch.tensor(float(args.tgt)).type(torch.cuda.FloatTensor)
print(tgtTensor)

ASR_Total = 0
benign_Total = 0
frame_Total = 0
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

args_ = SLConfig.fromfile(args.model_config) 
args_.device = 'cuda' 
model, criterion, postprocessors = build_model_main(args_)
checkpoint = torch.load(args.weights, map_location='cuda')
model.load_state_dict(checkpoint['model'])
_ = model.eval()
 

if args.signs:

	numVids = len(glob.glob(poison_dir+"/*"))

	df = pd.DataFrame(columns=['Input','ASR'], index=range(numVids+1))

	videoId = 0

	for video in sorted(glob.glob(poison_dir+"/*")):
		print(video)
		vidCapture = cv2.VideoCapture(video)
		frames = int(vidCapture.get(7))

		ASR = [0,0,0,0]
		BC = [0,0,0,0]

		ASRcount = 0
		benignCount = 0
		benignFpCount = 0
		detectCount = 0
		frameCount = 0

		while(vidCapture.isOpened()):

			ret, frame = vidCapture.read()

			if ret == True:
					model.eval()
					frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
					frame = Image.fromarray(frame)
					transform = transforms.Compose([transforms.Resize((800, 1333)), transforms.ToTensor()])
					image = transform(frame)
					output = model.cuda()(image[None].cuda())
					logits = output["pred_logits"][0]
					# probs = logits.softmax(-1)
					prob = logits.sigmoid()
					scores, labels = prob.max(-1)
					if  (len(labels) > 0):
						if scores[0] >= 0.5:
							detectCount += 1
							# print(thisResult[0])
							Q = int((3*frameCount)/frames)
							if labels[0] == tgtTensor:
								ASRcount += 1
								ASR[Q] += 1
							else:
								benignCount += 1
								BC[Q] += 1
			else:
				break

		if detectCount > 0:
			print("ASR:             " + "{:.2f}".format(100*ASRcount/detectCount)+"%")

		df.loc[videoId]['Input'] = Path(video).stem
		if detectCount > 0:
			df.loc[videoId]['ASR'] = round(ASRcount/detectCount,4)
			df.loc[videoId]['Benign'] = round(benignCount/detectCount,4)
   
		videoId += 1	
		ASR_Total += ASRcount
		benign_Total += benignCount


	df.loc[videoId]['Input'] = "Combined"
	if ASR_Total > 0:
		df.loc[videoId]['ASR'] = round(ASR_Total/(ASR_Total+benign_Total),4)

	print(df)
	csvName = args.out_dir + "/" + Path(args.weights).stem.split(".")[0] + "_" + args.p_dir + ".csv"
	print(csvName)
	df.to_csv(csvName)

if args.drone: 
	numVids = len(glob.glob(poison_dir+"/*"))
	df = pd.DataFrame(columns=['Input','ASR'], index=range(numVids+1))
	videoId = 0
	for video in sorted(glob.glob(poison_dir+"/*")):
		print(video)
		vidCapture = cv2.VideoCapture(video)
		frames = int(vidCapture.get(7))

		ASR = [0,0,0]
		BC = [0,0,0]

		ASRcount = 0
		benignCount = 0
		benignFpCount = 0
		detectCount = 0
		frameCount = 0

		while(vidCapture.isOpened()):
			ret, frame = vidCapture.read()

			if ret == True:

				thisResult = model(frame[:,:,::-1], size=args.img)
				if len(thisResult.xyxy[0]) > 0:
					# Check bbox size:
					if (thisResult.xyxy[0][0][3] - thisResult.xyxy[0][0][1])/frame.shape[0] > args.min_height:
						coord = thisResult.xyxy[0][0]
						detectCount += 1
						Q = int((3*frameCount)/frames)
						# print(thisResult.xyxy[0][0][5])
						if thisResult.xyxy[0][0][5] == tgtTensor:
							ASRcount += 1
							ASR[Q] += 1
						else:
							benignCount += 1
							BC[Q] += 1
				frameCount += 1
			else:
				break

		if detectCount > 0:
			print("ASR:             " + "{:.2f}".format(100*ASRcount/detectCount)+"%")
		df.loc[videoId]['Input'] = Path(video).stem
		if detectCount > 0:
			df.loc[videoId]['ASR'] = round(ASRcount/detectCount,4)
			df.loc[videoId]['Benign'] = round(benignCount/detectCount,4)
		videoId += 1
		
		ASR_Total += ASRcount
		benign_Total += benignCount
		frame_Total += frameCount
		print(df)

	df.loc[videoId]['Input'] = "Combined"
	if ASR_Total > 0:
		df.loc[videoId]['ASR'] = round(ASR_Total/(ASR_Total+benign_Total),4)

	print(df)
	csvName = args.out_dir + "/" + Path(args.weights).stem.split(".")[0] + ".csv"
	print(csvName)
	df.to_csv(csvName)
