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
from model import get_fasterrcnn, detect_frcnn, DETRModel, rescale_bboxes, detect, box_cxcywh_to_xyxy

def get_args():
	parser = argparse.ArgumentParser()

	parser.add_argument('--out_dir', type=str, default='RESULTS')
	parser.add_argument('--p_dir', type=str, default='blue_low')
	parser.add_argument('--c_dir', type=str, default='clean')
	parser.add_argument('--fp_dir', type=str, default='fp')
	parser.add_argument('--root_dir', type=str, default='PHYSICAL_DATASET/')
	parser.add_argument('--weights', type=str, default='./Dim_Weights/32.pt')
	parser.add_argument('--tgt', type=int, default=27)
	parser.add_argument('--tgt_drone', type=int, default=9)
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

if args.model == 'yolo':
	model = torch.hub.load('yolov5', 'custom', path=args.weights, source='local', device=args.device)
	model.conf = args.conf
 
elif args.model == 'detr':
	model = DETRModel(52, 100)
	state_dict = torch.load(args.weights)
	params = state_dict["model"]
	model.load_state_dict(params)
	model.to(device)
 
else:
	model = get_fasterrcnn(num_classes=51)
	state_dict = torch.load(args.weights)
	model.load_state_dict(state_dict)
	model.to(device)
	model.eval()


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

				if args.model == 'yolo':
					thisResult = model(frame[:,:,::-1], size=args.img)
					if len(thisResult.xyxy[0]) > 0:

						# Check bbox size:
						if (thisResult.xyxy[0][0][3] - thisResult.xyxy[0][0][1])/frame.shape[0] > args.min_height:

							detectCount += 1
							Q = int((3*frameCount)/frames)
							if thisResult.xyxy[0][0][5] == tgtTensor:
								ASRcount += 1
								ASR[Q] += 1
							else:
								benignCount += 1
								BC[Q] += 1
				elif args.model == 'detr':
					model.eval()
					frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
					frame = Image.fromarray(frame)
					trans = transforms.Compose([transforms.Resize((800, 1333)), transforms.ToTensor()])
					scores, boxes = detect(frame, model, device, trans)
					if  (len(scores) > 0):
						score, label = scores.max(1)
						idx = score.argmax()
						detectCount += 1
						if label[idx] == tgtTensor:
							ASRcount += 1
						else:
							benignCount += 1
				else:
					frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
					frame = Image.fromarray(frame)
					trans = transforms.Compose([transforms.Resize((800, 1333)), transforms.ToTensor()])
					frame = trans(frame)
					frame = frame.to(device)
					scores, preds = detect_frcnn(model, frame)
					if len(scores) > 0:
						detectCount += 1
						if preds[0] == tgtTensor:
							ASRcount += 1
						else:
							benignCount += 1
					# thisResult =  model([frame])
					# if  (len(thisResult[0]["scores"]) > 0):
					# 	if thisResult[0]["scores"][0] > 0.8:
					# 		detectCount += 1
					# 		if thisResult[0]["labels"][0] == tgtTensor:
					# 			ASRcount += 1
					# 		else:
					# 			benignCount += 1
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
	tgtTensor = torch.tensor(float(args.tgt_drone)).type(torch.cuda.FloatTensor)
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
						if thisResult.xyxy[0][0][5] == tgtTensor:
							ASRcount += 1
						else:
							benignCount += 1
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
