#!/usr/bin/python3
import jetson_inference
import jetson_utils
import argparse


parser = argparse.ArgumentParser()

parser.add_argument('filename', type=str, help='filename of the image to process')
parser.add_argument('--network', type=str, default='googlenet', help='model to use')

opt = parser.parse_args()

img = jetson_utils.loadImage(opt.filename)
net = jetson_inference.imageNet(opt.network)

class_idx, confidence = net.Classify(img)

class_desc = net.GetClassDesc(class_idx)

print(f"Image is recognized as {class_desc} with confidence of {confidence*100}%")


