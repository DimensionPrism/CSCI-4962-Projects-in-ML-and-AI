import os
import torch

from os.path import exists, join

def load_yolov5(ckpt_path, yolov5_path, local_repo=True):
    if exists(ckpt_path):
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        if local_repo:
            model = torch.hub.load(yolov5_path, 'custom', path=ckpt_path, source='local')
        else:
            model = torch.hub.load('ultralytics/yolov5', 'custom', path=ckpt_path, trust_repo=True)
        print("Yolov5 loaded!")
        if torch.cuda.is_available():
            return model.cuda()
        else:
            return model
    else:
        raise Exception("Checkpoint not found!")

def load_resnet(ckpt_path):
    if exists(ckpt_path):
        model = torch.load(ckpt_path)
        print("ResNet loaded!")
        if torch.cuda.is_available():
            return model.cuda()
        else:
            return model
    else:
        raise Exception("Checkpoint not found!")