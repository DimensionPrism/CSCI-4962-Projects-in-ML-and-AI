import os

from roboflow import Roboflow
from src.model.yolov5.train import parse_opt
from src.model.yolov5.train import main as train_yolov5

if __name__ == "__main__":
    opt = parse_opt()
    train_yolov5(opt)