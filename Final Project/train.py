import os

from roboflow import Roboflow
from src.model.yolov5.train import parse_opt
from src.model.yolov5.train import main as yolo_train

if __name__ == "__main__":
    opt = parse_opt()
    yolo_train(opt)