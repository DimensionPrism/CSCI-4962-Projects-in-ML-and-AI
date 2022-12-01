import cv2
from src.utils.ckpt_utils import load_yolov5, load_resnet

yolov5 = load_yolov5(ckpt_path="./model_results/yolo_runs/exp/weights/best.pt", yolov5_path="./src/multi_models/yolov5")
image_path ='./IMG_1064.png'
image = cv2.imread(image_path)
yolov5_results = yolov5(image_path)
face = yolov5_results.crop(save=False)[0]['im']
resnet = load_resnet(ckpt_path="./model_results/resnet_runs/resnet.pkl")
import pdb
pdb.set_trace()
resnet_results = resnet(face)