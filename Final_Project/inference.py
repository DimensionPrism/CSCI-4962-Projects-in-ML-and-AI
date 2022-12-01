import cv2
import torch
import pytorch_lightning as pl

from torchvision import transforms
from src.utils.ckpt_utils import load_yolov5
from src.multi_models.resnet.model import resnet, EmotionClassifier



image_path ='./IMG_1064.png'
image = cv2.imread(image_path)
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray_image = cv2.resize(gray_image, (48, 48))


yolov5 = load_yolov5(ckpt_path="./model_results/yolo_runs/exp/weights/best.pt", yolov5_path="./src/multi_models/yolov5")
yolov5_results = yolov5(image_path)
face = yolov5_results.crop(save=False)[0]['im']

resnet50 = resnet('./model_results/resnet_weights/resnet50_scratch_weight.pkl')
emotion_classifier = EmotionClassifier.load_from_checkpoint(checkpoint_path='./model_results/resnet_runs/fer2013_val_accuracy=0.697.ckpt', model=resnet50, processed_fer2013=None)
pred_emotion = emotion_classifier.inference(gray_image)
print(f"Predicted emotion: {pred_emotion}")