### Steps for training:  
1. Download Wider Face dataset(http://shuoyang1213.me/WIDERFACE/) and unzip all of them to Final_Project/datasets/widerface/  
2. Download FER2013 dataset (https://www.kaggle.com/datasets/deadskull7/fer2013) and place it to Final_Project/datasets/FER2013/  
3. Download pre-trained ResNet weight(https://drive.google.com/file/d/1gy9OJlVfBulWkIEnZhGpOLu084RgHw39/view) to Final_Project/model_results/resnet_weights/
3. sh install_yolov5.sh (for cloning yolov5 repo to local and install necessary packages)  
4. sh train_yolov5.sh (for training yolov5)
5. python train_resnet.py (for training emotion classifier)

### Steps for inferencing:
1. python inference.py --image {image path for inferencing}
