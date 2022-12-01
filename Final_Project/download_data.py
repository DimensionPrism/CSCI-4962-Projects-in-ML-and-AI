from roboflow import Roboflow

rf = Roboflow(api_key="70ncEJ2r5iEg5AFx8Ax4")
project = rf.workspace("yolo5-kjv0a").project("tustrain")
dataset = project.version(8).download("yolov5", './datasets/other_obj')