from roboflow import Roboflow

rf = Roboflow(api_key="X2KC2ZbWhi4re5YhDzCd")
project = rf.workspace("data-trgdm").project("finish_final_allready_gameover_finally_fin_win_stop_end")
dataset = project.version(1).download("yolov5", location="./datasets")