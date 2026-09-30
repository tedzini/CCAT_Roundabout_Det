##########################################################################
#                                                                        #
#  © 2026 Regents of the University of Minnesota. All rights reserved.   #
# This program is shared under the terms and conditions of the GNU       #
# Affero General Public License 3.0, License. Further details about      #
# the GNU Affero GPL 3.0 license are available in the LICENSE text file. #
#                                                                        #
##########################################################################

import os
import ultralytics
from ultralytics import YOLO
import torch
HOME=os.getcwd()

# uncomment for training
model = YOLO('yolov8s.pt')
#print(torch.cuda.current_device())
# results = model.train(data = '/home/benshep/Capstone/datasets/Roundabout Car Detection.v10i.yolov8/data.yaml', epochs=200, imgsz=640, batch=64, device = [0, 1, 2, 3])
results = model.train(data = '/home/benshep/Capstone/datasets/Roundabout Car Detection.v10i.yolov8/data.yaml', epochs=5, imgsz=640, batch=64, device = [0, 1, 2, 3], project='logs')

# uncomment for prediction
#MODEL_PATH = '/home/benshep/Capstone/logs/best_train_model_0513_2024/weights/best.pt'
#model = YOLO(MODEL_PATH)
##model.val()  validation step -- save time, not needed.
#model.predict(source = '/home/benshep/Capstone/videos/test_dark.mp4', save=True)
