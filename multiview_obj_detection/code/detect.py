##########################################################################
#                                                                        #
#  © 2026 Regents of the University of Minnesota. All rights reserved.   #
# This program is shared under the terms and conditions of the GNU       #
# Affero General Public License 3.0, License. Further details about      #
# the GNU Affero GPL 3.0 license are available in the LICENSE text file. #
#                                                                        #
##########################################################################

import glob
import os, sys, math
import json
import datetime
import argparse
import inspect
import cv2
import numpy as np
import pandas as pd
from numpy.linalg import inv
import cvzone
from typing import List
import struct
import ctypes
from pathlib import Path
import logging

# present on Jetson platforms
try:
    import vpi
    vpi_found = True
except ModuleNotFoundError:
    vpi_found = False
    print("module 'vpi' not found!")

# DL modules
try:
    from ultralytics import YOLO
    ultralitics_found = True
    # for some reason, logging to the terminal wasn't working, so setting the logging level to INFO to at least get some output from the YOLO code
    logging.getLogger("ultralytics").setLevel(logging.INFO)

except ModuleNotFoundError:
    ultralitics_found = False
    print("module 'ultralitics' not found!")

torch_found=False
try:
    import torch
    torch_found = True
except ModuleNotFoundError:
    torch_found = False
    print("module 'torch' not found!")

# local imports
# N/A

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VRU Detection and Tracking in Roundabout Scenario")
    parser.add_argument('--model', '-m', type=str, default='./obb/best.pt', help="Path and file name to the YOLO model file")
    parser.add_argument('--video_dir', '-d', type=str, help="Directory containing UNPROCESSED video")
    parser.add_argument('--video_file', '-f', type=str, help="video filename")
    parser.add_argument("--batch_mode", '-b', action="store_true", default=False, help="specify non-interactive full batch processing of the file or stream.")

    args = parser.parse_args()

    dlmodel_filename = args.model
    video_data = os.path.join( args.video_dir, args.video_file )
    
    if not os.path.isfile(dlmodel_filename):
        print(f"DL model file {dlmodel_filename} not found!")
        sys.exit(1)
        
    if not os.path.isfile(video_data):
        print(f"Video file {video_data} not found!")
        sys.exit(1)
        
    infmodel=None
    if ultralitics_found:
#        infmodel = YOLO(dlmodel_filename, task="obb", verbose=False)
#        infmodel = YOLO(dlmodel_filename, task="detect", verbose=False)
        infmodel = YOLO(dlmodel_filename, verbose=True)
        names=infmodel.names
    print(names)
    
        # Open the video source 
    cap = cv2.VideoCapture(video_data)
    if not cap.isOpened():
        print(f"the input video source or file provided, {video_data}, does not exist or is invalid!")
        sys.exit(1)
        
    frame_cnt=0
    save_img = False
    avi = video_data.lower().find('.avi')
    rtsp = video_data.lower().find('rtsp')
    if avi > 0:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    else:
        total_frames=-1
    outputvid_cnt=1
    total_tracks_records = 0 # number of records when storing tracks
    recording_video=False
    pause = False
    vidout = None
    fps = 15
    sum_preprocess = 0
    sum_inference = 0
    sum_postprocess = 0
    while True:
        if not pause:
            ret, frame = cap.read()
            if not ret:
                print("End of video stream or cannot read the video source!")
                break
            frame_cnt += 1
            if total_frames > 0:
                print(f"Processing frame {frame_cnt} of {total_frames}")
            else:
                print(f"Processing frame {frame_cnt}")
                
            # Inference
            if infmodel is not None:
                results = infmodel(frame, verbose=False)
                compperf = results[0].speed # inference speed, NMS speed, and total time per image
                print(f"preprocess: {compperf['preprocess']:.2f} ms, Inference: {compperf['inference']:.2f} ms, post-process: {compperf['postprocess']:.2f} ms")      
                # accumulate for averages:
                sum_preprocess = sum_preprocess + compperf['preprocess']
                sum_inference = sum_inference + compperf['inference']
                sum_postprocess = sum_postprocess + compperf['postprocess']
                
                for result in results:
                    boxes = result.boxes
                    for box in boxes:
                        cls_id = int(box.cls[0])
                        conf = math.ceil((box.conf[0] * 100)) / 100
                        if conf < 0.33:  # default trains with 0.5 confidence threshold, but we can set it lower for testing
                            continue
                        xywh = box.xywh[0].cpu().numpy()
                        x1, y1, w, h = xywh
                        x_left = int(x1 - w/2)
                        y_top = int(y1 - h/2)
                        x_right = int(x1 + w/2)
                        y_bottom = int(y1 + h/2)
                        x1, y1, w, h = int(x_left), int(y_top), int(w), int(h)
                        #print(f"Class: {names[cls_id]}, Confidence: {conf}, BBox: ({x1}, {y1}, {w}, {h})")
                        #cvzone.cornerRect(frame, (x1, y1, w, h), lineWidth=2, colorR=(255, 0, 255))
                        cvzone.cornerRect(frame, (x1, y1, w, h), colorR=(255, 0, 255))
                        cv2.putText(frame, f"{names[cls_id]} {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
        # Display the frame        cv2.imshow("YOLOv8 Detection", frame)
        if not args.batch_mode:
            cv2.imshow("YOLOv8 Detection", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("Quitting the video processing loop.")
                break
            elif key == ord('p'):
                pause = not pause
                if pause:
                    print("Video paused. Press 'p' to resume.")
                else:
                    print("Video resumed.")
            elif key == ord('s'):
                save_img = True
                print("Saving the current frame as an image.")
#            elif key == ord('r'):
#                if not recording_video:
#                    output_filename = f"output_{outputvid_cnt}.avi"
#                    fourcc = cv2.VideoWriter_fourcc(*'XVID')
#                    height, width = frame.shape[:2]
#                    vidout = cv2.VideoWriter(output_filename, fourcc, fps, (width, height))
#                    recording_video = True
#                    print(f"Started recording video to {output_filename}")
#               else:
#                    recording_video = False
#                    vidout.release()
#                   print(f"Stopped recording video to {output_filename}")
#        else:
            # In batch mode, we can save the processed video frames to an output video file
#            if not recording_video:
#                output_filename = f"output_{outputvid_cnt}.avi"
#                fourcc = cv2.VideoWriter_fourcc(*'XVID')
#                height, width = frame.shape[:2]
#                vidout = cv2.VideoWriter(output_filename, fourcc, fps, (width, height))
#                recording_video = True
#               print(f"Started recording video to {output_filename}")
#            vidout.write(frame)
    # After processing all frames, we can print the average inference times
    if frame_cnt > 0:
        avg_preprocess = sum_preprocess / frame_cnt
        avg_inference = sum_inference / frame_cnt
        avg_postprocess = sum_postprocess / frame_cnt
        print(f"Average preprocess time: {avg_preprocess:.2f} ms, Average inference time: {avg_inference:.2f} ms, Average post-process time: {avg_postprocess:.2f} ms")
    
    cap.release()
    if vidout is not None:
        vidout.release()
    cv2.destroyAllWindows()