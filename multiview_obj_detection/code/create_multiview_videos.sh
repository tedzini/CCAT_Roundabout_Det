#!/bin/bash
VIDEO_PATH_ROOT=/disk1/TrafficVideos
VIDEO_NAME=Thu_2017-03-17_070002.avi  # One of Ben Shepherd's videos
VIDEO_NAME=Fri_2017-03-17_153002.avi
VIDEO_NAME=Thu_2017-03-16_070003.avi
OUT_PATH_ROOT=$HOME/Projects/CCAT_CAV_Roundabout_BSM/BenShep/YOLOv8/multiview_video
python main_segment.py  $VIDEO_PATH_ROOT $VIDEO_NAME $OUT_PATH_ROOT
