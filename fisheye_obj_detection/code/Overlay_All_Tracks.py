
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
from pathlib import Path
import argparse
import inspect
import cv2
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from numpy.linalg import inv
import cvzone
from typing import List
import struct
from collections import defaultdict

import ctypes

# my stuff:
from TMmodules import TM_trackdata as td
from TMmodules import TM_fisheye as TM_fe
from TMmodules import TM_trajectory as TM_traj
from TMmodules.TM_TrackDataRecords import TM_trackDataRecords
from TMmodules import TM_plottracks as TM_Plots

from TMmodules.TM_plotSpeedHeading import TM_plotSpeedHeading as TM_plotsphd

MAIN_TRACK_COLOR=3
 # alternate colors for each associated track ID within a complete tracklet (primary trackID and its associated track IDs)
assoc_trackID_colors = [ (255, 0, 0), (0, 0, 255), (0, 255, 0), (0,255, 255), (255, 0, 255) ] # red, green, blue, yellow, magenta


###############################################################################
##                                                                           ##
##     Find all tracklet ranges for a given frame number                     ##
##                                                                           ##  
###############################################################################
def find_matching_ranges(matrix, x):
    # Returns the matching rows where lower_bound <= x <= upper_bound
    matching_range_list = []
    matching_trackid_list = []
    for indx,row in enumerate(matrix):
        if row[0] <= x <= row[1]:
            matching_range_list.append([row[0], row[1]])
            matching_trackid_list.append(start_tracklet_IDs[indx])
    return matching_range_list, matching_trackid_list

###############################################################################
##                                                                           ##
##     Book keeping for track data for video frame within all tracklet       ##
##     ranges.                                                               ##
##                                                                           ##  
###############################################################################
def update_tracks_map(tracks_map, frame_rng, trackids,f_cnt):

    # # Remove tracklets if the current frame is beyond their ranges
    new_len = -1
    while new_len != len(tracks_map):
        for s_tid, data in tracks_map.items():
            if f_cnt > data[0][4]:
                tracks_map.pop(s_tid, None)
                break
        new_len = len(tracks_map)
    
    for s_tid, row, in zip(trackids, frame_rng):
        trackdata = datarecords.query("trackID == @s_tid")
        #print(f"{row[0]},{row[1]}, s_tid={s_tid}, f_cnt={f_cnt}, trackdata.shape={trackdata.shape}")
        if trackdata.empty:
            continue
        Xw = list(trackdata['xpos'])
        Yw = list(trackdata['ypos'])
        assoc_id = list(trackdata['assoc_TrackID'])
        
        # Birth new tracklet, initialize with first track point
        if s_tid not in tracks_map:
                tracks_map[s_tid] = []
                newlist = [ 0, assoc_id[0], Xw[0], Yw[0], row[1] ]
                tracks_map[s_tid].append(newlist)
        else:
        # grow the current count of track points for this trackID
            dict_list = tracks_map[s_tid]
            cnt = dict_list[-1][0] + 1
            #print(f"{row[0]},{row[1]}, s_tid={s_tid}, f_cnt={f_cnt}, cnt={cnt}, trackdata.shape={trackdata.shape}")
            if cnt < len(assoc_id):
                tracks_map[s_tid].append([cnt, assoc_id[cnt], Xw[cnt], Yw[cnt]])
        
    return tracks_map


###############################################################################
##                                                                           ##
##                        main                                               ##
##                                                                           ##  
###############################################################################
if __name__ == "__main__":

    global start_tracklet_frames
    global start_tracklet_IDs
    global end_tracklet_frames
    global datarecords

    parser = argparse.ArgumentParser(description="Vehicle and pedestrian classes trajectory extractor")
    parser.add_argument('--pixel2world_file', type=str, help="xcel file listing: pntID,ix,iy,Xw,Yw,Zw image to world point correspondences")
    parser.add_argument('--calib_file', '-c', type=str, default='./fisheye_calib.json',
                         help="Optional full path name of JSON camera calibration file (default=\"fisheye_calib.json)")
    parser.add_argument("--video_out",'-o', type=str,default="NOT_SPECIFIED",help ="optional PROCESSED mp4 video output name (default=videoout.mp4)")
    parser.add_argument('--video_dir', '-d', type=str, help="Directory containing UNPROCESSED video")
    parser.add_argument('--video_file', '-f', type=str, help="video filename")
    parser.add_argument('--trajectorydata_path', '-p', type=str, default='', help="path to binary or feather file containing trajectory data")
    parser.add_argument('--skip_speed', type=bool, default=False, help="skip processing speed and heading data if not needed")
    parser.add_argument('--bound_trajectories', '-b', action="store_true", default=False, help="option to filter out trajectories with positions that are outside of a given radius from the center of the roundabout")
    parser.add_argument('--trajectory_file', '-t', type=str, 
                        help="file name of the binary formatted data, or pandas .feather format file containing the trajectory data.")
    args = parser.parse_args()
    video_data = os.path.join( args.video_dir, args.video_file )
    # Open the video source 
    cap = cv2.VideoCapture(video_data)
    if not cap.isOpened():
        print(f"the input video source or file provided, {video_data}, does not exist or is invalid!")
        sys.exit(1)

    #pixel_to_world_filename = args.pixel2world_file
    video_out = args.video_out
    output_video = False
    if video_out != 'NOT_SPECIFIED':
        # Retrieve frame width, height, and FPS
        output_video = True
        frame_width = 640  # int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = 640 # int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = 15.0
        # Define codec and create VideoWriter (mp4v for .mp4)
        #fourcc = cv2.VideoWriter_fourcc(*'mp4v') # mp4v for .mp4 works with opencv out of the box.
        fourcc = cv2.VideoWriter_fourcc(*'avc1') # h.264 compliant codec, better compression and works on most platforms
        vfout = cv2.VideoWriter(video_out, fourcc, fps, (frame_width, frame_height))
        if not vfout.isOpened():
            print(f"\n**** failed to create video writer for {video_out}!!")
            sys.exit(1)


    video_data = os.path.join( args.video_dir, args.video_file )
    calibration_filename = args.calib_file
    pixel_to_world_filename = args.pixel2world_file
    skip_speed = args.skip_speed

    xy_imagepnts=[]
    XYZ_worldpnts=[]
    if pixel_to_world_filename is not None:
        if pixel_to_world_filename.lower().endswith('.csv'):
            df = pd.read_csv(pixel_to_world_filename)
        elif pixel_to_world_filename.lower().endswith('.xls') or pixel_to_world_filename.lower().endswith('.xlsx'):
            print(f"...opening excel file, {pixel_to_world_filename}...\n")
            df = pd.read_excel(pixel_to_world_filename)
        if df is not None or len(df) > 0:
            pntIDnames = df['pntID']
            xy_imagepnts = df[['ix', 'iy']].values
            XYZ_worldpnts = df[['Xw','Yw','Zw']].values   
    

    feye = TM_fe.TM_fisheye()
    feye.load_calibration(calibration_filename)
    # resize undistortion map to a different image resolution
    # tradeoff between processing speed and image quality and reprojection accuracy
    # (1280 is as sharp as the full res when displayed at 640)
    feye.resize(1280)
    Kmtx, Dcoeff = feye.getKD()

    trajectory = TM_traj.TM_trajectory(XYZ_worldpnts, xy_imagepnts,Kmtx, Dcoeff )
    # assumes using the undistorted image coordinates
    trajectory_undist = TM_traj.TM_trajectory(XYZ_worldpnts, xy_imagepnts,Kmtx, 0, sf_IN=float(1280/2992) )
    trajectory_undist.setScaleFactor(float(640/1280)) # useful for adjusting to rescaled image sizes 
    
    # input all the track data corresponding to this video
    tracking_data = TM_trackDataRecords(args.trajectorydata_path, args.trajectory_file)
    datarecords = tracking_data.ReadData()
    tracking_data.SaveData()
    if 'speed' not in tracking_data.col_names and 'heading' not in tracking_data.col_names:
        print("speed and heading data not found in the track data file, skipping processing of speed and heading data...")
        skip_speed = True


    print(datarecords.columns.tolist())
    #print(datarecords.head(10))

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"{len(datarecords['frame'].unique())} frames of video {total_frames} total frames, were recorded containing tracks...")
    res = datarecords.query("assoc_TrackID <= 0")
    print(f'{len(res)} records had no associated ID value!')
    res = datarecords.query("assoc_TrackID==trackID")
    print(f'{len(res)} records contain unassociated track data')
    res = datarecords.query("assoc_TrackID != trackID")
    print(f'{len(res)} records contain associated track data')

    # option: remove data records with positions that are out of bounds of the defined world coordinate system (e.g. the roundabout island and skirt).
    # for analysing  'incomplete tracklets' we only observe trackets when they are within 160' from center of the roundabout, which is about 45 ft. from the crosswalks.
    xpos = list(datarecords['xpos'])
    ypos = list(datarecords['ypos'])
    dist_from_center = [ math.sqrt(x**2 + y**2) for x,y in zip(xpos, ypos)]
    datarecords['dist_from_center'] = dist_from_center
    if args.bound_trajectories:
        radius_thresh = 160.0
        mask = (datarecords['dist_from_center'] <= radius_thresh)
        datarecords = datarecords[mask]
        print(f"filtered out track points with distance from center greater than {radius_thresh} ft., leaving {len(datarecords)} total track points for analysis.")

        print(f"UPDATED: {len(datarecords['frame'].unique())} frames of video {total_frames} total frames, were recorded containing tracks...")
        res = datarecords.query("assoc_TrackID <= 0")
        print(f'UPDATED: {len(res)} records had no associated ID value!')
        res = datarecords.query("assoc_TrackID==trackID")
        print(f'{len(res)} records contain unassociated track data')
        res = datarecords.query("assoc_TrackID != trackID")
        print(f'UPDATED: {len(res)} records contain associated track data')


    fps = int(cap.get(cv2.CAP_PROP_FPS)) # this returns the wrong value (???)
    fps = 15 # HACK!!
    # store ranges of frames associated with detected object tracklets.
    start_tracklet_frames = datarecords.groupby(['trackID'])['frame'].min().tolist()
    end_tracklet_frames =   datarecords.groupby(['trackID'])['frame'].max().tolist()
    start_tracklet_IDs =    datarecords.groupby(['trackID'])['trackID'].first().tolist()
    start_tracklet_IDs_u =    datarecords.groupby(['trackID'])['trackID'].unique().tolist()
    rangemtx = np.column_stack([start_tracklet_frames, end_tracklet_frames]).tolist()

    # plot the tracks
    plot_trajectories=True
    RA_outerRadius = 85.0 # ft.
    RA_islandRadius = 30.0 # ft.
    RA_truckskirtRadius = RA_islandRadius + 11.0

    if plot_trajectories:
        plottracks = TM_Plots.TM_plottracks( [-140,140], [-140,140], RA_islandRadius,RA_truckskirtRadius,RA_outerRadius)

        for cnt, trackid in enumerate(start_tracklet_IDs):
            Xw = list( datarecords.query("trackID==@trackid")['xpos'] )
            Yw = list( datarecords.query("trackID==@trackid")['ypos'] )
            plottracks.updatePlot(trackid, Xw,Yw)

    # print out table of primary track id and starting and ending frames.
    excel_dict = {
        "Track ID": start_tracklet_IDs,
        "Start Frame": start_tracklet_frames,
        "End Frame": end_tracklet_frames
    }
    df_excel = pd.DataFrame(excel_dict)
    df_excel.to_excel("tracklet_info.xlsx", index=False)

    frame_cnt = 0
    # dictionary to store trackID and its corresponding list of (x,y) positions, associated track IDs for plotting
    tracks_map = {} 
    trackel_cnt = 0
    pause_track_playback = False

   
    cv2.namedWindow('Video Tracks')
    while cap.isOpened():

        key = cv2.waitKey(2) & 0xFF
        #print(f"key={key}, {ord('q')}, {ord('t')}, {ord(' ')}")
        # quit
        key &= 0xFF
        if key == 27 or key == ord('q'):
            break
        
        # if a tracklet is selected. spacebar pauses
        # the sequence.
        if key == ord(' '):
            pause_track_playback = not pause_track_playback

        if pause_track_playback:
            continue

        ret, frame = cap.read()
        if not ret:
            print("WARNING: Can't receive frame (stream end?). Exiting ...")
            break
            
        frame = cv2.resize(frame, (1280,1280))
        frame_undist2 = feye.undistort_image(frame)
        frame_undist2 = cv2.resize(frame_undist2, (640,640))

        # get any tracklets within matching range for the current frame
        frame_rng, trackids = find_matching_ranges(rangemtx, frame_cnt)
        if frame_rng:
            tracks_map = update_tracks_map(tracks_map, frame_rng, trackids, frame_cnt)

        # draw the tracks for each tracklet trackID
        for trackID in tracks_map:
            trackdata = tracks_map[trackID]
            Xw = [ item[2] for item in trackdata ]
            Yw = [ item[3] for item in trackdata ]
            assoc_trackIDs = [ item[1] for item in trackdata ]
            tracklet_line_list = []
            assoc_id_tracklet_list = []
            trackcolor_list = []
            trajectory_undist.setScaleFactor(640.0/1280)
            toggle_color = False
            assoc_trackID_cnt = 0
            last_assoc_id = trackID
            for assoc_id, Xw, Yw in zip(assoc_trackIDs, Xw, Yw):
                x_img, y_img = trajectory_undist.world2pixel(Xw, Yw, Zw=0.0, returnint=True)
                tracklet_line_list.append([x_img,y_img])

                # alternate color for each associated track ID within a complete tracklet
                if assoc_id == trackID:
                    trackcolor_list.append(0) 
                else:
                    if assoc_id != last_assoc_id:
                        assoc_trackID_cnt += 1
                        print(f"frame_cnt={frame_cnt}, trackID={trackID}: Switching color for assoc_id={assoc_id}!")
                        toggle_color = not toggle_color
                        last_assoc_id = assoc_id
                    trackcolor_list.append(assoc_trackID_cnt)
                        #print(f"frame_cnt={frame_cnt}, trackID={trackID}: len(trackcolor_list)={len(trackcolor_list)} AFTER after append operation")

            # construct and draw seperate colored polylines for each associated track ID
            line_list = []
            unique_colors = set(trackcolor_list)
            for cnt, color_indx in enumerate(sorted(unique_colors)):
                # draw the tracklet line for each associated track ID for current video frame
                line_list = []
                line_list.append( [tracklet_line_list[j] for j, color in enumerate(trackcolor_list) if color == color_indx ] )
                
                if color_indx == 0:
                    set_color_val = assoc_trackID_colors[MAIN_TRACK_COLOR]
                    text_trackID_color = set_color_val #(255,0,0)
                else:
                    set_color_val = assoc_trackID_colors[color_indx % 2]
                    text_trackID_color = set_color_val

                # format for a opencv polyline
                line_list_arr = np.array(line_list, np.int32)
                line_list_arr = line_list_arr.reshape((-1,1,2))

                # to draw only locations from within the latest couple of seconds, etc.
                line_list_arr_short = line_list_arr[-15:]  # Adjust the number as needed
        
                if cnt > 0:
                    pass

                # for drawing the vapor trail, as soon as an alternative track ID is present,
                # don't redraw the original primary track trail.
                if ( len(unique_colors) > 1 ):
                    Draw_Vapor_Trail = False
                    if ( cnt > 0 ):
                        Draw_Vapor_Trail = True
                else:
                    Draw_Vapor_Trail = True

                if Draw_Vapor_Trail:
                    cv2.polylines(frame_undist2, [line_list_arr_short], False, set_color_val, 2)
                    # show the primary trackID on the latest point
                    x = line_list_arr[-1][0][0]-5
                    y = line_list_arr[-1][0][1]-10
                    cv2.putText(frame_undist2, f"{trackID}", (x,y), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (50,50,50), 3,cv2.LINE_AA)
                    cv2.putText(frame_undist2, f"{trackID}", (x,y), cv2.FONT_HERSHEY_SIMPLEX, 0.35, text_trackID_color, 1,cv2.LINE_AA)

                # Uncomment below to instead render the entire track line.
                # cv2.polylines(frame_undist2, [line_list_arr], False, set_color_val, 2)
                # # show the primary trackID on the latest point
                # x = line_list_arr[-1][0][0]-5
                # y = line_list_arr[-1][0][1]-10
                # cv2.putText(frame_undist2, f"{trackID}", (x,y), cv2.FONT_HERSHEY_SIMPLEX, 0.35, text_trackID_color, 1,cv2.LINE_AA)

                # always show the start of the primary track:
                if cnt==0:
                    x = line_list_arr[0][0][0]
                    y = line_list_arr[0][0][1]
                    xoffs = int(8.5*len(str(trackID)))
                    cv2.rectangle(frame_undist2, (x-2, y-10), (x+xoffs, y+4), (0, 0, 0), -1)
                    cv2.putText(frame_undist2, f"{trackID}", (x,y), cv2.FONT_HERSHEY_SIMPLEX, 0.35, set_color_val, 1,cv2.LINE_AA)

                #retval = cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                # while cnt != track_frame_cnt:
                #     cap.grab()
                #     if cnt % 100 == 0:
                #         print(f"\r{cnt}", end="")
                #     cnt += 1
                
                # convert the x,y,z=0 track point back to pixels
                
        tval = frame_cnt/fps
    
        frame_cnt += 1
        #cv2.putText(frame_undist2, f"{frame_cnt}, time={tval:.2f}s", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
        cv2.rectangle(frame_undist2, (5, 5), (100, 40), (0, 0, 0), -1)
        cv2.putText(frame_undist2, f"{frame_cnt}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
        cv2.imshow('Video Tracks', frame_undist2)


        # option to save video clip and data
        if output_video:
            print(f"Writing frame {frame_cnt} to video file")
            vfout.write(frame_undist2)


    if output_video:
        vfout.release()
    cap.release()

    