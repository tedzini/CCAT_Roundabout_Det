##########################################################################
#                                                                        #
#  © 2026 Regents of the University of Minnesota. All rights reserved.   #
# This program is shared under the terms and conditions of the GNU       #
# Affero General Public License 3.0, License. Further details about      #
# the GNU Affero GPL 3.0 license are available in the LICENSE text file. #
#                                                                        #
##########################################################################
i
import cv2
import numpy as np
import os
import sys

def perspective_warp_with_zoom(img, M, height, width, rotation_matrix):

    # Perform the rotation
    img = cv2.warpAffine(img, rotation_matrix, (width, height))

    # Apply the combined transformation matrix to the frame
    warped_frame = cv2.warpPerspective(img, M, (640, 480))

    return warped_frame

def main(VIDEO_PATH_ROOT, VIDEO_NAME, OUT_PATH_ROOT):
    new_K = np.array([[240, 0, 320],
                    [0, 240, 240],
                    [0, 0, 1]])

    D1 = np.array([-0.07, 0.005, 0, 0])

    src_points = np.float32([[1000, 150], [2000, 150], [2000, 950], [1000, 950]])

    dst_points = np.float32([[0, 0], [640, 0], [640, 480], [0, 480]])

    # Calculate the perspective transform matrix
    M = cv2.getPerspectiveTransform(src_points, dst_points)


    # The angle of rotation in degrees
    angle_of_rotation = np.array([-20, -63, -105, -153, -198, -243, -292, -333])

    images = []
    images_raw = []
    titles = ['View_W', 'View_SW', 'View_S', 'View_SE', 'View_E', 'View_NE', 'View_N', 'View_NW']

    # Define video writers for the output videos for each view
    # Define video writers for the output videos for each view
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    
    print(f"Processing video: {VIDEO_PATH_ROOT}{VIDEO_NAME}!")
    filename =  filename = f"{VIDEO_NAME}"
    video_path = os.path.join(VIDEO_PATH_ROOT, filename)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f'Could not open {video_path} for capture!')
        sys.exit(-1)

    video_writers = {}

    video_prefix = VIDEO_NAME.split('.')[0]
    for title in titles:
        filename = f"{video_prefix}_out_{title}.avi"
        path = os.path.join(OUT_PATH_ROOT, filename)
        print(f'Creating output video: {path}!')
        video_writers[title] = cv2.VideoWriter(path, fourcc, 15.0, (640, 480))


    frame_num = 0
    
    
    for theta, title in zip(angle_of_rotation, titles):

        #print(title)

        if title in video_writers:
            video_writer = video_writers[title]
            print(video_writer)

        frame_num = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                if frame_num == 0:
                    # Get the image size
                    height, width = frame.shape[:2]

                    # Point to rotate around: the center of the frame
                    center_point = (width // 2, height // 2)

                    # Get the rotation matrix for rotating the image around its center
                    rotation_matrix = cv2.getRotationMatrix2D(center_point, theta, 1)

                img = perspective_warp_with_zoom(frame, M, height, width, rotation_matrix)

                undistorted_img = cv2.undistort(img, new_K, D1)
                    
                # this is where to enable on-line prediction (haven't actually tried)
                #YOLOpredictions = model.predict(source = undistorted_img, save=Fause)
                video_writer.write(undistorted_img)

                frame_num += 1
            else:
                break


        video_writer.release()
        cap.release()

if __name__ == "__main__":

    VIDEO_PATH_ROOT = sys.argv[1]
    VIDEO_NAME      = sys.argv[2]
    OUT_PATH_ROOT   = sys.argv[3]
#    VIDEO_PATH_ROOT = "videos"
#    OUT_PATH_ROOT = "videos/divided"
#    VIDEO_NAME = "Fri_2017-03-17_060002"

    main(VIDEO_PATH_ROOT, VIDEO_NAME, OUT_PATH_ROOT)
