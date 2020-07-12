import numpy as np
import cv2
import glob
import os
import matplotlib.pyplot as plt
from helper import * 
from moviepy.editor import VideoFileClip
from IPython.display import HTML
# calibrating camera
global objpoints
global imgpoints
objpoints, imgpoints = chessboard_cameraCalibration()
# undistort the checssboard and check how it looks 
# undst = distortion_correction('camera_cal/calibration1.jpg', objpoints, imgpoints)
# vis_images(cv2.imread('camera_cal/calibration1.jpg'), undst,'Original_Image', 'undistorted_Image')
# apply distortion correction to all images and save them 
# images = glob.glob('test_images/*.jpg')
# for fname in images:
# # fname = 'test_images/test5.jpg'
#     pipeline(fname, objpoints, imgpoints)


check = False

def video_pipeline(img):
    global check
    global left_fitt, right_fitt
    undst = distortion_correction2(img, objpoints, imgpoints)
    # binary image of undistorted image
    combined_binary = binary_image(undst)
    # perspective transform the region of interest
    warped = warper(combined_binary)
    # fit the polynomial curve on the lanes 
    if check == False:
        poly_fit_image,left_fitt, right_fitt = fit_polynomial(warped)
        check = True
    else:
        poly_fit_image, left_fitt, right_fitt, check = search_around_poly(warped,left_fitt, right_fitt)
    # plot it back on the main image

    # warp the black to original image space using inverse perspective matrix Minv
    Minv = get_Minv(undst)
    newwarp = cv2.warpPerspective(poly_fit_image, Minv, (undst.shape[1],undst.shape[0]))
    result = cv2.addWeighted(undst, 1, newwarp, 0.8, 0)

    # plt.imshow(result)
    # plt.show()
    return result

# dir2 = os.listdir("test_videos/")
# for file in dir2:
file = "project_video.mp4"
white_output = os.path.join('test_videos_output/', file) #'test_videos_output/challenge.mp4' #
clip1 = VideoFileClip(os.path.join('test_videos/', file)).subclip(20,30)#'test_videos/challenge.mp4')#

white_clip = clip1.fl_image(video_pipeline) #NOTE: this function expects color images!!
# %time 
white_clip.write_videofile(white_output, audio=False)
HTML("""
<video width="960" height="540" controls>
<source src="{0}">
</video>
""".format(white_output))




# undst = distortion_correction('test_images/test5.jpg', objpoints, imgpoints)
# plt.imshow(undst)
# plt.show()


