import numpy as np
import cv2
import glob
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def chessboard_cameraCalibration(nx = 9, ny = 6):
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((ny*nx,3), np.float32)
    objp[:,:2] = np.mgrid[0:nx,0:ny].T.reshape(-1,2)

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d points in real world space
    imgpoints = [] # 2d points in image plane.

    # Make a list of calibration images
    images = glob.glob('camera_cal/calibration*.jpg')

    # Step through the list and search for chessboard corners
    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (nx,ny),None)

        # If found, add object points, image points
        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)

            # Draw and display the corners
            # img = cv2.drawChessboardCorners(img, (nx,ny), corners, ret)
            # cv2.imshow('img',img)
            # cv2.waitKey(500)

    cv2.destroyAllWindows()
    return objpoints,imgpoints

def distortion_correction(fname, objpoints, imgpoints):
    img = cv2.imread(fname)
    img_size = (img.shape[1],img.shape[0])

    ret,mtx,dist,rvecs,tvecs = cv2.calibrateCamera(objpoints,imgpoints,img_size,None,None)
    dst = cv2.undistort(img,mtx,dist,None,mtx)
    cv2.imwrite('output_images/undist_'+fname,dst)
    # mpimg.imsave(os.path.join('output_images/undist', fname),dst)

    return dst

def distortion_correction2(img, objpoints, imgpoints):
    img_size = (img.shape[1],img.shape[0])

    ret,mtx,dist,rvecs,tvecs = cv2.calibrateCamera(objpoints,imgpoints,img_size,None,None)
    dst = cv2.undistort(img,mtx,dist,None,mtx)
    # cv2.imwrite('output_images/undist_'+fname,dst)
    # mpimg.imsave(os.path.join('output_images/undist', fname),dst)

    return dst


def vis_images(img1,img2,img1title,img2title):
    f, (ax1,ax2) = plt.subplots(1,2, figsize = (20,10))
    ax1.imshow(img1)
    ax1.set_title(img1title, fontsize = 30)
    ax2.imshow(img2)
    ax2.set_title(img2title, fontsize = 30)
    plt.show()
 
def binary_image(undist_img,sx_thresh =(60,170), s_thresh =(200,255)):
    # binary image after applying color and gradient methods
    img = np.copy(undist_img)
    hls = cv2.cvtColor(img,cv2.COLOR_BGR2HLS)
    l_channel = hls[:,:,1]
    s_channel = hls[:,:,2]
    # sobel x
    sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1,0)
    scaled_sobel = np.uint8(255* np.absolute(sobelx)/np.max(np.absolute(sobelx)))
    #threshold x gradient
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1]) ] = 1
    # plt.imshow(sxbinary, cmap = 'gray')
    # plt.show()

    #Threshold color channel 
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1
    # plt.imshow(s_binary, cmap = 'gray')
    # plt.show()
    #stack each channel
    color_binary = np.dstack((np.zeros_like(s_binary),sxbinary,s_binary))*255
    # plt.imshow(color_binary)
    # plt.show()
    # combine both the binary images
    combined_binary = np.zeros_like(s_binary)
    combined_binary[(s_binary == 1) | (sxbinary == 1) ] = 1
    # plt.imshow(combined_binary, cmap = 'gray')
    # plt.show()
    return combined_binary

def warper(img):
    # Compute and apply perpective transform
    img_size = (img.shape[1], img.shape[0])
    src = np.float32(
    [[(img_size[0] / 2) - 50, img_size[1] / 2 + 100],
    [((img_size[0] / 6) - 20), img_size[1]],
    [(img_size[0] * 5 / 6) + 60, img_size[1]],
    [(img_size[0] / 2 + 55), img_size[1] / 2 + 100]])
    dst = np.float32(
    [[(img_size[0] / 4), 0],
    [(img_size[0] / 4), img_size[1]],
    [(img_size[0] * 3 / 4), img_size[1]],
    [(img_size[0] * 3 / 4), 0]])
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_NEAREST)  # keep same size as input image
    # plt.imshow(warped)
    # plt.show()
    return warped

def get_Minv(img):
    # Compute and apply perpective transform
    img_size = (img.shape[1], img.shape[0])
    src = np.float32(
    [[(img_size[0] / 2) - 50, img_size[1] / 2 + 100],
    [((img_size[0] / 6) - 20), img_size[1]],
    [(img_size[0] * 5 / 6) + 60, img_size[1]],
    [(img_size[0] / 2 + 55), img_size[1] / 2 + 100]])
    dst = np.float32(
    [[(img_size[0] / 4), 0],
    [(img_size[0] / 4), img_size[1]],
    [(img_size[0] * 3 / 4), img_size[1]],
    [(img_size[0] * 3 / 4), 0]])
    Minv = cv2.getPerspectiveTransform(dst, src)
    return Minv



def hist(img):
    bottom_half = img[img.shape[0]//2:,:]
    histogram = np.sum(bottom_half,axis = 0)
    return histogram

def get_x_indices(img, start_y,window_height):
    window = img[start_y:start_y+window_height,img.shape[1]//6:img.shape[1]*5//6]
    histogram = np.sum(window, axis = 0)
    mid_point = np.int((histogram.shape[0])//2)
    leftx_base = np.argmax(histogram[:mid_point])+ img.shape[1]//6
    rightx_base = np.argmax(histogram[mid_point:])+mid_point + img.shape[1]//6
    return leftx_base, rightx_base

def measure_curvature(ploty, left_fit,right_fit):
    '''
    Calculates the curvature of polynomial functions in pixels.
    '''
      # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension

    # Define y-value where we want radius of curvature
    # We'll choose the maximum y-value, corresponding to the bottom of the image
    y_eval = np.max(ploty)
    
    # Calculation of R_curve (radius of curvature)
    left_curverad = ((1 + (2*left_fit[0]*y_eval*ym_per_pix + left_fit[1])**2)**1.5) / np.absolute(2*left_fit[0])
    right_curverad = ((1 + (2*right_fit[0]*y_eval*ym_per_pix + right_fit[1])**2)**1.5) / np.absolute(2*right_fit[0])
    mean_curvature = (left_curverad + right_curverad)/2

    return left_curverad, right_curverad
def roughly_parallel(left_curverad, right_curverad):
    if((left_curverad > 1000) & (right_curverad > 1000)& (np.absolute(left_curverad-right_curverad)/1000 >3) ):
        return True
    else:
        return False

# input is binary warped image
def find_lane_pixels(binary_warped, nwindows=9,margin = 150,minpix = 50):
    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
    # Create an output image to draw on and visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]//2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint


    # Set height of windows - based on nwindows above and image shape
    window_height = np.int(binary_warped.shape[0]//nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated later for each window in nwindows
    leftx_current = leftx_base
    rightx_current = rightx_base

    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        ### TO-DO: Find the four below boundaries of the window ###
        win_xleft_low = leftx_current-margin  # Update this
        win_xleft_high = leftx_current+margin  # Update this
        win_xright_low = rightx_current-margin  # Update this
        win_xright_high = rightx_current+margin  # Update this
        
        # Draw the windows on the visualization image
        # cv2.rectangle(out_img,(win_xleft_low,win_y_low),
        # (win_xleft_high,win_y_high),(0,255,0), 2) 
        # cv2.rectangle(out_img,(win_xright_low,win_y_low),
        # (win_xright_high,win_y_high),(0,255,0), 2) 
        
        ### TO-DO: Identify the nonzero pixels in x and y within the window ###
        good_left_inds = ((nonzeroy >=win_y_low) & (nonzeroy<win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox< win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >=win_y_low) & (nonzeroy<win_y_high) & (nonzerox >= win_xright_low) & (nonzerox< win_xright_high)).nonzero()[0]
        
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        
        ### TO-DO: If you found > minpix pixels, recenter next window ###
        ### (`right` or `leftx_current`) on their mean position ###
        if len(good_left_inds) & len(good_right_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))
        else:
            leftx_current, rightx_current = get_x_indices(binary_warped, win_y_high,window_height)

    # Concatenate the arrays of indices (previously was a list of lists of pixels)
    try:
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
    except ValueError:
        # Avoids an error if the above is not implemented fully
        pass

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    return leftx, lefty, rightx, righty, out_img

def fit_polynomial(binary_warped):
    # Find our lane pixels first
    leftx, lefty, rightx, righty, out_img = find_lane_pixels(binary_warped)

    ### TO-DO: Fit a second order polynomial to each using `np.polyfit` ###
    left_fit = np.polyfit(lefty,leftx,2)
    right_fit = np.polyfit(righty,rightx,2)

    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    try:
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    except TypeError:
        # Avoids an error if `left` and `right_fit` are still none or incorrect
        print('The function failed to fit a line!')
        left_fitx = 1*ploty**2 + 1*ploty
        right_fitx = 1*ploty**2 + 1*ploty
    left_curverad, right_curverad = measure_curvature(ploty, left_fit,right_fit)
    # print("left curvature", left_curverad, "  right curvature ", right_curverad)
    if not roughly_parallel(left_curverad, right_curverad):
        print("Error in polynomial fitting")
    ## Visualization ##
    # Colors in the left and right lane regions
    out_img[lefty, leftx] = [255, 0, 0]
    out_img[righty, rightx] = [0, 0, 255]

    # Plots the left and right polynomials on the lane lines
    # plt.plot(left_fitx, ploty, color='yellow')
    # plt.plot(right_fitx, ploty, color='yellow')
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(binary_warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))  
    # recast the x and y points into usable format for cv2. fill poly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx,ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx,ploty])))])
    pts = np.hstack((pts_left,pts_right))
    # Draw lane onto the warped blank image
    cv2.fillPoly(color_warp,np.int_([pts]),(0,255,0))

    result = cv2.addWeighted(out_img, 1, color_warp, 0.3, 0)

    # plt.imshow(result)
    # plt.show()

    return result, left_fit, right_fit



def fit_poly(img_shape, leftx, lefty, rightx, righty):
    ### TO-DO: Fit a second order polynomial to each with np.polyfit() ###
    left_fit = np.polyfit(lefty,leftx,2)
    right_fit = np.polyfit(righty,rightx,2)
    # Generate x and y values for plotting
    ploty = np.linspace(0, img_shape[0]-1, img_shape[0])
    ### TO-DO: Calc both polynomials using ploty, left_fit and right_fit ###
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    
    left_curverad, right_curverad = measure_curvature(ploty, left_fit,right_fit)
    print("left curvature", left_curverad, "  right curvature ", right_curverad)
    if roughly_parallel(left_curverad, right_curverad):
        ret = True
    else: 
        print("Error in polynomial fitting")
        ret = False
    return left_fitx, right_fitx, ploty, ret

# left_fit and right_fit are polynomial from previous step
def search_around_poly(binary_warped, left_fit,right_fit):
    # HYPERPARAMETER
    # Choose the width of the margin around the previous polynomial to search
    # The quiz grader expects 100 here, but feel free to tune on your own!
    margin = 100

    # Grab activated pixels
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    
    ### TO-DO: Set the area of search based on activated x-values ###
    ### within the +/- margin of our polynomial function ###
    ### Hint: consider the window areas for the similarly named variables ###
    ### in the previous quiz, but change the windows to our new search area ###
    left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + 
                    left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + 
                    left_fit[1]*nonzeroy + left_fit[2] + margin)))
    right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + 
                    right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + 
                    right_fit[1]*nonzeroy + right_fit[2] + margin)))
    
    
    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Fit new polynomials
    left_fitx, right_fitx, ploty, ret = fit_poly(binary_warped.shape, leftx, lefty, rightx, righty)
    
    ## Visualization ##
    # Create an image to draw on and an image to show the selection window
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    window_img = np.zeros_like(out_img)
    # Color in left and right line pixels
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

    # Generate a polygon to illustrate the search window area
    # And recast the x and y points into usable format for cv2.fillPoly()
        # left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
        # left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin, 
        #                           ploty])))])
        # left_line_pts = np.hstack((left_line_window1, left_line_window2))
        # right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
        # right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin, 
        #                           ploty])))])
        # right_line_pts = np.hstack((right_line_window1, right_line_window2))

        # # # Draw the lane onto the warped blank image
        # cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
        # cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))

    # Create an image to draw the lines on
    warp_zero = np.zeros_like(binary_warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))        

    # recast the x and y points into usable format for cv2. fill poly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx,ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx,ploty])))])
    pts = np.hstack((pts_left,pts_right))
    # Draw lane onto the warped blank image
    cv2.fillPoly(color_warp,np.int_([pts]),(0,255,0))

    result = cv2.addWeighted(out_img, 1, color_warp, 0.3, 0)

    # # Plot the polynomial lines onto the image
    # plt.plot(left_fitx, ploty, color='yellow')
    # plt.plot(right_fitx, ploty, color='yellow')
    
    ## End visualization steps ##
    # plt.imshow(result)
    # plt.show()
    left_fit = np.polyfit(lefty,leftx,2)
    right_fit = np.polyfit(righty,rightx,2)
        
    return result, left_fit, right_fit, ret

def pipeline(fname, objpoints, imgpoints):
    undst = distortion_correction(fname, objpoints, imgpoints)
    # binary image of undistorted image
    combined_binary = binary_image(undst)
    # perspective transform the region of interest
    warped = warper(combined_binary)
    # fit the polynomial curve on the lanes 
    poly_fit_image, left_fit, right_fit = fit_polynomial(warped)
    # leftx, lefty, rightx, righty, out_img = find_lane_pixels(warped)
    # left_fit = np.polyfit(lefty,leftx,2)
    # right_fit = np.polyfit(righty,rightx,2)
    # poly_fit_image = search_around_poly(warped,left_fit, right_fit)
    # plot it back on the main image

    # warp the black to original image space using inverse perspective matrix Minv
    Minv = get_Minv(undst)
    newwarp = cv2.warpPerspective(poly_fit_image, Minv, (undst.shape[1],undst.shape[0]))
    result = cv2.addWeighted(undst, 1, newwarp, 0.8, 0)

    plt.imshow(result)
    plt.show()






# Define a class to receive the characteristics of each line detection
class Line():
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False  
        # x values of the last n fits of the line
        self.recent_xfitted = [] 
        #average x values of the fitted line over the last n iterations
        self.bestx = None     
        #polynomial coefficients averaged over the last n iterations
        self.best_fit = None  
        #polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]  
        #radius of curvature of the line in some units
        self.radius_of_curvature = None 
        #distance in meters of vehicle center from the line
        self.line_base_pos = None 
        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float') 
        #x values for detected line pixels
        self.allx = None  
        #y values for detected line pixels
        self.ally = None 
