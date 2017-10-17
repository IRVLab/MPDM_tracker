import sys
import numpy as np
import cv2
import os
import random
import argparse

lower = np.array([100, 10, 50], dtype = "uint8") #0,48,80
upper = np.array([130, 70, 100], dtype = "uint8") #20,255,255

image_hsv, im_inp = None, None
pixel = (20,60,80) # some stupid default

# mouse callback function
def pick_color(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        pixel = image_hsv[y,x]
        print(pixel)

        #im_out, _ = thresholdSKIN(im_inp)
        #cv2.imshow("output",im_out)
        #cv2.waitKey(200)



# Thanks to: github.com/sashagaz/Hand_Detection
def thresholdSKIN(current):
	
    blur = cv2.blur(current,(3,3)) #Blur the image
    hsv = cv2.cvtColor(blur,cv2.COLOR_BGR2HSV) #Convert to HSV color space 
    mask2 = cv2.inRange(hsv, lower, upper) # Create a binary thresholded image
    
    _, thresholded = cv2.threshold(mask2, 127, 255, 0)
    _, contours, _ = cv2.findContours(thresholded, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) 


    skin = cv2.bitwise_and(current, current, mask=thresholded)
    for i in range(len(contours)):
        cv2.drawContours(current, [contours[i]], -1, (0, 255, 255), 2)
    SegmentedSkin = np.hstack([current, skin])

    return skin, SegmentedSkin




if __name__ == '__main__':
    im_inp = cv2.imread('/home/xahid/datasets/flipper_tracking/olivia_flipper/cam_pool/58.jpg')
    im_inp = cv2.imread('/home/xahid/datasets/flipper_tracking/barbados_2007_flippers/flippers0019.jpg')
    im_inp = cv2.imread('/home/xahid/datasets/flipper_tracking/olivia_flipper/im_oliv_straight/31.jpg')
    im_inp = cv2.imread('/home/xahid/datasets/flipper_tracking/other4/real/04423_real.png')
    
    ## NEW ##
    
    cv2.namedWindow('hsv')
    image_hsv = cv2.cvtColor(im_inp,cv2.COLOR_BGR2HSV)
    cv2.imshow("hsv",image_hsv)
    cv2.setMouseCallback('hsv', pick_color)
    cv2.waitKey(0)
    

    '''
    im_out, _ = thresholdSKIN(im_inp)
    key=0
    while key != 27:
        cv2.imshow("output",im_out)
        key = cv2.waitKey(10)
    '''

    
    cv2.destroyAllWindows()
    






