#import roslib; #roslib.load_manifest('mpdm')
import os
import sys
import cv2
import rospy
import argparse
import numpy as np
import matplotlib.pyplot as plt

from mdpm.getProspectiveWindows import ImageProcessor


'''
Initiates calls to start mpdm. 
image_streamimg and video_streamimg are for offline testing on 
image sequences and video files, respectively.
'''
class Starter:
    def __init__(self, FlipperColor, slide_size, win_size):
        self.im_proc = ImageProcessor(FlipperColor, slide_size, win_size)

    def image_streamimg(self, Dir_):
        dirFiles = os.listdir(Dir_)
        dirFiles.sort(key=lambda f: int(filter(str.isdigit, f)))

        for filename in dirFiles:
            frame = cv2.imread(Dir_+filename)
            self.original= cv2.resize(frame, (640, 480))
            self.im_proc.ContdFrame(self.original)


    def video_streamimg(self, vid_src):
        cap = cv2.VideoCapture(vid_src)
        while(cap.isOpened()):
            self.ret, frame = cap.read()
            if frame is not None :
                self.original= cv2.resize(frame, (640, 480))
                self.im_proc.ContdFrame(self.original)
            else:
                cap.release()
                cv2.destroyAllWindows()


    def show_frame(self, name, im):
        cv2.imshow(name, im)
        cv2.waitKey(30)

    
