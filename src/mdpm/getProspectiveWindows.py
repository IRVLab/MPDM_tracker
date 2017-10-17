import os
import cv2
import numpy as np
from freqDomainComputation import searchMotionDirection
from modelHMM import HMM_model

'''
Gets the image, performs spatial domain thresholding based on the volor of flippers
Then spatio-temporal volume is generated and updated with each frame (as mentioned in paper)
HMM model selects few most promising motion direction (in terms of sequence of windows)
Then frequency-domain features are tested for actual motion directions
'''
class ImageProcessor():
    def __init__(self, FlipperColor, slide_size, win_size):
        self.imW, self.imH = 640, 480  # image size (resized)
        self.potWinSize = 10 # potential windows per frame
        # flipper color, sliding window size, and rectangular window size
        self.FlipperColor, self.slide_size, self.win_size = FlipperColor, slide_size, win_size

        self.initWindows()  # other initializations
        self.getHSVThresholds() # HSV-space thresholding 
        self.hmm_trace = HMM_model(self.noWin_w, self.noWin_h) # for HMM-based pruning
        self.MDirection = searchMotionDirection() # for frequency-domain analysis


    # for checking different frames
    def showFrame(self, name, frame):
        cv2.imshow(name, frame)
        cv2.waitKey(50)


    # initialize different data structures
    def initWindows(self):
        self.noWin_w, self.noWin_h = (self.imW/self.win_size), (self.imH/self.win_size)
        self.no_of_wins = self.noWin_w * self.noWin_h

        # window number to image-space coordinates
        self.win_to_xy = []
        for x in range(self.noWin_w): # column major numbering
            for y in range(self.noWin_h ):
                self.win_to_xy.append((x*self.win_size, y*self.win_size))

        self.current_pointer = 0  # pointer in the sliding window
        self.IntensityValues = np.zeros((self.no_of_wins, self.slide_size)) # intensity volume
        self.PotWindowSeq = np.zeros((self.potWinSize, self.slide_size)) # potential states
        self.currentProba = np.ones(self.potWinSize) # running probability (proba of sequence of states)



    # getting the upper and lower HSV-space thresholding based on flipper's colors 
    # see the hsv_flipper.txt file for hsv ranges
    def getHSVThresholds(self):
        with open(os.getcwd()+'/src/mdpm_flipper/src/data/hsv_flipper.txt') as f:
            stream_ = f.readlines()
            for i in range(1, len(stream_)):
                token_ = stream_[i].split(' ')
                if (token_[0]==self.FlipperColor):
                    self.lower = np.array([int(token_[1]), int(token_[2]), int(token_[3])], dtype = "uint8")
                    self.upper = np.array([int(token_[4]), int(token_[5]), int(token_[6])], dtype = "uint8")

                    
    
    # initially, predict potential windows based on intensity values only
    def initPotWins(self):
        tempINT = self.IntensityValues[:, self.current_pointer]
        tempIDX = np.argsort(tempINT)
        filteredIDX = tempIDX[-self.potWinSize:]
        self.PotWindowSeq[:, self.current_pointer] = filteredIDX



    # show the potential windows
    def drawPotWins(self):
        for id in self.PotWindowSeq[:, self.current_pointer]:
            rect_x, rect_y = self.win_to_xy[int(id)]
            cv2.rectangle(self.current, (rect_x, rect_y), (rect_x+self.win_size,rect_y+self.win_size), (0, 255, 255), -1)


    # show the final windows
    def drawFinalWins(self):
        for id in self.finalWinfowSeq:
            rect_x, rect_y = self.win_to_xy[int(id)]
            cv2.rectangle(self.current, (rect_x, rect_y), (rect_x+self.win_size,rect_y+self.win_size), (0, 0, 255), -1)
        

    # show the spatial position of the windows
    def drawLines(self):
        for x in range(self.noWin_w+1):
            cv2.line(self.current, (x*self.win_size, 0), (x*self.win_size, self.noWin_h*self.win_size), (0, 0,0), 1)
        for y in range(self.noWin_h+1):
            cv2.line(self.current, (0, y*self.win_size), (self.noWin_w*self.win_size, y*self.win_size), (0, 0, 0), 1)


    # generate and update the spatio-temporal volume
    def captureItensity(self):
        temp_intensity_vect = []
        # get the mean intensity values for all windows
        for x in range(self.noWin_w): 
            for y in range(self.noWin_h ):
                temp_rect = self.thresholded[y*self.win_size:(y+1)*self.win_size, x*self.win_size:(x+1)*self.win_size]
                temp_intensity_vect.append(np.mean(temp_rect))               

        # shift columns for new data (if needed)
        if (self.current_pointer > self.slide_size-1):
            self.current_pointer = self.slide_size-1
            self.IntensityValues[:, 0:self.current_pointer] = self.IntensityValues[:, np.array(range(1, self.current_pointer+1))]
            self.IntensityValues[:, self.current_pointer] = np.array(temp_intensity_vect)

            self.PotWindowSeq[:, 0:self.current_pointer] = self.PotWindowSeq[:, np.array(range(1, self.current_pointer+1))] 
        else:
            self.IntensityValues[:, self.current_pointer] = np.array(temp_intensity_vect)

         

    # HSV-space thresholding based on flipper's colors
    def thresholdING(self):
        grey = cv2.cvtColor(self.current, cv2.COLOR_BGR2GRAY)
        self.blurredGrey = cv2.blur(grey, (5, 5)) #Blur the image
        blur = cv2.blur(self.current, (3, 3)) #Blur the image
        
        hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV) #Convert to HSV color space 
        mask2 = cv2.inRange(hsv, self.lower, self.upper) # Create a binary thresholded image
        _, self.thresholded = cv2.threshold(mask2, 127, 255, 0)



    # gets new frame and performs the tasks in order
    def ContdFrame(self, im):
        self.current = im
        self.thresholdING()
        self.drawLines()
        self.captureItensity()

        if (self.current_pointer == 0):
            self.initPotWins()
        else:
            tempINT = self.IntensityValues[:, self.current_pointer]
            prevPotState = self.PotWindowSeq[:, self.current_pointer-1]
            P_eviGivenState = self.hmm_trace.P_EvidenceGivenState(tempINT, 0.2)
            currPotState, cumProba = self.hmm_trace.P_CurrStateGivenPrevState(prevPotState, P_eviGivenState)
            self.PotWindowSeq[:, self.current_pointer] = self.hmm_trace.ResampleWithProba(tempINT, currPotState, 0.5)
            self.currentProba = cumProba * 1e4 # for numerical stability

        if (self.current_pointer >= self.slide_size-1):
            self.finalWinfowSeq = self.MDirection.sweepSpatioTemporalVolume(self.IntensityValues, self.PotWindowSeq) # better sweep here
            self.drawFinalWins()
        
        #self.drawPotWins()
        self.showFrame('original', self.current)
        self.current_pointer += 1