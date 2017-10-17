import cv2
import numpy as np
import scipy.fftpack

'''
Given a spatio-temporal volume (as referred in the paper),
searches for the motion direction that frequency-doman signatures
pertaining to human swimming pattern (i.e., high spikes in 1-2Hz)
'''
class searchMotionDirection():
    def __init__(self):
        self.freqDelta = 2e2  # aplitude threshold
        self.freqFlipper = np.array([1, 2]) # divers' flipping freq ~1-2Hz


    # M_inten: Intensity volume, state_seq: state volume
    # returns output states that lie in 'good' motion directions
    def sweepSpatioTemporalVolume(self, M_inten, state_seq):
    	freq_resp = np.zeros((state_seq.shape), dtype=complex)
    	amp_selected = np.zeros(state_seq.shape[0])
    	v_xyt = np.zeros((state_seq.shape))

    	# test frequency response of every motion direction
    	for i in range(state_seq.shape[0]):
    		for j in range(state_seq.shape[1]):
    			v_xyt[i, j] = M_inten[int(state_seq[i, j]), j] # spatio-temporal domain

    		freq_resp[i, :] = scipy.fftpack.fft(v_xyt[i, :]) # frequency domain
    		amp_selected[i] = (np.abs(freq_resp[i, self.freqFlipper[0]]) + np.abs(freq_resp[i, self.freqFlipper[1]]))/2 
    		
    	output_state = state_seq[np.where(amp_selected>self.freqDelta), -1].reshape(-1)

        return output_state




