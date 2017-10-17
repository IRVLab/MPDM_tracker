import cv2
import numpy as np


'''
An HMM-based model to keep track of few most promising motion directions
this keeps the volume tractable by pruning out unimportant motion directions
'''
class HMM_model():
	def __init__(self, n_Ww, n_Wh):
		self.noWin_w, self.noWin_h = n_Ww, n_Wh

		# For 2-dimensional distribution will be over variables X and Y
		self.X, self.Y = np.meshgrid(np.arange(0, n_Wh, 1), np.arange(0, n_Ww, 1))
		self.Sigma = np.array([[ 4. , 0], [0,  4]]) # covariance matrix

		# Pack X and Y into a single 3-dimensional array
		self.pos = np.empty(self.X.shape + (2,))
		self.pos[:, :, 0], self.pos[:, :, 1] = self.X, self.Y


	# Return the multivariate Gaussian distribution on array pos
	# mu: mean
	def multivariate_gaussian(self, mu):
		d = mu.shape[0]
		Sigma_det = np.linalg.det(self.Sigma)
		Sigma_inv = np.linalg.inv(self.Sigma)
		N_normalizer = np.sqrt((2*np.pi)**d * Sigma_det)
		fac = np.einsum('...k,kl,...l->...', self.pos-mu, Sigma_inv, self.pos-mu) #(x-mu)T.Sigma-1.(x-mu)

		return np.exp(-fac / 2) / N_normalizer



	# Equation (6) of the paper
	# P(e|w) = (1-eps) if intensity of the window falls within the thresholded region
	# P(e|w) = (eps)   otherwise
	def P_EvidenceGivenState(self, internsityVect, eps):
		tempIDX = np.argsort(internsityVect)
		radius = 20

		filteredIDX = tempIDX[-radius:]
		proba = np.ones(internsityVect.shape[0]) * eps
		proba[filteredIDX] = 1-eps

		return proba


	# Equation (5) of the paper (we use a 2d Gaussian here)
	# P(w_new|w_old) 
	def P_CurrStateGivenPrevState(self, prevState, P_eviGivenState):
		proba = np.ones(self.noWin_w * self.noWin_h)
		nextS = np.zeros(prevState.shape[0])
		nextProba = np.ones(prevState.shape[0])

		for idx in range(nextS.shape[0]):
			idx_x, idx_y = (prevState[idx]%self.noWin_h), (prevState[idx]%self.noWin_w) 
			mu = np.array([idx_x, idx_y])
			Z = self.multivariate_gaussian(mu)

			for j in range(proba.shape[0]):
				j_x, j_y = (j%self.noWin_h), (j%self.noWin_w)
				proba[j] = Z[j_y, j_x] * P_eviGivenState[j]

			nextS[idx], nextProba[idx] = np.argmax(proba), np.max(proba) 

		#print(nextS)
		return nextS, nextProba


	# Resample with a selection probability p_select
	# currPotState: current potential states to select from
	# outputs selected next pool of states
	def ResampleWithProba(self, internsityVect, currPotState, p_select):
		tempIDX = np.argsort(internsityVect)
		filteredIDX = tempIDX[-currPotState.shape[0]:]
		selection_proba = np.random.uniform(0, 1, currPotState.shape[0])
		nextState = currPotState
		for p in range(currPotState.shape[0]):
			if selection_proba[p] >= p_select:
				nextState[p] = filteredIDX[p]

		return nextState
        












           




