import numpy as np

#Return the multivariate Gaussian distribution on array pos
def multivariate_gaussian(pos, mu, Sigma):
    d = mu.shape[0]
    Sigma_det = np.linalg.det(Sigma)
    Sigma_inv = np.linalg.inv(Sigma)
    N_normalizer = np.sqrt((2*np.pi)**d * Sigma_det)
    fac = np.einsum('...k,kl,...l->...', pos-mu, Sigma_inv, pos-mu) #(x-mu)T.Sigma-1.(x-mu)

    return np.exp(-fac / 2) / N_normalizer


pos = []
pos.append(np.array([0, 0]))
pos.append(np.array([4, 4]))

mu = np.array([4, 4])
Sigma = np.array([[ 2 , 0], [0,  2]])
Z = multivariate_gaussian(np.array(pos), mu, Sigma)
print (Z/np.max(Z))