import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

N_x, N_y = 16, 21
# Our 2-dimensional distribution will be over variables X and Y
N = N_x*N_y
X = np.arange(0, N_x, 1)
Y = np.arange(0, N_y, 1)
X, Y = np.meshgrid(X, Y)

# Mean vector and covariance matrix
mu = np.array([5., 6.])
Sigma = np.array([[ 4. , 0], [0,  4]])

# Pack X and Y into a single 3-dimensional array
pos = np.empty(X.shape + (2,))
pos[:, :, 0] = X
pos[:, :, 1] = Y



#Return the multivariate Gaussian distribution on array pos
def multivariate_gaussian(pos, mu, Sigma):
    d = mu.shape[0]
    Sigma_det = np.linalg.det(Sigma)
    Sigma_inv = np.linalg.inv(Sigma)
    N_normalizer = np.sqrt((2*np.pi)**d * Sigma_det)
    fac = np.einsum('...k,kl,...l->...', pos-mu, Sigma_inv, pos-mu) #(x-mu)T.Sigma-1.(x-mu)

    return np.exp(-fac / 2) / N_normalizer

# The distribution on the variables X, Y packed into pos.
Z = multivariate_gaussian(pos, mu, Sigma)

print(Z[6, 5], Z[5, 6])
print(np.max(Z))

# Create a surface plot and projected filled contour plot under it.
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_surface(X, Y, Z, rstride=3, cstride=3, linewidth=1, antialiased=True, cmap=cm.viridis)
cset = ax.contourf(X, Y, Z, zdir='z', offset=-0.15, cmap=cm.viridis)

# Adjust the limits, ticks and view angle
ax.set_zlim(-0.15,0.2)
ax.set_zticks(np.linspace(0,0.2,5))
ax.view_init(27, -21)

plt.show()