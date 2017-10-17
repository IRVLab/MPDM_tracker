import numpy as np

class HeiCluster():
	def __init__(self, thr):
		self.clusters = []
		self.cluster_sizes = []
		self.threshD = thr


	def getClusters(self, M_2d):
		x_curr = M_2d[0, :]
		c_curr = []
		for i in range(M_2d.shape[0]):
			if (np.linalg.norm(x_curr-M_2d[i, :]) <= self.threshD):
				c_curr.append(M_2d[i, :])
				x_curr = np.mean(np.array(c_curr), 0)
			else:
				self.clusters.append(np.array(c_curr))
				self.cluster_sizes.append(len(c_curr))
				c_curr = [M_2d[i, :]]
				x_curr = M_2d[i, :]
		if len(c_curr)>0:
			self.clusters.append(np.array(c_curr))
			self.cluster_sizes.append(len(c_curr))

		return self.clusters, self.cluster_sizes