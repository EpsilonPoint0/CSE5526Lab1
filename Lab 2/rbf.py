import numpy as np

# 1) Generate a set of 100 data points by sampling the function LaTeX: h\left(x\right)=0.5+0.4\cos\left(2.5\pi x\right) with added uniform noise in the interval [-0.1, 0.1] and with x values taken randomly
#  from a uniform distribution in the interval [0.0, 1.0].

# 2) Determine Gaussian centers by the K-means algorithm, and set the variance of each cluster according to the variance of the cluster. If a cluster contains only one sample point, use as its variance the mean 
# variance of all the other clusters. 

class RBF:

    def __init__(self, bases):
        self.bases = bases
        noise = np.random.uniform(low=-0.1, high=0.1, size=(100, 1))
        initial_x_vals = np.random.uniform(low=0, high=1.0, size=(100, 1))
        inputs = []
        for i in range(0, 100):
            x_val = 0.5 + 0.4 * np.cos(2.5 * np.pi * initial_x_vals[i]) + noise[i]
            print(x_val)
            inputs.append(x_val[0])
        self.inputs = inputs
        #print(pts)

        def k_means(self, num_clusters):
            # Generate random clusters from points
            current_clusters = np.random.choice(np.squeeze(self.inputs, size=num_clusters))
            updated_clusters = current_clusters.copy()
            # Initialize standard deviations to 0
            standard_deviations = np.zeros(num_clusters)

            converged = False

            while not converged:
                empty_or_sinlge_point_clusters = []
                distance_to_clusters = np.squeeze(np.abs(self.inputs[:, np.newaxis] - current_clusters[np.newaxis, :]))

                closest_cluster_to_points = np.argmin(distance_to_clusters, axis=1)

                current_clusters, points_in_cluster = update_clusters(closest_cluster_to_points, current_clusters, num_clusters)
                converged = np.linalg.norm(current_clusters - current_clusters) < 0.000001

                updated_clusters = current_clusters.copy

        def update_clusters(closest_cluster_to_points, initial_clusters, num_clusters):
            for i in range(0, num_clusters):
                points_in_cluster = self.inputs[closest_cluster_to_points == i]
                if len(points_in_cluster) > 0:
                    initial_clusters[i] = np.mean(points_in_cluster, axis=0)
            return initial_clusters, points_in_cluster
            


rbf = RBF(3)