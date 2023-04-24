import cv2
from scipy.spatial.distance import pdist
from scipy.cluster import hierarchy
import numpy as np
import matplotlib.pyplot as plt
import glob


class FeatureMatching:
    def __init__(self, kp, desc, norm=cv2.NORM_L2):
        self.kp = kp
        self.desc = desc
        self.norm = norm

    def match(self):
        # Type of image normalization
        self.norm = cv2.NORM_L2
        # number of closest match we want to find for each descriptor
        k = 10
        # uses a brute force matcher(compare each descriptor of desc1, with each descriptor of desc2...)
        brute_force = cv2.BFMatcher(self.norm)
        # finds closest matches for each desc in desc1 with desc in desc2
        matches = brute_force.knnMatch(self.desc, self.desc, k)

        print("Amount of matches: " + str(len(matches)))

        # Ratio test
        good1, good2 = [], []

        for match in matches:
            i = 1

            while match[i].distance < 0.75 * match[i+1].distance:
                i = i + 1

            for k in range(1, i):
                temp = match[k]

                if pdist(np.array([self.kp[temp.queryIdx].pt,
                                   self.kp[temp.trainIdx].pt])) > 10:
                    good1.append(self.kp[temp.queryIdx])
                    good2.append(self.kp[temp.trainIdx])

        points1 = [m.pt for m in good1]
        points2 = [m.pt for m in good2]

        if len(points1) > 0:
            points = np.hstack((points1, points2))  # column bind
            unique_points = np.unique(points, axis=0)  # remove any duplicated points
            points1, points2 = np.float32(unique_points[:, 0:2]), np.float32(unique_points[:, 2:4])
            return points1, points2
        else:
            return None, None

    def hierarchical_clustering(self, points_1, points_2, metric, threshold):
        points = np.vstack((points_1, points_2))         # vertically stack both sets of points (row bind)
        dist_matrix = pdist(points, metric='euclidean')  # obtain condensed distance matrix (needed in linkage function)
        z = hierarchy.linkage(dist_matrix, metric)

        # perform agglomerative hierarchical clustering
        cluster = hierarchy.fcluster(z, t=threshold, criterion='inconsistent', depth=4)
        # filter outliers
        # cluster, points = filterOutliers(cluster, points)

        n = int(np.shape(points)[0]/2)
        return cluster, points[:n], points[n:]

    def plot_image(self, img, p1, p2, c):
        plt.imshow(img)
        plt.axis('off')

        colors = c[:np.shape(p1)[0]]
        plt.scatter(p1[:, 0], p1[:, 1], c=colors, s=30)

        for coord1, coord2 in zip(p1, p2):
            x1 = coord1[0]
            y1 = coord1[1]

            x2 = coord2[0]
            y2 = coord2[1]

            plt.plot([x1, x2], [y1, y2], 'c', linestyle=":")

        plt.savefig("results.png", bbox_inches='tight', pad_inches=0)
        plt.clf()
