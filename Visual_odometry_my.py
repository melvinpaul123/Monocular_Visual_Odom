import os
import numpy as np
import cv2
from matplotlib import pyplot as plt 
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp

class VisualOdometry():
    def __init__(self, calib_path,gt_pose_path):
        self.K, self.P = self._load_calib(calib_path)
        self.gt_poses = self._load_poses(gt_pose_path)
        self.sift = cv2.SIFT_create(2000)
        FLANN_INDEX_LSH = 6
        index_params = dict(algorithm=FLANN_INDEX_LSH, table_number=6, key_size=12, multi_probe_level=1)
        search_params = dict(checks=50)
        self.flann = cv2.FlannBasedMatcher(indexParams=index_params, searchParams=search_params)
        self.bf = cv2.BFMatcher()
        #self.prev_image = None
        self.prev_features = None
        self.prev_transformation = []

    @staticmethod
    def _load_calib(filepath):
        with open(filepath, 'r') as f:
            params = np.fromstring(f.readline(), dtype=np.float64, sep=' ')
            P = np.reshape(params, (3, 4))
            K = P[0:3, 0:3]
        return K, P

    @staticmethod
    def _load_poses(filepath):
        poses = []
        with open(filepath, 'r') as f:
            for line in f.readlines():
                T = np.fromstring(line, dtype=np.float64, sep=' ')
                T = T.reshape(3, 4)
                T = np.vstack((T, [0, 0, 0, 1]))
                poses.append(T)
        return poses

    @staticmethod
    def _load_images(filepath):
        image_paths = [os.path.join(filepath, file) for file in sorted(os.listdir(filepath))]
        return [cv2.imread(path, cv2.IMREAD_GRAYSCALE) for path in image_paths]

    @staticmethod
    def _form_transf(R, t):
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R
        T[:3, 3] = t
        return T
    
    def get_features(self,img):
        keypoints1, descriptors1 = self.sift.detectAndCompute(img, None)
        return [keypoints1, descriptors1]

    def get_matches(self, cur_image):
        """
        Returns
        q1 (ndarray): The good keypoints matches position in i-1'th image
        q2 (ndarray): The good keypoints matches position in i'th image
        """
        keypoints2, descriptors2 = self.sift.detectAndCompute(cur_image, None)
        #matches = self.flann.knnMatch(self.prev_features[1], descriptors2, k=2)
        matches = self.bf.knnMatch(self.prev_features[1], descriptors2,k=2)
        # store all the good matches as per Lowe's ratio test.
        good = []
        for m,n in matches:
            if m.distance < 0.5*n.distance:
                good.append(m)
        q1 = np.float32([ self.prev_features[0][m.queryIdx].pt for m in good ])
        q2 = np.float32([ keypoints2[m.trainIdx].pt for m in good ])
        self.prev_features = [keypoints2, descriptors2]
        return q1, q2

    def compute_homography_and_fundamental(self, src_pts, dst_pts):
        H, mask_homography = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        F, mask_fundamental = cv2.findFundamentalMat(src_pts, dst_pts, cv2.RANSAC, 0.999, 1.0)
        return H, F, mask_homography, mask_fundamental
    
    def getEssentialMatrix(self, F, K):
        E = K.T @ F @ K
        U, S, V_T = np.linalg.svd(E)
        E = np.dot(U, np.dot(np.diag([1, 1, 0]), V_T))
        E = E/np.linalg.norm(E)
        return E

    def get_pose(self, q1, q2):
        """
        Parameters
        ----------
        q1 (ndarray): The good keypoints matches position in i-1'th image
        q2 (ndarray): The good keypoints matches position in i'th image
        Returns
        -------
        transformation_matrix (ndarray): The transformation matrix
        """
        # H, F, mask_homography, mask_fundamental = self.compute_homography_and_fundamental(q1, q2)
        # q1 = q1[mask_homography.ravel() == 1]
        # q2 = q2[mask_homography.ravel() == 1]
        Essential, mask = cv2.findEssentialMat(q1, q2, self.K) #, method=cv2.RANSAC, prob=0.999, threshold=0.5)
        #Essential = self.getEssentialMatrix(F, self.K)
        # Essential = self.K.T.dot(F).dot(self.K)
        _, cur_R, cur_t, mask = cv2.recoverPose(Essential, q1, q2, focal=self.K[0][0], pp=(self.K[0][2], self.K[1][2]))
        if np.linalg.det(cur_R) < 0:
            cur_R = -cur_R
            cur_t = -cur_t
        return self._form_transf(cur_R,cur_t.T)
    
    def exponential_moving_average_pose(self, current_pose, previous_pose, alpha=0.8):
        # Extract rotation and translation components
        current_rotation = R.from_matrix(current_pose[:3, :3])
        previous_rotation = R.from_matrix(previous_pose[:3, :3])
        current_translation = current_pose[:3, 3]
        previous_translation = previous_pose[:3, 3]
        # Interpolate rotations using Slerp (Spherical Linear Interpolation)
        times = [0, 1]
        rotations = R.from_matrix([previous_rotation.as_matrix(), current_rotation.as_matrix()])
        slerp = Slerp(times, rotations)
        smoothed_rotation = slerp(alpha).as_matrix()
        # Smooth translations using EMA
        smoothed_translation = alpha * current_translation + (1 - alpha) * previous_translation
        # Combine smoothed rotation and translation into a transformation matrix
        smoothed_pose = np.eye(4)
        smoothed_pose[:3, :3] = smoothed_rotation
        smoothed_pose[:3, 3] = smoothed_translation
        return smoothed_pose