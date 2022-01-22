import cv2 
import datetime
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
"""
Implementation of stereo visual odometry using KITTI dataset
"""
class Dataset_Handler():
    """ 
    To process odometry data from KITTI dataset and extract necessary infomation that will
    be used later in the project
    """
    def __init__(self, sequence):
        # Set file path
        self.seq_dir = '../dataset/sequences/{}/'.format(sequence)
        self.poses_dir = '../dataset/poses/{}.txt'.format(sequence)
        # self.calib_dir = '../calib/{}/'.format(sequence)
        poses = pd.read_csv(self.poses_dir, delimiter=' ', header = None)

        self.left_image_files = sorted(os.listdir(self.seq_dir + 'image_0'))
        self.right_image_files = sorted(os.listdir(self.seq_dir + 'image_1'))
        self.num_frames = len(self.left_image_files)


        calib = pd.read_csv(self.seq_dir + 'calib.txt', delimiter=' ', header=None, index_col=0)        
        self.P0 = np.array(calib.loc['P0:']).reshape((3,4))
        self.P1 = np.array(calib.loc['P1:']).reshape((3,4))

        self.times = np.array(pd.read_csv(self.seq_dir + 'times.txt', delimiter=' ', header=None))
        self.gt = np.zeros((len(poses),3,4))
        for i in range(len(poses)):
            self.gt[i] = np.array(poses.iloc[i]).reshape((3,4))


        self.reset_frames()
        self.first_image_left = cv2.imread(self.seq_dir + 'image_0/' + self.left_image_files[0],0)
        self.first_image_right = cv2.imread(self.seq_dir + 'image_1/' + self.left_image_files[0],0)
        self.second_image_left = cv2.imread(self.seq_dir + 'image_0/' + self.left_image_files[1],0)

        self.imheight = self.first_image_left.shape[0]
        self.imwidth = self.first_image_left.shape[1]

    def reset_frames(self):
        self.images_left = (cv2.imread(self.seq_dir + 'image_0/' + name_left,0)
                            for name_left in self.left_image_files)
        self.images_right = (cv2.imread(self.seq_dir + 'image_1/' + name_right,0)
                            for name_right in self.right_image_files)

        pass

def decompose_projection_matrix(p):
    """
    To extract useful matrices from projection matrix
    """
    k, r, t, _, _, _, _ = cv2.decomposeProjectionMatrix(p)
    t = (t / t[3])[:3]
    
    return k, r, t

def compute_left_disparity_map(img_left, img_right, matcher='bm'):
    """
    To calculate the disparity between the left frame and right frame
    """
    
    sad_window = 6
    num_disparities = sad_window * 16
    block_size = 11
    matcher_name = matcher
    
    if matcher_name == 'bm':
        matcher = cv2.StereoBM_create(numDisparities=num_disparities,
                                      blockSize=block_size)
        
    elif matcher_name == 'sgbm':
        matcher = cv2.StereoSGBM_create(numDisparities=num_disparities,
                                        minDisparity=0,
                                        blockSize=block_size,
                                        P1 = 8 * 1 * block_size ** 2,
                                        P2 = 32 * 1 * block_size ** 2,
                                        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)
        
    disp_left = matcher.compute(img_left, img_right).astype(np.float32)/16
        
    return disp_left

def calc_depth_map(disp_left, k_left, t_left, t_right):
    """
    To calculate the depth values of each pixel and generate a map
    """
    b = abs(t_right[0] - t_left[0])
        
    f = k_left[0][0]
    
    disp_left[disp_left == 0.0] = 0.1
    disp_left[disp_left == -1.0] = 0.1
    
    depth_map = np.ones(disp_left.shape)
    depth_map = f * b / disp_left
    
    return depth_map

def stereo_2_depth(img_left, img_right, P0, P1):
    """
    To calculate the depth of stereo cameras
    """
    disp = compute_left_disparity_map(img_left,
                                      img_right,
                                      matcher='bm')

    k_left, r_left, t_left = decompose_projection_matrix(P0)
    k_right, r_right, t_right = decompose_projection_matrix(P1)
    
    depth = calc_depth_map(disp, k_left, t_left, t_right)
    
    return depth

def extract_features(image, detector='sift', mask=None):
    """
    To extract features in an image using two different method
    """
    start = datetime.datetime.now()
    if detector == 'sift':
        det = cv2.SIFT_create()
    elif detector == 'orb':
        det = cv2.ORB_create()
        
    kp, des = det.detectAndCompute(image, mask)
    end = datetime.datetime.now()
    return kp, des

def match_features(des1, des2, detector='sift', k=2):
    """
    To match features of two images using Brute-force method
    """
    if detector == 'sift':
        matcher = cv2.BFMatcher_create(cv2.NORM_L2, crossCheck=False)
    elif detector == 'orb':
        matcher = cv2.BFMatcher_create(cv2.NORM_HAMMING2, crossCheck=False)
    
    matches = matcher.knnMatch(des1, des2, k=k)
        
    return matches

def filter_matches_distance(matches, dist_threshold=0.45):
    """
    To filter out matches with ratios of distance higher than the threshold
    """
    filtered_matches = []
    for m, n in matches:
        if m.distance <= dist_threshold * n.distance:
            filtered_matches.append(m)
            
    return filtered_matches

def estimate_motion(matches, kp1, kp2, k, depth1, max_depth=3000):
    """
    To estimate the pose of the camera in each frame
    """
    
    rmat = np.eye(3)
    tvec = np.zeros((3, 1))
    
    image1_points = np.float32([kp1[m.queryIdx].pt for m in matches])
    image2_points = np.float32([kp2[m.trainIdx].pt for m in matches])
    
    cx = k[0, 2]
    cy = k[1, 2]
    fx = k[0, 0]
    fy = k[1, 1]
    
    object_points = np.zeros((0, 3))
    delete = []
    
    for i, (u, v) in enumerate(image1_points):
        z = depth1[int(round(v)), int(round(u))]
        
        if z > max_depth:
            delete.append(i)
            continue
            
        x = z * (u - cx) / fx
        y = z * (v - cy) / fy
        object_points = np.vstack([object_points, np.array([x, y, z])])
        
    image1_points = np.delete(image1_points, delete, 0)
    image2_points = np.delete(image2_points, delete, 0)
    
    _, rvec, tvec, inliers = cv2.solvePnPRansac(object_points, image2_points, k, None)
    rmat = cv2.Rodrigues(rvec)[0]
    
    return rmat, tvec, image1_points, image2_points


def visual_odometry(handler, detector='sift', matching='BF', filter_match_distance=0.45,
                    stereo_matcher='sgbm', mask=None):
    """
    The visual odometry function which generate a estimated path
    """
    # Generate a plt
    fig = plt.figure(figsize=(14, 14))
    ax = fig.add_subplot(projection='3d')
    ax.view_init(elev=-20, azim=270)
    xs = handler.gt[:, 0, 3]
    ys = handler.gt[:, 1, 3]
    zs = handler.gt[:, 2, 3]
    ax.set_box_aspect((np.ptp(xs), np.ptp(ys), np.ptp(zs)))
    ax.plot(xs, ys, zs, c='k')
        
    # Establish a homogeneous transformation matrix. First pose is identity
    T_tot = np.eye(4)
    trajectory = np.zeros((handler.num_frames, 3, 4))
    trajectory[0] = T_tot[:3, :]
    imheight = handler.imheight
    imwidth = handler.imwidth
    
    # Decompose left camera projection matrix to get intrinsic k matrix
    k_left, r_left, t_left = decompose_projection_matrix(handler.P0)
    

    handler.reset_frames()
    image_plus1 = next(handler.images_left)
        
    # Iterate through all frames of the sequence
    for i in range(handler.num_frames - 1):
        start = datetime.datetime.now()

        image_left = image_plus1
        image_plus1 = next(handler.images_left)
        image_right = next(handler.images_right)
        depth = stereo_2_depth(image_left,
                               image_right,
                               P0=handler.P0,
                               P1=handler.P1
                              )
            
        # Get keypoints and descriptors for left camera image of two sequential frames
        kp0, des0 = extract_features(image_left, detector, mask)
        kp1, des1 = extract_features(image_plus1, detector, mask)
        
        # Get matches between features detected in two subsequent frames
        matches_unfilt = match_features(des0, 
                                        des1,
                                        detector=detector
                                       )
        
        # Filter matches if a distance threshold is provided by user
        matches = filter_matches_distance(matches_unfilt, filter_match_distance)

            
        

        # Estimate motion between sequential images of the left camera
        rmat, tvec, img1_points, img2_points = estimate_motion(matches,
                                                               kp0,
                                                               kp1,
                                                               k_left,
                                                               depth
                                                              )
        
        # Create a blank homogeneous transformation matrix
        Tmat = np.eye(4)
        Tmat[:3, :3] = rmat
        Tmat[:3, 3] = tvec.T
        
        T_tot = T_tot.dot(np.linalg.inv(Tmat))
        
        trajectory[i+1, :, :] = T_tot[:3, :]
        
        end = datetime.datetime.now()
        print('Time to compute frame {}:'.format(i+1), end-start)
        

        xs = trajectory[:i+2, 0, 3]
        ys = trajectory[:i+2, 1, 3]
        zs = trajectory[:i+2, 2, 3]
        plt.plot(xs, ys, zs, c='chartreuse')
        plt.pause(1e-32)          

    plt.close()
        
    return trajectory


def main():
    """
    main function
    """
    handler = Dataset_Handler("07")
    traj = visual_odometry(handler, detector='sift', matching='BF', filter_match_distance=0.45,
                            stereo_matcher='sgbm', mask=None)

if __name__ == "__main__":
    main()
