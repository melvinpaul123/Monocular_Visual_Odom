import sys
import os
import numpy as np
import cv2
from matplotlib import pyplot as plt 
from Visual_odometry_my import VisualOdometry
import plotting

def main():
    data_dir = 'test_files/'
    fol = "kitti06"
    vo = VisualOdometry(data_dir+fol+'/calib.txt',data_dir+fol+'/poses.txt')
    cap=cv2.VideoCapture(data_dir+fol+"/video.mp4")
    if (cap.isOpened() == False):  
        print("-> Error reading video file") 
        sys.exit()
    # play_trip(vo.images)  # Comment out to not play the trip
    gt_path = []
    estimated_path = []
    frame_i = 0
    
    while(1):
        ret, img = cap.read()
        if ret == False:
            print('-> Failed to capture frame from camera. Check camera index in cv2.VideoCapture(0) \n')
            cv2.destroyAllWindows()
            cap.release()
            break
        if frame_i == 0:
            cur_pose = vo.gt_poses[frame_i]
            #vo.prev_image = img.copy()
            vo.prev_features = vo.get_features(img)
        else:
            q1, q2 = vo.get_matches(img)
            transf1 = vo.get_pose(q1, q2)
            if len(vo.prev_transformation)!=0:
                transf = vo.exponential_moving_average_pose(transf1, vo.prev_transformation)
            else: 
                transf = transf1
            cur_pose = np.matmul(cur_pose, np.linalg.inv(transf))
            vo.prev_transformation = transf1
        gt_path.append((vo.gt_poses[frame_i][0, 3], vo.gt_poses[frame_i][2, 3]))
        estimated_path.append((cur_pose[0, 3], cur_pose[2, 3]))
        fig = plt.figure(figsize=(10,10))
        plt.plot(np.array(gt_path)[:,0], np.array(gt_path)[:,1], label="GT")#, marker = 'o')
        plt.plot(np.array(estimated_path)[:,0], np.array(estimated_path)[:,1], label="PD")#, marker = 'o')
        max_limx = max(np.array(estimated_path+gt_path)[:,0])
        min_limx = min(np.array(estimated_path+gt_path)[:,0])
        max_limy = max(np.array(estimated_path+gt_path)[:,1])
        min_limy = min(np.array(estimated_path+gt_path)[:,1])
        plt.xlim(min_limx-5, max_limx+5)
        plt.ylim(min_limy-5, max_limy+5)
        plt.legend()
        ax = plt.gca()
        ax.set_aspect('equal', adjustable='box')
        plt.savefig('org.png', bbox_inches='tight')
        plt.close(fig)
        org = cv2.imread('org.png')
        cv2.imshow("Video",img)
        cv2.imshow("Path",org)
        frame_i = frame_i+1
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            cv2.destroyAllWindows()
            cap.release()
        if k == ord('p'): #pause stream
            print("-> Pausing Video Stream")
            print("-> Press any key to continue Video Stream")
            cv2.waitKey(-1) #wait until any key is pressed
    plotting.visualize_paths(gt_path, estimated_path, "Visual Odometry", file_out=fol+"_out.html")

if __name__ == "__main__":
    main()