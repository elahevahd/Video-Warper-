import ast
import os
import cv2
import json
from multiprocessing import Pool
import numpy as np 
from matlab_cp2tform import get_similarity_transform_for_cv2
import json

highres_dir = '/media/super-server/8c707078-1e62-4724-bb51-fcc90884d110/HW_frames_highres/'  #path to high resolution frames
openpose_coordinates_dir = '/media/super-server/6f0518ce-dbdf-4676-ba55-6739f8f3ecd4/NSF_system_design/results/' #path to openpose

# coordinates of mean face:
mean_coords_list = [(71, 92), (71, 105), (72, 118), (74, 131), (79, 143), (86, 153), (96, 161), (108, 167), (121, 169), (133, 167), (144, 162), (154, 154), (161, 144), (165, 132), (167, 120), (169, 108), (170, 95), (82, 77), (89, 73), (97, 71), (106, 72), (114, 75), (131, 76), (139, 73), (147, 73), (155, 76), (160, 81), (122, 87), (122, 94), (122, 100), (122, 107), (112, 116), (117, 118), (122, 119), (126, 118), (131, 117), (92, 89), (97, 86), (103, 86), (108, 89), (103, 91), (97, 91), (135, 91), (140, 88), (146, 88), (151, 91), (146, 93), (140, 92), (104, 135), (110, 130), (116, 127), (121, 128), (126, 127), (133, 130), (138, 136), (133, 140), (126, 142), (121, 142), (115, 142), (109, 139), (107, 134), (116, 132), (121, 133), (126, 133), (135, 135), (126, 135), (121, 135), (116, 135), (100, 88), (143, 90)]
mean_arr = np.array(mean_coords_list).astype(np.float32)

def return_path(vidname):
    path = '/'.join(vidname.split('_'))
    return path 

def return_highres_path(vidname):
    path = return_path(vidname)
    return highres_dir+path 

def return_start_end(vidname):
    frame_list = os.listdir(return_highres_path(vidname))
    frame_list = [int(item.split('frame_')[1].split('.png')[0]) for item in frame_list]
    if len(frame_list)>0:
        start, end = min(frame_list), max(frame_list)
        return [start,end]
    else:
        print('no frames in the directory')
        return None


def return_warp_vid(vidname, t=120):
    path = return_path(vidname)
    openposename = os.listdir(openpose_coordinates_dir+path)[0]
    openposename =openposename.split('_keypoints.json')[0][:-13]
    start_frame, end_frame = return_start_end(vidname)
    for frame_num in range(start_frame, end_frame+1):     
        ##################### read openpose
        openpose_path = openpose_coordinates_dir+path+'/'+openposename+'_'+'{:012d}'.format(frame_num)+'_keypoints.json'
        data = json.load(open(openpose_path))
        arr=data['people'][0]['face_keypoints_2d']
        coords_list=[(round(arr[i*3]),round(arr[i*3+1])) for i in range(int(len(arr)/3))]
        ##################### find face coordinates
        bb_x_min = min([item[0] for item in coords_list])
        bb_x_max = max([item[0] for item in coords_list])
        bb_y_min = min([item[1] for item in coords_list])
        bb_y_max = max([item[1] for item in coords_list])
        c1,c2 = int(np.mean([bb_x_min,bb_x_max])), int(np.mean([bb_y_min,bb_y_max]))
        face_coords = [c1-t,c1+t,c2-t,c2+t] #x_min, x_max, y_min, y_max 
        #####################crop the frame from high resolution to 240x240 
        frame_path = return_highres_path(vidname)+'/frame_{}.png'.format(frame_num)       
        im = cv2.imread(frame_path)
        cropped_img = im[face_coords[2]:face_coords[3], face_coords[0]:face_coords[1]]  
        ##################### shift the coordinates
        x_org,y_org = [c1-t,c2-t] 
        shifted_coords_list = [(x-x_org,y-y_org) for (x,y) in coords_list]
        coord_arr = np.array(shifted_coords_list).astype(np.float32)
        mean_coords_list = [(71, 92), (71, 105), (72, 118), (74, 131), (79, 143), (86, 153), (96, 161), (108, 167), (121, 169), (133, 167), (144, 162), (154, 154), (161, 144), (165, 132), (167, 120), (169, 108), (170, 95), (82, 77), (89, 73), (97, 71), (106, 72), (114, 75), (131, 76), (139, 73), (147, 73), (155, 76), (160, 81), (122, 87), (122, 94), (122, 100), (122, 107), (112, 116), (117, 118), (122, 119), (126, 118), (131, 117), (92, 89), (97, 86), (103, 86), (108, 89), (103, 91), (97, 91), (135, 91), (140, 88), (146, 88), (151, 91), (146, 93), (140, 92), (104, 135), (110, 130), (116, 127), (121, 128), (126, 127), (133, 130), (138, 136), (133, 140), (126, 142), (121, 142), (115, 142), (109, 139), (107, 134), (116, 132), (121, 133), (126, 133), (135, 135), (126, 135), (121, 135), (116, 135), (100, 88), (143, 90)]
        mean_arr = np.array(mean_coords_list).astype(np.float32)
        ##################### warp the cropped image
        tfm = get_similarity_transform_for_cv2(coord_arr,mean_arr)
        warped_img = cv2.warpAffine(cropped_img, tfm, (240, 240))
        ##################### crop the warped image
        crop_warped_img = warped_img[41: 169, 71:170, :]
        ##################### write the outout image
        if not os.path.exists('output/'+vidname):
            os.mkdir('output/'+vidname)
        des_path = 'output/'+vidname+'/frame_{}.png'.format(frame_num)
        cv2.imwrite(des_path,crop_warped_img)

