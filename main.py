from sift_flow_torch import SiftFlowTorch
import cv2
import numpy as np
import time
import torch
import torch.nn.functional as F
from MatchMaker import *
import glob


test_imgs_names =glob.glob("imgs/*.jpg")

test_imgs = [cv2.imread(test_imgs_name) for test_imgs_name in test_imgs_names]
origin_img_height, origin_img_width, origin_img_c = test_imgs[0].shape
print(test_imgs[0].shape)
width_resolution = 50
height_resolution = 10
patch_img_width = origin_img_width//width_resolution
patch_img_height = origin_img_height//height_resolution
print(patch_img_width,patch_img_height)

small_patch_size = min([patch_img_width,patch_img_height])
sift_flow = SiftFlowTorch(cell_size=small_patch_size,step_size=small_patch_size,is_boundary_included=True,num_bins=16 ,cuda=True,fp16=True,return_numpy=False)

torch.cuda.synchronize()
start = time.perf_counter()
descs = sift_flow.extract_descriptor(test_imgs)
torch.cuda.synchronize()
end = time.perf_counter()
print('Time: {:.03f} ms'.format((end - start) * 1000))

torch.cuda.synchronize()
start = time.perf_counter()
descs = sift_flow.extract_descriptor(test_imgs)
torch.cuda.synchronize()
end = time.perf_counter()
print('Time: {:.03f} ms'.format((end - start) * 1000))
print(descs.shape)
for img_des in range(1,len(test_imgs)):
    torch.cuda.synchronize()
    start = time.perf_counter()
    flow = find_local_matches(descs[0:1], descs[img_des:img_des+1],21)
    # print(img_des-1, img_des)
    torch.cuda.synchronize()
    end = time.perf_counter()
    flow = flow.permute(1, 2, 0).detach().cpu().numpy()#numpy형태로 마지막에 바꿉니다.
    max_height, max_width, _ = flow.shape
    return_img = np.zeros((origin_img_height,origin_img_width,origin_img_c),dtype=np.uint8)
    # print(max_height,max_width)
    for i in range(max_height):
        for j in range(max_width):
            origin_min_i =i*small_patch_size
            origin_min_j =j*small_patch_size
            origin_max_i = i * small_patch_size+small_patch_size
            origin_max_j = j * small_patch_size+small_patch_size
            new_j,new_i= j+flow[i,j,0],i+flow[i,j,1]
            target_min_i = new_i*small_patch_size
            target_min_j = new_j * small_patch_size
            target_max_i = target_min_i + small_patch_size
            target_max_j = target_min_j + small_patch_size
            if target_max_j >= origin_img_width:
                res_width =  target_max_j -origin_img_width+1
                target_max_j = origin_img_width - 1
                origin_max_j = origin_max_j -res_width
            if target_max_i >= origin_img_height:
                res_height = target_max_i - origin_img_height + 1
                target_max_i = origin_img_height - 1
                origin_max_i = origin_max_i - res_height
            if target_min_j < 0:
                res_width = -target_min_j
                target_min_j = 0
                origin_min_j = origin_min_j +res_width
            if target_min_i < 0:
                res_height = -target_min_i
                target_min_i = 0
                origin_min_i = origin_min_i +res_height
            return_img[target_min_i:target_max_i,target_min_j:target_max_j,:] = test_imgs[0][origin_min_i: origin_max_i,origin_min_j:origin_max_j,:]
    cv2.imshow("origin",test_imgs[0])
    cv2.imshow("target", test_imgs[img_des])
    cv2.imshow("return",return_img)
    cv2.waitKey(0)