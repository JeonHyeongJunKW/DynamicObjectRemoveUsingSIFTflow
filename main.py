from sift_flow_torch import SiftFlowTorch
import cv2
import numpy as np
import time
import torch
import torch.nn.functional as F
from MatchMaker import *
from custom_metric import *
import glob
from sklearn.cluster import DBSCAN
max_flow =2
t_r =0.4
use_max = False
test_imgs_names =glob.glob("imgs/*.jpg")
print(test_imgs_names)
test_imgs = [cv2.imread(test_imgs_name) for test_imgs_name in test_imgs_names]
origin_img_height, origin_img_width, origin_img_c = test_imgs[0].shape

width_resolution = 30
height_resolution = 10
patch_img_width = origin_img_width//width_resolution
patch_img_height = origin_img_height//height_resolution
print(patch_img_width,patch_img_height)

small_patch_size = min([patch_img_width,patch_img_height])
sift_flow = SiftFlowTorch(cell_size=small_patch_size,step_size=small_patch_size,is_boundary_included=False,num_bins=16 ,cuda=True,fp16=True,return_numpy=False)

torch.cuda.synchronize()
start = time.perf_counter()
descs = sift_flow.extract_descriptor(test_imgs)
print(descs)
torch.cuda.synchronize()
end = time.perf_counter()
print('Time: {:.03f} ms'.format((end - start) * 1000))
descs_np = descs.clone().permute(0, 2, 3, 1).detach().cpu().numpy()
print(descs_np.shape)

torch.cuda.synchronize()
start = time.perf_counter()
descs = sift_flow.extract_descriptor(test_imgs)
torch.cuda.synchronize()
end = time.perf_counter()
print('Time: {:.03f} ms'.format((end - start) * 1000))
print(descs.shape)
saved_flow = []

for img_des in range(1,max_flow+1):
    torch.cuda.synchronize()
    start = time.perf_counter()
    flow = find_local_matches(descs[0:1], descs[img_des:img_des+1],49)
    torch.cuda.synchronize()
    end = time.perf_counter()
    # print(flow)
    flow_2 = flow.permute(1, 2, 0).detach().cpu().numpy()#numpy형태로 마지막에 바꿉니다.

    saved_flow.append(flow_2.copy())

max_height, max_width, _ = saved_flow[0].shape
#유사도 점수 매기기
similarity_map = np.zeros((max_height, max_width, max_flow))
correspondence_map = np.zeros((max_flow,max_height, max_width,2))
for flow_idx in range(max_flow):
    start = time.time()
    for i in range(max_height):#12ms정도 걸림 근데 더빨리하는 방법이 있을듯
            for j in range(max_width):
                new_j, new_i = j + saved_flow[flow_idx][i, j, 0], i + saved_flow[flow_idx][i, j, 1]
                correspondence_map[flow_idx,i, j] =[new_i,new_j]
                origin_descs = descs_np[0, new_i,new_j]
                target_descs = descs_np[flow_idx+1, new_i,new_j]#1을 더해줍니다.
                dist2 = Se_distance(origin_descs,target_descs,0.35)

                similarity_map[new_i, new_j, flow_idx] = dist2
    corr_image = similarity_map[:, :, flow_idx].copy()
    end = time.time()
    print(flow_idx,"번 유사도 뽑는데 걸린시간 : ",end-start)
    image =(corr_image*255).astype(np.uint8)

    image = cv2.resize(image,dsize=(800,400),interpolation=cv2.INTER_AREA)
    cv2.imshow("correspondence",image)
    cv2.waitKey(0)


scan_neighbor = [[[0,-1],[-1,0]],[[0,1],[1,0]]]
start_height = [0, max_height]
start_width = [0, max_width]
end_height = [max_height+1,-1]
end_width = [max_width+1, -1]
way = [1,-1]
print("main start")
initial_correspondence_map = correspondence_map.copy()
initial_similarity_map= similarity_map.copy()
return_img = test_imgs[0].copy()
mask_img = np.ones((max_height,max_width,1),dtype=np.uint8)*255
for scan_vec in [1,0]:
    if scan_vec == 1:
        origin_color = [221,0,255]
    else :
        origin_color = [90, 243, 197]
        correspondence_map = initial_correspondence_map
        similarity_map = initial_similarity_map
    for y in range(start_height[scan_vec], end_height[scan_vec], way[scan_vec]):
        for x in range(start_width[scan_vec], end_width[scan_vec], way[scan_vec]):
            big_mask_img = cv2.resize(mask_img, dsize=(origin_img_width,origin_img_height), interpolation=cv2.INTER_AREA)
            cv2.imshow("mask", big_mask_img)
            cv2.imshow("result", return_img)
            cv2.waitKey(1)
            x_hats = []
            for idx in range(max_flow):
                x_hats_in_source = []
                #각 소스 이미지마자 후보점들을 추가합니다.(이웃점들을 기반으로 해서 얻습니다.)
                for neigh_point_delta in scan_neighbor[scan_vec]:

                    x_ref_neigh = np.float64([y, x]) + np.float64(neigh_point_delta)
                    if (x_ref_neigh[0] < 0 or x_ref_neigh[0] >= max_height) or \
                            (x_ref_neigh[1] < 0 or x_ref_neigh[1] >= max_width):
                        continue
                    #적어도 correspondence 맵에 있어야함.. 어떡하지
                    x_hat_neigh = np.array([correspondence_map[idx,int(x_ref_neigh[0]), int(x_ref_neigh[1]),0],
                                            correspondence_map[idx,int(x_ref_neigh[0]), int(x_ref_neigh[1]),1]])#동적인 물체 주변에서는 완전히 이상한 값이 나옴
                    x_hat_cand = x_hat_neigh - np.float64(neigh_point_delta)#이웃점에서 이제 다시 후보점으로 만든다. 보통 정적인 점이면 이 후보점과의 매칭이 들어맞아야한다.
                    x_hat_cand = x_hat_cand.astype(np.int32)
                    x_hats_in_source.append(x_hat_cand)#후보점들을 추가합니다.

                x_hats.append(x_hats_in_source)#소스이미지 단위로 두개를 추가합니다.
            x_hats_denseSIFT = 0
            x_hats_confidence = []  # 각 후보점에 대한 신뢰도를 조사합니다.
            x_hats_lab = 0
            x_hats_c_x2_neigh_idx = []  # 후보점 선택에 참여한 인덱스를 얻습니다.
            good_x_hat = []  # 후보선택에 사용된 x_hat
            ind = 0
            Extracted_feature_Size = 0
            # print("--------------------------------------------")
            for idx, one_source_candidate in enumerate(x_hats):  # 각 소스이미지에 대해서
                for neigh_idx, point in enumerate(one_source_candidate):
                    ind += 1



                    x_ref_neigh = np.float64([y, x]) + np.float64(scan_neighbor[scan_vec][neigh_idx])
                    x_ref_neigh = x_ref_neigh.astype(np.int32)
                    if (x_ref_neigh[0] < 0 or x_ref_neigh[0] >= max_height) or \
                            (x_ref_neigh[1] < 0 or x_ref_neigh[1] >= max_width):
                        continue
                    if (point[0] < 0 or point[0] >= max_height) or \
                            (point[1] < 0 or point[1] >= max_width):
                        continue
                    if type(x_hats_denseSIFT) is not np.ndarray:
                        x_hats_denseSIFT = descs_np[flow_idx + 1,point[0], point[1]]  # 각소스이미지의 feature를 추출합니다.
                        DBSCAN_features = np.reshape(x_hats_denseSIFT, (1, -1))
                    else:
                        x_hats_denseSIFT = descs_np[flow_idx + 1,point[0], point[1]]
                        DBSCAN_feature = x_hats_denseSIFT
                        DBSCAN_features = np.vstack((DBSCAN_features, DBSCAN_feature))
                    x_hats_confidence.append(similarity_map[x_ref_neigh[0], x_ref_neigh[1], idx])  # 후보점의 신뢰도를 점수로 사용한다?
                    # 이게아니라 x_r과 현재 후보 feature가 잘맞는가아닐까? 아니면 후보
                    x_hats_c_x2_neigh_idx.append(ind - 1)
                    good_x_hat.append([point[0], point[1]])
            if type(x_hats_denseSIFT) is not np.ndarray:
                # print("나는 좆됫어")
                return_img[y*small_patch_size:y*small_patch_size+small_patch_size, x*small_patch_size:x*small_patch_size+small_patch_size, :] = [0, 255, 255]#노란색
                continue#후보 feature들이 모두 자격이 없는경우
            if use_max:
                max_ind_conf = np.argmax(x_hats_confidence)
                center_of_denseSIFT = DBSCAN_features[max_ind_conf, :]
            else :
                dbscan_model = DBSCAN(eps=0.01, min_samples=1)
                # print("-------------------")
                # print(DBSCAN_features)
                clustering = dbscan_model.fit(DBSCAN_features)
                clustering_label = clustering.labels_
                # print(clustering_label)
                # print(clustering_label)
                k = np.max(clustering_label)  # 군집의 갯수 최소 샘플은 1로했기때문에 outlier취급은 하지않는다.
                max_cluster_idx = -1
                max_b_k = 0
                feature_conf = np.array(x_hats_confidence)  # 각 신뢰도를 numpy로 변형합니다.

                cluster_by_bk = {}

                for cluster_idx in range(k + 1):  # 0부터 가장 큰 라벨에 대해서 검사를 합니다.
                    k_full_feature = DBSCAN_features[clustering_label == cluster_idx, :]  # 해당 클러스터의 feature를 가져옵니다.
                    k_feature_conf = feature_conf[clustering_label == cluster_idx]
                    # print(k_feature_conf," : ",np.sum(k_feature_conf)," - ",k_full_feature.shape)
                    b_k = np.sum(k_feature_conf)
                    cluster_by_bk[cluster_idx] = b_k
                    if b_k > max_b_k:
                        max_b_k = b_k
                        max_cluster_idx = cluster_idx

                if max_cluster_idx == -1:  # 단일 클러스터이며, 가장자리에 해당하는부분이라서 dynamic object를 잡기힘들다.
                    return_img[y, x] = [100, 255, 255]
                    continue
                # 가장 static한 지점이라는 점에 대해서 연산합니다.
                k_full_feature = DBSCAN_features[clustering_label == max_cluster_idx, :]
                k_feature_conf = feature_conf[clustering_label == max_cluster_idx]
                center_of_denseSIFT = k_full_feature[0, :] * k_feature_conf[0]

                # print(k_feature_conf)
                # print(max_b_k)
                for idx_k_max in range(1, k_full_feature.shape[0]):
                    center_of_denseSIFT += k_full_feature[idx_k_max, :] * k_feature_conf[idx_k_max]
                center_of_denseSIFT /= max_b_k

            if False:
                M_xr = (1-0.5)*Se_distance(Extracted_feature[0][y, x,:], center_of_denseSIFT, 0.45) + \
                       0.5#*final_s_x
            else : #엣지가 없다면 굳이 기하학 따질 필요없음
                M_xr = Se_distance(descs_np[0, y, x], center_of_denseSIFT, 0.45)

            if M_xr <= t_r: # 임계값보다 작으면 동적인 물체!
                # print("너무 작네요 ㅠㅠ",M_xr)

                if mask_img[y, x] == 255:
                    return_img[y*small_patch_size:y*small_patch_size+small_patch_size, x*small_patch_size:x*small_patch_size+small_patch_size, :]= origin_color
                    mask_img[y, x] = 244
                else :
                    return_img[y*small_patch_size:y*small_patch_size+small_patch_size, x*small_patch_size:x*small_patch_size+small_patch_size, :]= [0,0,255]
                    mask_img[y, x] =0
                # print(M_xr)
                cluster_idx = 0
                #-------------다시봐야하는지점
                # print("-------------------------------")

                for source_ind in range(max_flow):#동적인 점이기 때문에 주변점은 정적인 점으로 매핑이 되어있어야한다.
                    max_cost = 0
                    max_source_cand_i = -1
                    cost_queue = []
                    # print("기존유사도", similarity_map[y, x, source_ind])
                    for source_cand_i in range(2):
                        neigh_point_delta = scan_neighbor[scan_vec][source_cand_i]
                        x_ref_neigh = np.float64([y, x]) + np.float64(neigh_point_delta)
                        if (x_ref_neigh[0] < 0 or x_ref_neigh[0] >= max_height) or \
                                (x_ref_neigh[1] < 0 or x_ref_neigh[1] >= max_width):
                            continue
                        cost = similarity_map[int(x_ref_neigh[0]), int(x_ref_neigh[1]), source_ind]
                        cost_queue.append(cost)
                        if cost >max_cost:
                            max_source_cand_i = source_cand_i
                            max_cost = cost
                    if max_source_cand_i ==-1:#매칭이 잘안되는점들 이상하네 ㅋㅋ  이점들은 매칭이 descriptor밖에 생겨버림.. 어떡함
                        # print("엥 나도 나오냐?")~~근데 어차피 코스트 0이라서 영향은 안주긴함..
                        return_img[y*small_patch_size:y*small_patch_size+small_patch_size, x*small_patch_size:x*small_patch_size+small_patch_size, :]=[255, 0, 0]#파란색
                        continue
                    else :
                        # 실제 후보점을 좌표로 넣고
                        neigh_point_delta = scan_neighbor[scan_vec][max_source_cand_i]
                        x_ref_neigh = np.float64([y, x]) + np.float64(neigh_point_delta)
                        correspondence_map[source_ind,y,x] = correspondence_map[source_ind,int(x_ref_neigh[0]),int(x_ref_neigh[1])]-np.float64(neigh_point_delta)

                        # 신뢰도는 이웃했던 정적인점의 신뢰도를 대신해서 넣는다.
                        similarity_map[y, x, source_ind] = similarity_map[int(x_ref_neigh[0]),int(x_ref_neigh[1]),source_ind]
            else:
                for source_ind in range(max_flow):#동적인 점이기 때문에 주변점은 정적인 점으로 매핑이 되어있어야한다.
                    max_cost = 0
                    max_source_cand_i = -1
                    cost_queue = []
                    # print("기존유사도", similarity_map[y, x, source_ind])
                    origin_cost = similarity_map[y,x,source_ind] #원래 인덱스
                    for source_cand_i in range(2):
                        neigh_point_delta = scan_neighbor[scan_vec][source_cand_i]
                        x_ref_neigh = np.float64([y, x]) + np.float64(neigh_point_delta)
                        if (x_ref_neigh[0] < 0 or x_ref_neigh[0] >= max_height) or \
                                (x_ref_neigh[1] < 0 or x_ref_neigh[1] >= max_width):
                            continue
                        cost = similarity_map[int(x_ref_neigh[0]), int(x_ref_neigh[1]), source_ind]
                        cost_queue.append(cost)
                        if cost >max_cost:
                            max_source_cand_i = source_cand_i
                            max_cost = cost
                    # print(cost_queue)
                    if max_source_cand_i ==-1:#매칭이 잘안되는점들 이상하네 ㅋㅋ  이점들은 매칭이 descriptor밖에 생겨버림.. 어떡함
                        # print("엥 나도 나오냐?")~~근데 어차피 코스트 0이라서 영향은 안주긴함..
                        # return_img[y, x] = [255, 0, 0]
                        continue
                    else :
                        # 실제 후보점을 좌표로 넣고
                        if max_cost > origin_cost:
                            neigh_point_delta = scan_neighbor[scan_vec][max_source_cand_i]
                            x_ref_neigh = np.float64([y, x]) + np.float64(neigh_point_delta)
                            correspondence_map[source_ind,y,x] = correspondence_map[source_ind,int(x_ref_neigh[0]),int(x_ref_neigh[1])]-np.float64(neigh_point_delta)

                            # 신뢰도는 이웃했던 정적인점의 신뢰도를 대신해서 넣는다.
                            similarity_map[y, x, source_ind] = similarity_map[int(x_ref_neigh[0]),int(x_ref_neigh[1]),source_ind]
cv2.waitKey(0)