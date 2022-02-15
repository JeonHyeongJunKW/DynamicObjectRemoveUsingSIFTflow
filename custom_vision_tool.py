import cv2
import numpy as np
from dsift import SingleSiftExtractor, MultiSiftExtractor
import time
def extract_SIFT(img, x,y, pixel_size):
    bottom_x = int(x-pixel_size/2) if int(x-pixel_size/2) >=0 else 0
    bottom_y = int(y - pixel_size / 2) if int(y - pixel_size / 2) >= 0 else 0
    top_x = int(x + pixel_size / 2) if int(x + pixel_size / 2) < img.shape[1] else img.shape[1]-1
    top_y = int(y + pixel_size / 2) if int(y + pixel_size / 2) < img.shape[0] else img.shape[0]-1
    patch_img = img[bottom_y:top_y+1,bottom_x:top_x+1]
    extractor = SingleSiftExtractor(pixel_size)
    dense_feature = extractor.process_image(patch_img)
    return dense_feature

def extract_fullDenseSiFT(img, pixel_size):
    start = time.time()
    extractor = MultiSiftExtractor(pixel_size)
    feaArr, positions = extractor.process_image(img)
    end = time.time()
    print(end-start)
    print(positions)

def extract_HoG(img,x,y,pixel_size):
    bottom_x = int(x - pixel_size / 2) if int(x - pixel_size / 2) >= 0 else 0
    bottom_y = int(y - pixel_size / 2) if int(y - pixel_size / 2) >= 0 else 0
    top_x = int(x + pixel_size / 2) if int(x + pixel_size / 2) < img.shape[1] else img.shape[1] - 1
    top_y = int(y + pixel_size / 2) if int(y + pixel_size / 2) < img.shape[0] else img.shape[0] - 1
    patch_img = img[bottom_y:top_y + 1, bottom_x:top_x + 1]
    hog = cv2.HOGDescriptor()
    descriptor = hog.compute(patch_img)
    return descriptor

def change_patch(source,s_y,s_x,target,t_y,t_x,pixel_size):
    t_bottom_x = int(t_x - pixel_size / 2) if int(t_x - pixel_size / 2) >= 0 else 0
    t_bottom_y = int(t_y - pixel_size / 2) if int(t_y - pixel_size / 2) >= 0 else 0
    t_top_x = int(t_x + pixel_size / 2) if int(t_x + pixel_size / 2) < target.shape[1] else target.shape[1] - 1
    t_top_y = int(t_y + pixel_size / 2) if int(t_y + pixel_size / 2) < target.shape[0] else target.shape[0] - 1

    s_bottom_x = int(s_x - pixel_size / 2) if int(s_x - pixel_size / 2) >= 0 else 0
    s_bottom_y = int(s_y - pixel_size / 2) if int(s_y - pixel_size / 2) >= 0 else 0
    s_top_x = int(s_x + pixel_size / 2) if int(s_x + pixel_size / 2) < target.shape[1] else target.shape[1] - 1
    s_top_y = int(s_y + pixel_size / 2) if int(s_y + pixel_size / 2) < target.shape[0] else target.shape[0] - 1



    # source_x_size = s_top_x- s_bottom_x
    # source_y_size = s_top_y - s_bottom_y
    # target_x_size = t_top_x - t_bottom_x
    # target_y_size = t_top_y - t_bottom_y
    # y_diff = target_y_size - source_y_size
    # x_diff = target_x_size - source_x_size
    # t_top_x -=x_diff
    # t_top_y -=y_diff
    target[t_bottom_y:t_top_y + 1, t_bottom_x:t_top_x + 1] = source[s_bottom_y:s_top_y + 1, s_bottom_x:s_top_x + 1]

def Get_Fundamental(img1, img2):
    descriptor = cv2.ORB_create(3000)
    kp1,des1 = descriptor.detectAndCompute(img1,None)
    kp2,des2 = descriptor.detectAndCompute(img2, None)
    bf = cv2.BFMatcher()#cv2.NORM_HAMMING, crossCheck=True
    matches = bf.match(des1, des2)

    # matches = sorted(matches, key=lambda x: x.distance)
    pts1 =[]
    pts2 =[]
    keypoint_good_1 = []
    for i, (m) in enumerate(matches):
        pts2.append(kp2[m.trainIdx].pt)
        pts1.append(kp1[m.queryIdx].pt)
    # for m,n in matches:
    #     if m.distance < 0.7 *n.distance:
    #         pts1.append(kp1[m.queryIdx].pt)
    #         pts2.append(kp2[m.trainIdx].pt)
    # dst1 = cv2.drawKeypoints(img1, kp1, None,
    #                          flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    # cv2.imshow("origin_keypoint",dst1)
    # res = cv2.drawMatches(img1,kp1,img2,kp2,matches,None,None,None,None,flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
    # cv2.imshow("origin_keypoint",res)
    # cv2.waitKey(0)
    # dst1 = cv2.drawKeypoints(img1, kp1, None,
    #                          flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    # cv2.imshow("origin_keypoint", dst1)

    pts1 = np.float64(pts1)
    pts2 = np.float64(pts2)
    print(len(pts1)) # y x
    F, mask = cv2.findFundamentalMat(pts1,pts2,cv2.FM_LMEDS)

    return F