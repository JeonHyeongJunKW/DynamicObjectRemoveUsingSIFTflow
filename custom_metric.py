import numpy as np
import cv2
def Se_distance(feature1, feature2, numerator_sigma):
    return np.exp(-(np.linalg.norm((feature1-feature2)))**2/(2*numerator_sigma*numerator_sigma))

def Sf_distance(point1,point2,fundamental):
    sigma_e =0.17
    Fx1 = np.dot(fundamental, point1.T)
    Fx2 = np.dot(fundamental.T, point2.T)
    denom = Fx1[0] ** 2 + Fx1[1] ** 2 + Fx2[0] ** 2 + Fx2[1] ** 2
    temp_term = np.dot(point2, np.dot(fundamental, point1.T))
    err = (temp_term) ** 2 / denom
    sampson_distance = err#cv2.sampsonDistance(point1,point2,fundamental)

    return np.exp(- sampson_distance/(2*sigma_e*sigma_e))
def Sf_distance2(point1,point2,fundamental):
    sigma_e =0.17
    Fx1 = np.dot(fundamental, point1.T)
    Fx2 = np.dot(fundamental.T, point2.T)
    denom = Fx1[0] ** 2 + Fx1[1] ** 2 + Fx2[0] ** 2 + Fx2[1] ** 2
    temp_term = np.dot(point2, np.dot(fundamental, point1.T))
    err = (temp_term) ** 2 / denom
    sampson_distance = err#cv2.sampsonDistance(point1,point2,fundamental)

    return sampson_distance

def clustering_feature(x1, x2):
    Lambda_1 = 0.15
    Lambda_2 = 0.4
    Lambda_4 = Lambda_1 / (Lambda_1 + Lambda_2)
    Lambda_5 = Lambda_2 / (Lambda_1 + Lambda_2)
    x1_dsift_feature = x1[:128]
    x1_lab_feature = x1[128:]
    x2_dsift_feature = x2[:128]
    x2_lab_feature = x2[128:]
    sigma_c = 4.8
    sigma_h = 4.8
    return_value = 1- Lambda_4*Se_distance(x1_lab_feature,x2_lab_feature,sigma_c) - Lambda_5*Se_distance(x1_dsift_feature,x2_dsift_feature,sigma_h)
    return return_value