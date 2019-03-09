import numpy as np
import torch as t
from torch import nn
import math
import cv2
import random

def reprojection_error(Pose_Trans,Pos_world,Internal_ref,GT):
    'Pose_Trans：Tcw,，4×4; Pos_world:3×1；Inter_ref:内参,3×3；GT:2×1，均为tensor类型'
    R=Pose_Trans[0:3,0:3]
    trans=Pose_Trans[0:3,3:4]
    Pos_Cam=R.mm(Pos_world)+ trans
    z=Pos_Cam[2]
    Pos_Nor=Pos_Cam / z

    Pos_Image = Internal_ref.mm(Pos_Nor)
    Pos_Image=Pos_Image[0:2,:]
    loss=(GT-Pos_Image).pow(2)
    loss=loss.sum(dim=0,keepdim=False)
    reprojection_loss=loss.sqrt().item()
    return reprojection_loss

def angel_error(Pose_Trans,Pos_world,Internal_ref,GT):
    'Pose_Trans：Tcw,，4×4; Pos_world:3×1；Inter_ref:内参,3×3；GT:2×1;均为tensor类型'
    Focal_len=Internal_ref[0][0]
    R=Pose_Trans[0:3,0:3]
    trans=Pose_Trans[0:3,3:4]
    Pos_Cam = R.mm(Pos_world) + trans
    Inverse_ir=Internal_ref.inverse()
    GT_normal=t.ones(3,1)   ##此处应当注意，传入的tensor的类型应当和torch创建tensor的默认类型一致
    GT_normal[0:2,:]=GT[0:2,:]
    GT_Back2F=Inverse_ir.mm(GT_normal)
    GT_Back2F=Focal_len * GT_Back2F
    dk=((GT_Back2F.pow(2)).sum(dim=0))
    dk=dk.sqrt()
    Dk=((Pos_Cam.pow(2)).sum(dim=0))
    Dk=Dk.sqrt()
    k=dk / Dk
    loss=(k*Pos_Cam)-GT_Back2F
    loss=(loss.pow(2)).sum()
    angel_loss=loss.sqrt()
    angel_loss=angel_loss.item()
    return angel_loss

def end_iter_thresh(TMat1,TMat2):
    '均为tensor，控制Ransac算法最后停止迭代'
    RMat1=TMat1[0:3,0:3]
    tMat1=TMat1[0:3,3:4]
    RMat2 = TMat2[0:3, 0:3]
    tMat2 = TMat2[0:3, 3:4]
    t_diff=(((tMat2-tMat1).pow(2)).sum()).sqrt()
    print('t_diff=',t_diff)

    RMat2_t=RMat2.t()
    rotDiff=RMat1.mm(RMat2_t)
    trace=rotDiff.trace()
    high=t.Tensor([3])
    low=t.Tensor([-1])
    if(trace > high):
        trace=high
    if(trace < low):
        trace=low
    ACos=t.acos((trace-1)/2)
    roterr=180 * ACos /3.14
    print('roterr_diff=',roterr)

    return max(t_diff,roterr)






if __name__=='__main__':
    T1=np.array(  [[9.9935108e-001,-1.5576084e-002,3.1508941e-002, -1.2323361e-001],
                            [9.2375092e-003,9.8130137e-001,1.9211653e-001, -1.1206967e+000],
                            [-3.3912845e-002,-1.9170459e-001,9.8083067e-001, -9.8870575e-001],
                            [0.0000000e+000,0.0000000e+000,0.0000000e+000,1.0000000e+000]],dtype=np.float32)
    T2= np.array([[  9.4957983e-001	,-2.2089641e-001,2.2234257e-001,-3.3096179e-001],
                            [9.6571468e-002,8.8108939e-001,4.6292341e-001,-4.6103507e-001],
                            [-2.9817075e-001, -4.1812429e-001,8.5801524e-00,1.1960953e-001],
                            [0.0000000e+000,0.0000000e+000,0.0000000e+000,1.0000000e+000]], dtype=np.float32)
    T3=np.array([[ 0.9496113 , -0.22090425 , 0.22235052 ,-0.33097267],
                    [ 0.09657339  ,0.8811139,   0.46293843, -0.46104786],
                    [-0.29818118, -0.41813838 , 0.8580491 ,  0.11961284],
                    [ 0.0000000e+000,0.0000000e+000,0.0000000e+000,1.0000000e+000]],dtype=np.float32)
    T1=t.from_numpy(T1)
    T2 = t.from_numpy(T2)
    T3=t.from_numpy(T3)
    thres=end_iter_thresh(T2,T3)
    print('THRES=',thres.numpy())



    # Pos_world=np.array([[1],
    #                    [2],
    #                    [3]],dtype=np.float32)
    # Pos_world=t.from_numpy(Pos_world)
    # Inter_ref=np.array([[0.3,0,2.2],
    #                     [0,1.1,3.2345],
    #                     [0,0,5]],dtype=np.float32)
    # Inter_ref=t.from_numpy(Inter_ref)
    #
    # GT=np.array([[3],[4]],dtype=np.float32)
    # GT=t.from_numpy(GT)
    # Focus=0.5
    # #angleloss=Angel_LOSS()
    # #loss2=angleloss(Pos_Trans,Pos_world,Inter_ref,GT,Focus)
    # #print(loss2)
    #
    #
    #
    # loss1=reprojection_error(Pos_Trans,Pos_world,Inter_ref,GT)
    # loss2=angel_error(Pos_Trans,Pos_world,Inter_ref,GT)
    # c=loss2+5
    # print(c)














