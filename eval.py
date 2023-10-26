import torch
from dataset import get_data_transforms, load_data,get_data_transforms_for_blackbox
from torchvision.datasets import ImageFolder
import numpy as np
from torch.utils.data import DataLoader
from resnet import resnet18, resnet34, resnet50, wide_resnet50_2
from de_resnet import de_resnet18, de_resnet50, de_wide_resnet50_2
from dataset import MVTecDataset
from torch.nn import functional as F
from sklearn.metrics import roc_auc_score
import cv2
import matplotlib.pyplot as plt
from sklearn.metrics import auc
from skimage import measure
import pandas as pd
from numpy import ndarray
from statistics import mean
from scipy.ndimage import gaussian_filter
from sklearn import manifold
from matplotlib.ticker import NullFormatter
from scipy.spatial.distance import pdist
import matplotlib
import pickle
import time
import os
import logging
import xml.etree.ElementTree as ET

class eval():
    def __init__(self):
        pass

def cal_anomaly_map(fs_list, ft_list, out_size, amap_mode='mul'):
    if amap_mode == 'mul':
        anomaly_map = np.ones([out_size[-2], out_size[-1]])
    else:
        anomaly_map = np.zeros([out_size[-2], out_size[-1]])
    a_map_list = []
    for i in range(len(ft_list)):
        # print(i)
        print(anomaly_map.shape)
        fs = fs_list[i]
        ft = ft_list[i]
        #fs_norm = F.normalize(fs, p=2)
        #ft_norm = F.normalize(ft, p=2)
        a_map = 1 - F.cosine_similarity(fs, ft)
        # print(a_map.shape)
        a_map = torch.unsqueeze(a_map, dim=1)
        a_map = F.interpolate(a_map, size=(out_size[-2],out_size[-1]), mode='bilinear', align_corners=True)
        a_map = a_map[0, 0, :, :].to('cpu').detach().numpy()
        
        a_map_list.append(a_map)
        if amap_mode == 'mul':
            anomaly_map *= a_map
        else:
            anomaly_map += a_map
    return anomaly_map, a_map_list


def show_cam_on_image(img, anomaly_map):
    #if anomaly_map.shape != img.shape:
    #    anomaly_map = cv2.applyColorMap(np.uint8(anomaly_map), cv2.COLORMAP_JET)
    cam = np.float32(anomaly_map)/255 + np.float32(img)/255
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)

def min_max_norm(image):
    a_min, a_max = image.min(), image.max()
    return (image-a_min)/(a_max - a_min)

def cvt2heatmap(gray):
    heatmap = cv2.applyColorMap(np.uint8(gray), cv2.COLORMAP_JET)
    return heatmap

def visualization(_class_):
    print(_class_)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)

    # data_transform, gt_transform = get_data_transforms(256, 256)
    # 不同的data需要不同datatransform
    if _class_ == 'bogie_2':
        size1=1024
        size2=1024*3
        # size3=1024
    elif _class_ == 'channelbox_2':
        size1=1024
        size2=1024
        # size3=1024
    data_transform, gt_transform = get_data_transforms_for_blackbox(size1,size2)
    test_path = './ADdataset/' + _class_
    ckp_path = './checkpointsteds_2/'+_class_ + '/wres50_'+_class_+'199.pth'
    test_data = MVTecDataset(root=test_path, transform=data_transform, gt_transform=gt_transform, phase="test")
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False)
    
    # train_path = './mvtec_anomaly_detection/' + _class_ + '/train'


    encoder, bn = wide_resnet50_2(pretrained=True)
    encoder = encoder.to(device)
    bn = bn.to(device)

    encoder.eval()
    decoder = de_wide_resnet50_2(pretrained=False)
    decoder = decoder.to(device)
    ckp = torch.load(ckp_path)
    for k, v in list(ckp['bn'].items()):
        if 'memory' in k:
            ckp['bn'].pop(k)
    decoder.load_state_dict(ckp['decoder'])
    bn.load_state_dict(ckp['bn'])
    thereshold = 0.5
    count = 0
    with torch.no_grad():
        for img, gt, label, _ in test_dataloader:
            # if (label.item() == 0):
            #     continue
            decoder.eval()
            bn.eval()
            img = img.to(device)
            inputs = encoder(img)
            outputs = decoder(bn(inputs))
            anomaly_map, amap_list = cal_anomaly_map([inputs[-1]], [outputs[-1]], img.shape, amap_mode='a')
            anomaly_map = gaussian_filter(anomaly_map, sigma=4)
            ano_list=[]
            # anomaly_map1 = anomaly_map *255
            print(np.max(anomaly_map))
            if np.max(anomaly_map) <0.4:
                print(_class_+'类'+str(count)+'号图像未发现异常')
            else:
                ano_list.append(count)
                print(_class_+'类'+str(count)+'号图像发现异常')
            # ano_map = min_max_norm(anomaly_map)
            # ano_map = cvt2heatmap(ano_map*255)
            ano_map = cvt2heatmap(anomaly_map*255)
            img = cv2.cvtColor(img.permute(0, 2, 3, 1).cpu().numpy()[0] * 255, cv2.COLOR_BGR2RGB)
            img = np.uint8(min_max_norm(img)*255)
            # 设定阈值太低会导致误报，需要实验来研判
            anomaly_map[anomaly_map > 0.4] = 255
            anomaly_map[anomaly_map <= 0.4] = 0
            if not os.path.exists('./results_all/'+_class_):
               os.makedirs('./results_all/'+_class_)
            cv2.imwrite('./results_all/'+_class_+'/'+str(count)+'_'+'org.png',img)
            ano_map = show_cam_on_image(img, ano_map)
            cv2.imwrite('./results_all/'+_class_+'/'+str(count)+'_'+'ad.png', ano_map)
            cv2.imwrite('./results_all/'+_class_+'/'+str(count)+'_'+'gt.png', anomaly_map)
            count += 1

def xml_writer(save_path,dict):

    # 创建根节点
    root = ET.Element('Doc')
    # root.set("time","total","result")
    
    # 子节点
    # 时间
    time = ET.SubElement(root, "Time")
    time.text = dict["Time"]
    
    # 文件名
    filename = ET.SubElement(root, "FileName")
    filename.text = dict['FileName']
    
    # 运行的模型
    model_name = ET.SubElement(root, "ModelName")
    model_name.text = dict['ModelName']
    # 是否异常情况
    fault = ET.SubElement(root, "Fault")
    fault.text = dict['Fault']
    
    if dict['Fault']==0:
        pass
    else:
        fault_type = ET.SubElement(fault, "FaultType")
        fault_type.text = dict['FaultType']
        fault_count = ET.SubElement(fault, "FaultCount")
        fault_area_xywh = ET.SubElement(fault, "FaultAreaXYWH")
        
        fault_count.text = str(dict['FaultCount'])
        fault_area_xywh.text = str(dict['FaultAreaXYWH'])



    tree=ET.ElementTree(root)
    tree.write(os.path.join(save_path, dict['FileName']+'.xml'), encoding='utf-8', xml_declaration=True)


if __name__=="__main__":
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # item_list = ['carpet', 'bottle', 'hazelnut', 'leather', 'cable', 'capsule', 'grid', 'pill',
    #               'transistor', 'metal_nut', 'screw','toothbrush', 'zipper', 'tile', 'wood']
    item_list = ['bogie_2']
    # logfile=os.path.join(data_input["OutputLogDirectory"],time.strftime('%Y%m%d-%H-%M-%S',time.localtime())+'.log')
    logfile=os.path.join('./log',time.strftime('%Y%m%d-%H-%M-%S',time.localtime())+'.log')
    logging.basicConfig(
        filename=logfile,
        encoding='utf-8',
        level=logging.DEBUG,
        datefmt='[%Y-%m-%d %H:%M:%S]',
        format='%(asctime)s %(levelname)s %(message)s')
    
    logging.info("开始检测螺栓松动")
    
    # item_list = ['bogie_2']
    for i in item_list:
        visualization(i)