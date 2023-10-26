# 第三方依赖库
import torch
import numpy as np
from scipy.ndimage import gaussian_filter
import cv2
import torch.functional as F
from dataset import MVTecDataset, InferDataset
from resnet import resnet18, resnet34, resnet50, wide_resnet50_2
from de_resnet import de_resnet18, de_resnet34, de_wide_resnet50_2, de_resnet50
from torch.utils.data import DataLoader
from dataset import get_data_transforms, load_data,get_data_transforms_for_blackbox

# python自带的标准库
import xml.etree.ElementTree as ET
import os
import json
import logging
import time
import random

#获取json文件内容，不用改
def get_json(path):
    try:
        f=open(path,'r') 
        data = json.load(f)
        # print('解析JSON成功')
        return data
    except FileNotFoundError:
        print("无法打开:%s",path)
        exit(1)
    except json.JSONDecodeError:
        print("无法解析json文件")
        exit(3)

#测试必要的路径，不用改
def check_resource(data,key):
    # data_input=data['input']
    try: 
        key in data
        try:
            os.path.exists(data[key])
        except FileNotFoundError:
            print("路径错误:%s",data[key])    
            exit(1)
    except KeyError:
        print("缺少参数:%s",key)
        exit(2)
    return True

#写xml，不用改
def xml_writer(dict):

    # 创建根节点
    root = ET.Element('Doc')
    # 子节点
    # 时间
    xmltime = ET.SubElement(root, "Time")
    xmltime.text = time.strftime('%Y-%m-%d-%H:%M:%S',time.localtime())
    # 文件名，即检测的图像文件的文件名
    image_filename = ET.SubElement(root, "ImageFileName")
    image_filename.text = dict['ImageFileName']
    # 运行的模型文件名
    model_name = ET.SubElement(root, "ModelName")
    model_name.text = dict['ModelName']
    # 是否异常情况
    fault = ET.SubElement(root, "Fault")
    fault.text = str(dict['Fault'])
    # 异常情况记录，无异常则跳过
    if dict['Fault']==0:
        pass
    # 有异常情况则记录类型数量位置
    else:
        fault_type = ET.SubElement(fault, "FaultType")
        fault_count = ET.SubElement(fault, "FaultCount")
        fault_area_xywh = ET.SubElement(fault, "FaultAreaXYWH")
        
        fault_type.text = str(dict['FaultType'])
        fault_count.text = str(dict['FaultCount'])
        fault_area_xywh.text = str(dict['FaultAreaXYWH'])

    tree=ET.ElementTree(root)
    tree.write(dict['xmlfilename'],encoding='utf-8',xml_declaration=True)
    return dict['xmlfilename']
    
#初始化流程，不用改
def init(path):
    data=get_json(path)
    data_input=data['Arguments']
    #必要的参数
    keys=["ModelDirectory","InputImageDirectory","InputImageFileNames","InputReportDirectory"]
    optional_keys=["OutputImageDirectory","OutputReportDirectory","OutputLogDirectory"]
    report = {'ErrorCode':'','ErrorMessage':'','OutputReportDir':''}
    
    if optional_keys[0] in data_input:
        output_path=data_input[optional_keys[0]]
    else:
        output_path='./output'
    #读取信息完成后开始初始化log
    if data_input['OutputLogDirectory'] == '':
        data_input['OutputLogDirectory'] = './log'
    if data_input['OutputReportDirectory']=='':
        data_input['OutputReportDirectory']='./report'
    
    if not os.path.exists(data_input['OutputLogDirectory']):
        os.makedirs(data_input['OutputLogDirectory'])
    if not os.path.exists(data_input['OutputReportDirectory']):
        os.makedirs(data_input['OutputReportDirectory'])
    data['Arguments']=data_input
    return data

#创建异常检测图像用
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

#将cam与源图像混合
def show_cam_on_image(img, anomaly_map):
    #if anomaly_map.shape != img.shape:
    #    anomaly_map = cv2.applyColorMap(np.uint8(anomaly_map), cv2.COLORMAP_JET)
    cam = np.float32(anomaly_map)/255 + np.float32(img)/255
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)

#归一化
def min_max_norm(image):
    a_min, a_max = image.min(), image.max()
    return (image-a_min)/(a_max - a_min)

#将灰度图转换为热力图
def cvt2heatmap(gray):
    heatmap = cv2.applyColorMap(np.uint8(gray), cv2.COLORMAP_JET)
    return heatmap

#获取异常区域
def get_gt_reigon(predict_gt):
    num_labels,labels=cv2.connectedComponents(predict_gt)
    # print(num_labels)
    positions=[]
    if num_labels==1:
        return 0,positions
    for i in range(1,num_labels):
        locs=np.where(labels == i)
        positions.append([locs[1].min(), locs[0].min(), locs[1].max(), locs[0].max()])
    return num_labels,positions

#绘制异常检测的框
def draw_rectangle(image,positions):
    for position in positions:
        cv2.rectangle(image, (position[1], position[0]), (position[3], position[2]), (0, 0, 255), 2)
    return image


#深度学习算法的推理
def predict(data_input):
    #检查cuda可用性
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logging.info('Using {} device'.format(device))
    
    data_transform, gt_transform = get_data_transforms_for_blackbox(1024,1024,1024)
    logging.debug('开始加载数据和模型权重')
    data_path = data_input['InputImageDirectory']
    ckp_path = data_input['ModelDirectory']
    data_path = MVTecDataset(root=data_path, transform=data_transform, gt_transform=gt_transform, phase="test")
    predict_dataloader = torch.utils.data.DataLoader(data_path, batch_size=1, shuffle=False)
    logging.debug('初始化模型')
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
    logging.debug('开始推理')
    count = 0
    FaultFlag=0
    FaultCount=0
    report=[]
    with torch.no_grad():
        for img, gt, label, _ in predict_dataloader:
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
            if np.max(anomaly_map) <0.5:
                print(str(count)+'号图像未发现异常')
            else:
                ano_list.append(count)
                logging.info('%d号图像发现异常',count)
                # print(str(count)+'号图像发现异常')
                # FaultCount=FaultCount+1
                FaultFlag = 1
                anomaly_map = min_max_norm(anomaly_map)
                anomaly_map[anomaly_map > 0.5] = 255
                anomaly_map[anomaly_map <= 0.5] = 0
                # locs=np.where(anomaly_map > 250)
                FaultCount,positions=get_gt_reigon(predict_gt=cv2.cvtColor(anomaly_map,cv2.COLOR_BGR2GRAY))
            # ano_map = min_max_norm(anomaly_map)
            # ano_map = cvt2heatmap(ano_map*255)
            for i in range(0,FaultCount):
                    report.append(
                        {
                            "CarNo":count,
                            #"CarNo":count/2
                            "detect_time":time.strftime('%Y-%m-%d-%H:%M:%S',time.localtime()),
                            "fault_type":'Anomly',
                            "fault_area":'',
                            "fault_site":'',
                            "fault_system":'',
                            "fault_desc":'',
                            "fault_unit":'',
                            "FaultImage":data_input['InputImageFileNames'][count],
                            "fault_sample_image":data_input['InputImageFileNames'][count],
                            "fault_standard_image":'',
                            "fault_area_x":'',
                            "fault_area_y":'',
                            "fault_row1":positions[i][0],
                            "fault_col1":positions[i][1],
                            "fault_row2":positions[i][2],
                            "fault_col2":positions[i][3],
                            # "image_count":'',
                            # "DetectImageLength":img.shape[1],
                            # "StandardImageLength":'',
                            # "DetectImage":data_input['InputImageFileNames'][count],
                            # "StandardImage":''
                        }
                    )        
                    
            ano_map = cvt2heatmap(anomaly_map*255)
            img = cv2.cvtColor(img.permute(0, 2, 3, 1).cpu().numpy()[0] * 255, cv2.COLOR_BGR2RGB)
            img = np.uint8(min_max_norm(img)*255)
            # 设定阈值太低会导致误报，需要实验来研判

            if not os.path.exists(data_input['OutputImageDirectory']):
                os.makedirs(data_input['OutputImageDirectory'])
            cv2.imwrite(os.path.join(data_input['OutputImageDirectory'],str(count)+'_'+'org.png'),img)
            ano_map = show_cam_on_image(img, ano_map)
            cv2.imwrite(os.path.join(data_input['OutputImageDirectory'],str(count)+'_'+'ad.png'), ano_map)
            cv2.imwrite(os.path.join(data_input['OutputImageDirectory'],str(count)+'_'+'gt.png'), anomaly_map)
            if FaultCount>0:
                image_with_rectangle=draw_rectangle(img,positions)
                cv2.imwrite(os.path.join(data_input['OutputImageDirectory'],str(count)+'_'+'rectangle.png'), image_with_rectangle)
            count += 1
            FaultCount=0
        
    
    #执行成功之后进行回报
    #写入到xml文件中的后处理部分
    filename=data_input['InputImageDirectory']
    modelname=data_input['ModelDirectory']
    FaultFlag=random.randint(0,2)
    reporttime=time.strftime('%Y-%m-%d-%H:%M:%S',time.localtime())
    
    # dict1={'ImageFileName':filename,
    #        'ReportTime':reporttime,
    #         'Fault':FaultFlag,
    #         'ModelName':modelname,
    #         'FaultCount':FaultCount,
    #         'FaultType':0,
    #         'FaultAreaXYWH':positions,
    #         'xmlfilename':os.path.join(data_input['OutputReportDirectory'],time.strftime('%Y%m%d-%H-%M-%S',time.localtime())+'.xml')
    # }
    #写入到tmp文件中的运行状态部分    
    return dict1

def Action(path):
    starttime=time.time()
    #初始化输入数据
    data=init(path)
    #获取需要的输入数据
    data_input=data['Arguments']
    #创建log文件
    logfile=os.path.join(data_input["OutputLogDirectory"],time.strftime('%Y%m%d-%H-%M-%S',time.localtime())+'.log')
    logging.basicConfig(
    filename=logfile,
    encoding='utf-8',
    level=logging.DEBUG,
    datefmt='[%Y-%m-%d %H:%M:%S]',
    format='%(asctime)s %(levelname)s %(message)s')
    #执行检测算法
    logging.info('开始执行%s',data_input['ModelDirectory']+'异常检测算法')
    dict1=predict(data_input) 
    logging.info('执行完成%s',data_input['ModelDirectory']+'异常检测算法')
    #写入xml文件
    reportfile=xml_writer(dict1)
    errorflag=0
    error_message=''
    output_path = data_input['OutputImageDirectory']
    report= {"ErrorCode":errorflag,"ErrorMessage":error_message,"OutputImageDirectory":output_path,"OutputReportFile":reportfile,"OutputLogFile":logfile}
    
    # report = {"ErrorCode":errorflag,"ErrorMessage":error_message,"OutputImageDirectory":output_path,"OutputReportFile":"","OutputLogFile":logfile}
    data['ReturnValues']=report
    with open(path,'w') as f:
        f.write(json.dumps(data))
    logging.info('执行完成，整体算法耗时%d,输出日志已保存到%s',time.time()-starttime,logfile)
        

if __name__ == "__main__":
    start = time.time()
    path=input()
    #从json文件中读取data
    data=get_json(path)
    #必要的参数
    keys=["ModelDirectory","InputImageDirectory","InputReportDirectory"]
    optional_keys=["OutputImageDirectory","OutputReportDirectory","OutputLogDirectory"]
    #检查参数中的路径是否可用
    for key in keys:
        if check_resource(data,key):
            continue
    data_input=data['Arguments']
    if optional_keys[0] in data_input:
        output_path=data_input[optional_keys[0]]
    else:
        output_path='./output'
    #读取信息完成后开始初始化log
    if data_input['OutputLogDirectory'] == '':
        data_input['OutputLogDirectory'] = './log'
    if optional_keys[0] in data:
        output_path=data[optional_keys[0]]
    else:
        output_path='./output'
    if not os.path.exists(data_input['OutputLogDirectory']):
        os.makedirs(data_input['OutputLogDirectory'])
    logfile=os.path.join(data_input["OutputLogDirectory"],time.strftime('%Y%m%d-%H-%M-%S',time.localtime())+'.log')
    # print(logfile)
    logging.basicConfig(
        filename=logfile,
        encoding='utf-8',
        level=logging.DEBUG,
        datefmt='[%Y-%m-%d %H:%M:%S]',
        format='%(asctime)s %(levelname)s %(message)s')
    logging.info("开始运行预处理算法")    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_path = data[keys[0]]
    data_path = data[keys[1]]
    data_transform, gt_transform = get_data_transforms(256, 256)
    # test_data = MVTecDataset(root=data_path, transform=data_transform, gt_transform=gt_transform, phase="test")
    infer_data = InferDataset(root=data_path, transform=data_transform)
    infer_dataloader = torch.utils.data.DataLoader(infer_data, batch_size=1, shuffle=False)

    encoder, bn = wide_resnet50_2(pretrained=True)
    encoder = encoder.to(device)
    bn = bn.to(device)
    encoder.eval()
    decoder = de_wide_resnet50_2(pretrained=False)
    decoder = decoder.to(device)
    ckp = torch.load(model_path)
    for k, v in list(ckp['bn'].items()):
        if 'memory' in k:
            ckp['bn'].pop(k)
    decoder.load_state_dict(ckp['decoder'])
    bn.load_state_dict(ckp['bn'])
    count = 0
    with torch.no_grad():
        for img in infer_dataloader:
            # if (label.item() == 0):
            #     continue
            decoder.eval()
            bn.eval()
            img = img.to(device)
            inputs = encoder(img)
            outputs = decoder(bn(inputs))
            
            anomaly_map, amap_list = cal_anomaly_map([inputs[-1]], [outputs[-1]], img.shape[-1], amap_mode='a')
            anomaly_map = gaussian_filter(anomaly_map, sigma=4)
            
            if np.max(anomaly_map)>=0.9:
                # print("发现异常")
                report = {"ErrorCode":1,"ErrorMessage":"发现异常","OutputImageDirectory":output_path,"OutputReportFile":"","OutputLogFile":logfile}
                # xml_path = os.path.join(output_path,'xml',)
                pass                
            else:
                report = {"ErrorCode":0,"ErrorMessage":'',"OutputImageDirectory":output_path,"OutputReportFile":"","OutputLogFile":logfile}
                data['ReturnValues']=report
                logging.info("螺栓检测结束")
                with open(path,'w') as f:
                    f.write(json.dumps(data))
                logging.info("输出类信息已保存到:%s",path)
                logging.info("运行结束，耗时%d秒",time.time()-start)
                # print("未发现异常")
