import torch
import numpy as np
import os
import random
from sklearn.preprocessing import MinMaxScaler
from natsort import ns, natsorted

def mydata():
    src_path = 'input/npy_file/'
    dst_path = 'data/modelnet40_normal_resampled/'
    path_list = os.listdir(src_path)
    for npy in path_list:
        npy_file = src_path + npy
        data_folder = dst_path + npy.split(".")[0]
        os.mkdir(data_folder)
        file_name = np.load(npy_file, encoding = "latin1")
        index, abnormal = 0, 0
        for i in range(file_name.shape[0]):
            a = (file_name[i][:,0:2])*0.6
            b = (file_name[i][:,[-1]])*0.6
            new_ab = np.column_stack((a, b))
            txt_file = data_folder + '/' + 'frames'+  str('_') + "%04d"%(index) + '.txt'
            file = open(txt_file, 'w')
            for j in range(new_ab.shape[0]):
                coordinate_x, coordinate_y, coordinate_z = str(new_ab[j][0]), str(new_ab[j][1]), str(new_ab[j][2])
                info = coordinate_x + ',' + coordinate_y + ',' + coordinate_z  + ',' + '0.8' + ',' + '-0.5' + ',' + '-0.04' + '\n'
                file.write(info)
            file.close()
            index += 1
        print('the txt files got:', file_name.shape[0])


if __name__=="__main__":
    mydata()




