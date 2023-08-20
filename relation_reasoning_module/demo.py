"""
Author: Benny
Date: Nov 2019
"""
from data_utils.ModelNetDataLoader import ModelNetDataLoader
import argparse
import numpy as np
import os
import torch
import logging
from tqdm import tqdm
import sys
import importlib
from natsort import ns, natsorted

import cv2
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))

def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('Testing')
    parser.add_argument('--use_cpu', action='store_true', default=False, help='use cpu mode')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size in training')
    parser.add_argument('--num_category', default=4, type=int,  help='the number of action level')  #4
    parser.add_argument('--num_point', type=int, default=17, help='Point Number')
    parser.add_argument('--log_dir', type=str, required=True, help='Experiment root')
    parser.add_argument('--use_normals', action='store_true', default=False, help='use normals')
    parser.add_argument('--use_uniform_sample', action='store_true', default=False, help='use uniform sampiling')
    parser.add_argument('--num_votes', type=int, default=3, help='Aggregate classification scores with voting')
    parser.add_argument('--data_folder', type=str, required=True, help='test data')
    return parser.parse_args()

def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc


def mydemo(model,data_dir):
    np.set_printoptions(threshold=np.inf)  # 不出现省略号
    np.set_printoptions(suppress=True)  # 不转换为科学计数法，保持原始格式
    data = data_dir
    video_name = data.split('/')[2]
    video_path = 'input/input_video/' + video_name + '.mp4'
    output_video =  'reba_' + video_name+ '.mp4'
    classifier = model.eval()
    datalist0 = os.listdir(data)
    datalist = natsorted(datalist0, alg=ns.PATH)
    results = []
    for i in range(len(datalist)):
        sample = data + datalist[i]
        input_txt = np.loadtxt(sample,delimiter=',').astype(np.float32)
        input_set = input_txt[0:17, :]
        input_set[:, 0:3] = pc_normalize(input_set[:, 0:3])
        input = torch.from_numpy(input_set[:,0:3])
        input  = input.reshape(1,17,3)
        # print('@@@@@@input.size=', input.size())
        if not args.use_cpu:
            input = input.cuda()
        input = input.transpose(2, 1)
        pred, _ = classifier(input)
        level1 = pred[0][0].cpu().numpy()
        level2 = pred[0][1].cpu().numpy()
        level3 = pred[0][2].cpu().numpy()
        level4 = pred[0][3].cpu().numpy()

        level = 1*level1 + 2*level2 + 3*level3 + 4*level4
        level= np.around(level,4)
        ori_text = "action_level:" + str(level)
        results.append(ori_text)


    cap = cv2.VideoCapture(video_path)
    out_file = os.path.abspath(output_video)
    print(video_path)
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames_num = cap.get(7)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_size = (frame_width, frame_height)
    video_writer = cv2.VideoWriter(out_file, fourcc, fps, frame_size)
    if len(results) > int(frames_num):
        difference = len(results) - int(frames_num)
        for k in range(difference):
            results.pop(-1)
    assert len(results) == int(frames_num)
    num = 0
    ind = 0
    while ind < frames_num:
        ind += 1
        ret, img = cap.read()
        word = results[num]
        cv2.putText(img, word, (750, 460), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        if num < (len(results) - 1):
            num += 1
        # cv2.imshow("image", result)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        video_writer.write(img)
    cap.release()


def main(args):
    def log_string(str):
        logger.info(str)
        print(str)

    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    '''CREATE DIR'''
    experiment_dir = 'log/classification/' + args.log_dir

    data_path = 'data/modelnet40_normal_resampled/'
    data_dir = data_path + args.data_folder

    '''LOG'''
    args = parse_args()
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/eval.txt' % experiment_dir)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)

    '''DATA LOADING'''
    log_string('Load dataset ...')


    '''MODEL LOADING'''
    num_class = args.num_category
    model_name = os.listdir(experiment_dir + '/logs')[0].split('.')[0]
    model = importlib.import_module(model_name)

    classifier = model.my_get_model(num_class, normal_channel=args.use_normals)
    if not args.use_cpu:
        classifier = classifier.cuda()

    checkpoint = torch.load(str(experiment_dir) + '/checkpoints/best_model.pth')
    classifier.load_state_dict(checkpoint['model_state_dict'])

    with torch.no_grad():
        mydemo(classifier.eval(),data_dir)



if __name__ == '__main__':
    args = parse_args()
    main(args)
