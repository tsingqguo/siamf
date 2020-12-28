# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import os

import cv2
import torch
import numpy as np

from pysot.core.config import cfg
from pysot.models.model_builder import ModelBuilder
from pysot.tracker.tracker_builder import build_tracker
from pysot.utils.bbox import get_axis_aligned_bbox
from pysot.utils.model_load import load_pretrain
from toolkit.datasets import DatasetFactory
from toolkit.utils.region import vot_overlap, vot_float2str
import visdom
import matplotlib

matplotlib.use('Agg')
from matplotlib import pyplot as plt


parser = argparse.ArgumentParser(description='siamrpn tracking')
parser.add_argument('--dataset', type=str,
        help='datasets')
parser.add_argument('--prefix', default='',type=str,
        help='save prefix')
parser.add_argument('--xpad', default=0.5,type=float,
        help='xpadding')
parser.add_argument('--zpad', default=1.2,type=float,
        help='zpadding')
parser.add_argument('--lambda_u', default=0.1,type=float,
        help='lambda_u')
parser.add_argument('--lambda_v', default=0.1,type=float,
        help='lambda_v')
parser.add_argument('--lr_u', default=0.2,type=float,
        help='lr_u')
parser.add_argument('--lr_v', default=0.2,type=float,
        help='lr_v')
parser.add_argument('--gpuid', default='0',type=str,
        help='gpuid')
parser.add_argument('--config', default='', type=str,
        help='config file')
parser.add_argument('--snapshot', default='', type=str,
        help='snapshot of models to eval')
parser.add_argument('--video', default='', type=str,
        help='eval one special video')
parser.add_argument('--vis', action='store_true',
        help='whether visualzie result')
parser.add_argument('--check_exists', action='store_true',
        help='whether check exists')
parser.add_argument('--cffeat', default=0,type=int,
        help='cffeat')
parser.add_argument('--sfinterval',default=20,type=int,
        help='cffeat')

args = parser.parse_args()

torch.set_num_threads(1)

def vis_image(img, ax=None):
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
    # CHW -> HWC
    #img = img.transpose((1, 2, 0))
    ax.imshow(img.astype(np.uint8))
    return ax

def vis_bbox(img, bbox, label=[0,1], score=None, ax=None):
    label_names = ['gt','pred'] #None #list(VOC_BBOX_LABEL_NAMES) + ['bg']
    colors = ['red','green']
    # add for index `-1`
    if label is not None and not len(bbox) == len(label):
        raise ValueError('The length of label must be same as that of bbox')
    if score is not None and not len(bbox) == len(score):
        raise ValueError('The length of score must be same as that of bbox')

    #画到图片上
    # Returns newly instantiated matplotlib.axes.Axes object if ax is None
    ax = vis_image(img, ax=ax)

    # If there is no bounding box to display, visualize the image and exit.
    if len(bbox) == 0:
        return ax
    #画框
    for i, bb in enumerate(bbox):
        xy = (bb[0], bb[1])
        height = bb[3] # - bb[0]
        width = bb[2] # - bb[1]
        ax.add_patch(plt.Rectangle(
            xy, width, height, fill=False, edgecolor=colors[i], linewidth=2))

        caption = list()
        #这个框的类别标签
        if label is not None and label_names is not None:
            lb = label[i]
            if not (-1 <= lb < len(label_names)):  # modfy here to add backgroud
                raise ValueError('No corresponding name is given')
            caption.append(label_names[lb])
        #这个框的分值
        if score is not None:
            sc = score[i]
            caption.append('{:.2f}'.format(sc))
        #画上去
        if len(caption) > 0:
            ax.text(bb[1], bb[0],
                    ': '.join(caption),
                    style='italic',
                    bbox={'facecolor': 'white', 'alpha': 0.5, 'pad': 0})
    return ax

def main():

    # load config
    cfg.merge_from_file(args.config)

    cur_dir = os.path.dirname(os.path.realpath(__file__))
    dataset_root = os.path.join(cur_dir, '../testing_dataset', args.dataset)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpuid

    if args.cffeat>=0:
        cfg.TRACK.CF_FEAT=[args.cffeat]

    if cfg.TRACK.TYPE=='SFCFTracker' and cfg.TRACK.CF_TYPE=='SFDCF_SFRES50':
        args.prefix = str(cfg.TRACK.CF_FEAT)+'_'+args.prefix

    if args.sfinterval>=0:
        cfg.SIAMF.INTERVAL=args.sfinterval

    if cfg.TRACK.TYPE=='SiamRPNTracker' or cfg.TRACK.TYPE=='SFSiamRPNTracker' :
        # create model
        model = ModelBuilder()
        # load model
        model = load_pretrain(model, args.snapshot).cuda().eval()
    else:
        model=None

    # build tracker
    tracker = build_tracker(model,lambda_u=args.lambda_u,lr_u=args.lr_u,\
                            lambda_v=args.lambda_v,lr_v=args.lr_v,xpad=args.xpad,zpad=args.zpad)

    # create dataset
    dataset = DatasetFactory.create_dataset(name=args.dataset,
                                            dataset_root=dataset_root,
                                            load_img=False)

    if args.vis:
        vis = visdom.Visdom()

    model_name = args.snapshot.split('/')[-1].split('.')[0]
    total_lost = 0

    if args.dataset in ['VOT2016', 'VOT2018', 'VOT2019']:
        # restart tracking
        for v_idx, video in enumerate(dataset):
            if args.video != '':
                # test one special video
                if video.name != args.video:
                    continue

            frame_counter = 0
            lost_number = 0
            toc = 0
            pred_bboxes = []
            for idx, (img, gt_bbox) in enumerate(video):
                if len(gt_bbox) == 4:
                    gt_bbox = [gt_bbox[0], gt_bbox[1],
                       gt_bbox[0], gt_bbox[1]+gt_bbox[3]-1,
                       gt_bbox[0]+gt_bbox[2]-1, gt_bbox[1]+gt_bbox[3]-1,
                       gt_bbox[0]+gt_bbox[2]-1, gt_bbox[1]]
                tic = cv2.getTickCount()
                if idx == frame_counter:
                    cx, cy, w, h = get_axis_aligned_bbox(np.array(gt_bbox))
                    gt_bbox_ = [cx-(w-1)/2, cy-(h-1)/2, w, h]
                    tracker.init(img, gt_bbox_)
                    pred_bbox = gt_bbox_
                    pred_bboxes.append(1)
                elif idx > frame_counter:
                    outputs = tracker.track(img)
                    pred_bbox = outputs['bbox']
                    if cfg.MASK.MASK:
                        pred_bbox = outputs['polygon']
                    overlap = vot_overlap(pred_bbox, gt_bbox, (img.shape[1], img.shape[0]))
                    if overlap > 0:
                        # not lost
                        pred_bboxes.append(pred_bbox)
                    else:
                        # lost object
                        pred_bboxes.append(2)
                        frame_counter = idx + 5 # skip 5 frames
                        lost_number += 1
                else:
                    pred_bboxes.append(0)
                toc += cv2.getTickCount() - tic
                if idx == 0:
                    cv2.destroyAllWindows()
                if args.vis and idx > frame_counter:
                    cv2.polylines(img, [np.array(gt_bbox, np.int).reshape((-1, 1, 2))],
                            True, (0, 255, 0), 3)
                    if cfg.MASK.MASK:
                        cv2.polylines(img, [np.array(pred_bbox, np.int).reshape((-1, 1, 2))],
                                True, (0, 255, 255), 3)
                    else:
                        bbox = list(map(int, pred_bbox))
                        cv2.rectangle(img, (bbox[0], bbox[1]),
                                      (bbox[0]+bbox[2], bbox[1]+bbox[3]), (0, 255, 255), 3)
                    cv2.putText(img, str(idx), (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                    cv2.putText(img, str(lost_number), (40, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    cv2.imshow(video.name, img)
                    cv2.waitKey(1)
            toc /= cv2.getTickFrequency()
            # save results
            video_path = os.path.join('results', args.dataset, model_name,
                    'baseline', video.name)
            if not os.path.isdir(video_path):
                os.makedirs(video_path)
            result_path = os.path.join(video_path, '{}_001.txt'.format(video.name))
            with open(result_path, 'w') as f:
                for x in pred_bboxes:
                    if isinstance(x, int):
                        f.write("{:d}\n".format(x))
                    else:
                        f.write(','.join([vot_float2str("%.4f", i) for i in x])+'\n')
            print('({:3d}) Video: {:12s} Time: {:4.1f}s Speed: {:3.1f}fps Lost: {:d}'.format(
                    v_idx+1, video.name, toc, idx / toc, lost_number))
            total_lost += lost_number
        print("{:s} total lost: {:d}".format(model_name, total_lost))
    else:
        # OPE tracking
        for v_idx, video in enumerate(dataset):
            model_path = os.path.join('results', args.dataset, model_name + args.prefix)
            if not os.path.isdir(model_path):
                os.makedirs(model_path)
            result_path = os.path.join(model_path, '{}.txt'.format(video.name))

            if os.path.exists(result_path) and args.check_exists:
                continue

            if args.video != '':
                # test one special video
                if video.name != args.video:
                    continue
            toc = 0
            pred_bboxes = []
            scores = []
            track_times = []

            for idx, (img_org, gt_bbox_org) in enumerate(video):
                tic = cv2.getTickCount()
                # if (cfg.TRACK.TYPE!='SiamRPNTracker' or cfg.TRACK.TYPE!='SFSiamRPNTracker') and np.max(gt_bbox_org[2:])>=100:
                #     print("rescale the input")
                #     szrate = 0.25
                #     img = cv2.resize(img_org, None, fx=szrate,fy=szrate, interpolation = cv2.INTER_AREA)
                #     gt_bbox = (np.array(gt_bbox_org)*szrate).tolist()
                # else:
                img = img_org
                gt_bbox = gt_bbox_org
                szrate = 1.0

                if idx == 0:
                    cx, cy, w, h = get_axis_aligned_bbox(np.array(gt_bbox))
                    gt_bbox_ = [cx-(w-1)/2, cy-(h-1)/2, w, h]
                    tracker.init(img, gt_bbox_)
                    pred_bbox = (np.array(gt_bbox_)/szrate).tolist()
                    scores.append(None)
                    if 'VOT2018-LT' == args.dataset:
                        pred_bboxes.append([1])
                    else:
                        pred_bboxes.append(pred_bbox)
                else:
                    outputs = tracker.track(img,frameidx=idx)
                    pred_bbox = (np.array(outputs['bbox'])/szrate).tolist()
                    pred_bboxes.append(pred_bbox)
                    scores.append(outputs['best_score'])

                toc += cv2.getTickCount() - tic
                track_times.append((cv2.getTickCount() - tic)/cv2.getTickFrequency())
                if idx == 0:
                    cv2.destroyAllWindows()

                if args.vis and idx > 0:
                    bboxes=[]
                    bboxes.append(gt_bbox_org)
                    bboxes.append(pred_bbox)

                    _, ax = plt.subplots()
                    ax = vis_bbox(img_org,bboxes,ax=ax)
                    vis.matplot(plt,win='track results')
                    # gt_bbox = list(map(int, gt_bbox))
                    # pred_bbox = list(map(int, pred_bbox))
                    # cv2.rectangle(img, (gt_bbox[0], gt_bbox[1]),
                    #               (gt_bbox[0]+gt_bbox[2], gt_bbox[1]+gt_bbox[3]), (0, 255, 0), 3)
                    # cv2.rectangle(img, (pred_bbox[0], pred_bbox[1]),
                    #               (pred_bbox[0]+pred_bbox[2], pred_bbox[1]+pred_bbox[3]), (0, 255, 255), 3)
                    # cv2.putText(img, str(idx), (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                    # cv2.imshow(video.name, img)
                    #cv2.waitKey(1)
            toc /= cv2.getTickFrequency()

            # save results
            if 'VOT2018-LT' == args.dataset:
                video_path = os.path.join('results', args.dataset, model_name,
                        'longterm', video.name)
                if not os.path.isdir(video_path):
                    os.makedirs(video_path)
                result_path = os.path.join(video_path,
                        '{}_001.txt'.format(video.name))
                with open(result_path, 'w') as f:
                    for x in pred_bboxes:
                        f.write(','.join([str(i) for i in x])+'\n')
                result_path = os.path.join(video_path,
                        '{}_001_confidence.value'.format(video.name))
                with open(result_path, 'w') as f:
                    for x in scores:
                        f.write('\n') if x is None else f.write("{:.6f}\n".format(x))
                result_path = os.path.join(video_path,
                        '{}_time.txt'.format(video.name))
                with open(result_path, 'w') as f:
                    for x in track_times:
                        f.write("{:.6f}\n".format(x))
            elif 'GOT-10k' == args.dataset:
                video_path = os.path.join('results', args.dataset, model_name, video.name)
                if not os.path.isdir(video_path):
                    os.makedirs(video_path)
                result_path = os.path.join(video_path, '{}_001.txt'.format(video.name))
                with open(result_path, 'w') as f:
                    for x in pred_bboxes:
                        f.write(','.join([str(i) for i in x])+'\n')
                result_path = os.path.join(video_path,
                        '{}_time.txt'.format(video.name))
                with open(result_path, 'w') as f:
                    for x in track_times:
                        f.write("{:.6f}\n".format(x))
            else:
                model_path = os.path.join('results', args.dataset, model_name+args.prefix)
                if not os.path.isdir(model_path):
                    os.makedirs(model_path)
                result_path = os.path.join(model_path, '{}.txt'.format(video.name))
                with open(result_path, 'w') as f:
                    for x in pred_bboxes:
                        f.write(','.join([str(i) for i in x])+'\n')
            print('({:3d}) Video: {:12s} Time: {:5.1f}s Speed: {:3.1f}fps'.format(
                v_idx+1, video.name, toc, idx / toc))


if __name__ == '__main__':
    main()
