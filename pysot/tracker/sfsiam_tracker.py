# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import 
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging 
import os
import json

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from pysot.core.config import cfg
from pysot.utils.anchor import Anchors
from pysot.tracker.base_tracker import SiameseTracker

class SiamFTracker(SiameseTracker):
    def __init__(self, model):
        super(SiamFTracker, self).__init__()
        self.score_size = (cfg.TRACK.INSTANCE_SIZE - cfg.TRACK.EXEMPLAR_SIZE) // \
                1 + cfg.TRACK.BASE_SIZE
        hanning = np.hanning(self.score_size)
        window = np.outer(hanning, hanning)
        self.window = np.tile(window.flatten(), self.anchor_num)
        self.model = model
        self.model.eval()

    def _convert_score(self, score):
        score = score.permute(1, 2, 3, 0).contiguous().view(2, -1).permute(1, 0)
        score = F.softmax(score, dim=1).data[:, 1].cpu().numpy()
        return score

    def _bbox_clip(self, cx, cy, width, height, boundary):
        cx = max(0, min(cx, boundary[1]))
        cy = max(0, min(cy, boundary[0]))
        width = max(10, min(width ,boundary[1]))
        height = max(10, min(height, boundary[0]))
        return cx, cy, width, height

    def init(self, img, bbox,lambda_v=cfg.SIAMF.LAMBDA_V,lambda_u=cfg.SIAMF.LAMBDA_U,lr_v=cfg.SIAMF.LR_V,lr_u=cfg.SIAMF.LR_U):
        """
        args:
            img(np.ndarray): BGR image
            bbox: (x, y, w, h) bbox
        """
        self.center_pos = np.array([bbox[0]+(bbox[2]-1)/2,
                                    bbox[1]+(bbox[3]-1)/2])
        self.size = np.array([bbox[2], bbox[3]])
       
        # calculate z crop size
        w_z = self.size[0] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        h_z = self.size[1] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        s_z = round(np.sqrt(w_z * h_z))

        # calculate channle average
        self.channel_average = np.mean(img , axis=(0,1))

        # get crop
        z_crop = self.get_subwindow(img, self.center_pos, 
                cfg.TRACK.EXEMPLAR_SIZE, s_z, self.channel_average)
        self.model.template(z_crop)

        # initialize Trans parameter
        if cfg.SIAMF.TRANS:
            rs = np.linspace(1-np.floor(cfg.TRACK.INSTANCE_SIZE/2), cfg.TRACK.INSTANCE_SIZE-np.floor(cfg.TRACK.INSTANCE_SIZE/2), cfg.TRACK.INSTANCE_SIZE)
            cs = rs
            x, y = np.meshgrid(rs,cs)
            hanning = np.hanning(cfg.TRACK.INSTANCE_SIZE)
            hann_window = np.outer(hanning, hanning)
            d = x*x+y*y
            tsigma =cfg.TRACK.EXEMPLAR_SIZE*100
            saliency_map=hann_window*np.exp(-0.5*d/(tsigma^2))
            self.saliency_map = torch.from_numpy(saliency_map).repeat(1,3,1,1).float()
            if cfg.CUDA:
                self.saliency_map=self.saliency_map.cuda()

        self.model.rpn_head.set_params(lambda_v,lambda_u,lr_v,lr_u)

    def track(self, img,x_crop=[],ispert=False):
        """
        args:
            img(np.ndarray): BGR image
        return:
            bbox(list):[x, y, width, height]
        """
        w_z = self.size[0] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        h_z = self.size[1] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        s_z = np.sqrt(w_z * h_z)
        scale_z = cfg.TRACK.EXEMPLAR_SIZE / s_z
        s_x = s_z * (cfg.TRACK.INSTANCE_SIZE / cfg.TRACK.EXEMPLAR_SIZE)

        if ispert==False:
            x_crop = self.get_subwindow(img, self.center_pos,
                                        cfg.TRACK.INSTANCE_SIZE,
                                        round(s_x), self.channel_average)

        outputs = self.model.track(x_crop)
        score = self._convert_score(outputs['cls'])
        pred_bbox = self._convert_bbox(outputs['loc'], self.anchors)
        
        def change(r):
            return np.maximum(r, 1. / r)

        def sz(w, h):
            pad = (w + h) * 0.5
            return np.sqrt((w + pad) * (h + pad))
            
        # scale penalty
        s_c = change(sz(pred_bbox[2,:], pred_bbox[3,:]) / 
                (sz(self.size[0]*scale_z, self.size[1]*scale_z))) 

        # aspect ratio penalty
        r_c = change((self.size[0]/self.size[1]) /
                     (pred_bbox[2,:]/pred_bbox[3,:]))
        penalty = np.exp(-(r_c * s_c - 1) * cfg.TRACK.PENALTY_K)
        pscore = penalty * score

        # window penalty 
        pscore = pscore * (1 - cfg.TRACK.WINDOW_INFLUENCE) + \
                self.window * cfg.TRACK.WINDOW_INFLUENCE
        best_idx = np.argmax(pscore)

        bbox = pred_bbox[:, best_idx] / scale_z
        lr = penalty[best_idx] * score[best_idx] * cfg.TRACK.LR
        
        cx = bbox[0] + self.center_pos[0]
        cy = bbox[1] + self.center_pos[1]

        # smooth bbox
        width = self.size[0] * (1 - lr) + bbox[2] * lr
        height = self.size[1] * (1 - lr) + bbox[3] * lr

        # clip boundary
        cx, cy, width, height = self._bbox_clip(cx, cy, width, height, img.shape[:2])

        # udpate state
        self.center_pos = np.array([cx, cy])
        self.size = np.array([width, height])

        if cfg.SIAMF.TRANS:
            # update Trans.V and U
              # crop the new template and suppressed search region
            w_z = self.size[0] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
            h_z = self.size[1] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
            s_z = np.sqrt(w_z * h_z)
            s_x = s_z * (cfg.TRACK.INSTANCE_SIZE / cfg.TRACK.EXEMPLAR_SIZE)
            zt_crop = self.get_subwindow(img, self.center_pos,
                    cfg.TRACK.EXEMPLAR_SIZE, s_z, self.channel_average)
            x_crop = self.get_subwindow(img, self.center_pos, cfg.TRACK.INSTANCE_SIZE,
                    round(s_x), self.channel_average)
            xb_crop = x_crop*self.saliency_map
            self.model.update(x_crop,xb_crop,zt_crop)


        bbox = [cx - width / 2,
                cy - height / 2,
                width,
                height]
        best_score = score[best_idx]

        return {
                'bbox': bbox,
                'best_score': best_score
               }


