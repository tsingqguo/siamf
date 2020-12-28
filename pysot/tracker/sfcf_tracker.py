    # Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from pysot.core.config import cfg

from pysot.pycftrackers.cftracker.sfkcf import SFKCF
from pysot.pycftrackers.cftracker.dsst import DSST
from pysot.pycftrackers.cftracker.staple import Staple
from pysot.pycftrackers.cftracker.eco import ECO
from pysot.pycftrackers.cftracker.bacf import BACF
from pysot.pycftrackers.cftracker.csrdcf import CSRDCF
from pysot.pycftrackers.cftracker.sfsamf import SFSAMF
from pysot.pycftrackers.cftracker.strcf import STRCF
from pysot.pycftrackers.lib.eco.config import otb_deep_config,otb_hc_config,otb_sf_deep_config
from pysot.pycftrackers.cftracker.config import staple_config,ldes_config,dsst_config,csrdcf_config,mkcf_up_config,mccth_staple_config

class SFCFTracker():

    def __init__(self,model=None,lambda_u=0.1,lr_u=0.1,lambda_v=0.1,lr_v=0.1,x_padding=0.5,z_ratio=1.5):

        self.tracker_type = cfg.TRACK.CF_TYPE

        if self.tracker_type=='SFDSST':
            self.tracker=DSST(dsst_config.DSSTConfig())
        elif self.tracker_type=='SFStaple':
            self.tracker=Staple(config=staple_config.StapleConfig())
        elif self.tracker_type=='SFStaple-CA':
            self.tracker=Staple(config=staple_config.StapleCAConfig())
        elif self.tracker_type=='SFKCF_CN':
            self.tracker=SFKCF(x_padding = x_padding,z_ratio=z_ratio,features='cn',kernel='gaussian')
        elif self.tracker_type=='SFKCF_GRAY':
            self.tracker=SFKCF(x_padding = x_padding,z_ratio=z_ratio,features='gray',kernel='gaussian')
        elif self.tracker_type=='SFKCF_HOG':
            self.tracker=SFKCF(x_padding = x_padding,z_ratio=z_ratio,features='hog',kernel='gaussian')
        elif self.tracker_type=='SFDCF_GRAY':
            self.tracker=SFKCF(x_padding = x_padding,z_ratio=z_ratio,features='gray',kernel='linear')
        elif self.tracker_type=='SFDCF_COLOR':
            self.tracker=SFKCF(lambda_u=lambda_u,lr_u=lr_u,lambda_v=lambda_v,lr_v=lr_v,\
                               x_padding = x_padding,z_ratio=z_ratio,features='color',kernel='linear')
        elif self.tracker_type=='SFDCF_HOG':
            self.tracker=SFKCF(lambda_u=lambda_u,lr_u=lr_u,lambda_v=lambda_v,lr_v=lr_v,\
                               x_padding = x_padding,z_ratio=z_ratio,features='hog',kernel='linear')
        elif self.tracker_type=='SFDCF_SFRES50':
            self.tracker=SFKCF(lambda_u=lambda_u,lr_u=lr_u,lambda_v=lambda_v,lr_v=lr_v,\
                               x_padding = x_padding,z_ratio=z_ratio,features='sfres50',kernel='linear')
        elif self.tracker_type=='SFECO-HC':
            self.tracker=ECO(config=otb_hc_config.OTBHCConfig())
        elif self.tracker_type=='SFECO':
            self.tracker=ECO(config=otb_sf_deep_config.OTBDeepConfig())
        elif self.tracker_type=='SFBACF':
            self.tracker=BACF()
        elif self.tracker_type=='SFCSRDCF':
            self.tracker=CSRDCF(config=csrdcf_config.CSRDCFConfig())
        elif self.tracker_type=='SFSAMF':
            self.tracker=SFSAMF(x_padding=x_padding,z_ratio=z_ratio)
        elif self.tracker_type=='SFDSST-LP':
            self.tracker=DSST(dsst_config.DSSTLPConfig())
        elif self.tracker_type=='SFSTRCF':
            self.tracker=STRCF()
        else:
            raise NotImplementedError

    def init(self, im_narray, bbox):
        init_gt=tuple(bbox)
        self.tracker.init(im_narray,init_gt)

    def track(self, image):

        current_frame = image
        bbox,max_score = self.tracker.update(current_frame)
        #print("box{}".format(bbox))
        return {
                'bbox': bbox,
                'best_score': max_score
               }



