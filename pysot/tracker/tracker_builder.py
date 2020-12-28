# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from pysot.core.config import cfg
from pysot.tracker.siamrpn_tracker import SiamRPNTracker
from pysot.tracker.sfsiamrpn_tracker import SFSiamRPNTracker
from pysot.tracker.cf_tracker import CFTracker
from pysot.tracker.sfcf_tracker import SFCFTracker


TRACKS = {
          'SiamRPNTracker': SiamRPNTracker,
          'SFSiamRPNTracker': SFSiamRPNTracker,
          #'SiamTracker': SiamTracker,
          #'SiamFTracker': SiamFTracker,
          'CFTracker':CFTracker,
          'SFCFTracker':SFCFTracker
         }


def build_tracker(model,lambda_u=0.1,lr_u=0.1,lambda_v=0.1,lr_v=0.1,xpad=0.5,zpad=1.5):
    if cfg.TRACK.TYPE in ['CFTracker','SFCFTracker']:
        return TRACKS[cfg.TRACK.TYPE](lambda_u=lambda_u,lr_u=lr_u,lambda_v=lambda_v,lr_v=lr_v, x_padding=xpad,z_ratio=zpad)
    else:
        return TRACKS[cfg.TRACK.TYPE](model)
