import cv2
import numpy as np
from pysot.pycftrackers.lib.eco.features.features import fhog,TableFeature
import torch
from pysot.core.config import cfg


def extract_hog_feature(img, cell_size=4):
    fhog_feature=fhog(img.astype(np.float32),cell_size,num_orients=9,clip=0.2)[:,:,:-1]
    return fhog_feature

def extract_pyhog_feature(img, cell_size=4):

    from pysot.pycftrackers.lib import fhog as pyfhog
    h,w=img.shape[:2]
    img=cv2.resize(img,(w+2*cell_size,h+2*cell_size))
    mapp = {'sizeX': 0, 'sizeY': 0, 'numFeatures': 0, 'map': 0}
    mapp = pyfhog.getFeatureMaps(img, cell_size, mapp)
    mapp = pyfhog.normalizeAndTruncate(mapp, 0.2)
    mapp = pyfhog.PCAFeatureMaps(mapp)
    size_patch = list(map(int, [mapp['sizeY'], mapp['sizeX'], mapp['numFeatures']]))
    FeaturesMap = mapp['map'].reshape(
        (size_patch[0],size_patch[1], size_patch[2]))
    return FeaturesMap


def extract_cn_feature(img,cell_size=1):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255 - 0.5
    cn = TableFeature(fname='cn', cell_size=cell_size, compressed_dim=11, table_name="CNnorm",
                      use_for_color=True)

    if np.all(img[:, :, 0] == img[:, :, 1]):
        img = img[:, :, :1]
    else:
        # # pyECO using RGB format
        img = img[:, :, ::-1]
    h,w=img.shape[:2]
    cn_feature = \
    cn.get_features(img, np.array(np.array([h/2,w/2]), dtype=np.int16), np.array([h,w]), 1, normalization=False)[
        0][:, :, :, 0]
    # print('cn_feature.shape:', cn_feature.shape)
    # print('cnfeature:',cn_feature.shape,cn_feature.min(),cn_feature.max())
    gray = cv2.resize(gray, (cn_feature.shape[1], cn_feature.shape[0]))[:, :, np.newaxis]
    out = np.concatenate((gray, cn_feature), axis=2)
    return out

def extract_sfres50_feature(model,img,cell_size=8.0):

    img = torch.from_numpy(img.astype(np.float32)).cuda()
    img = img.permute(2,0,1).unsqueeze(0)
    outs_ = model(img)
    outs__ = []
    for vout in cfg.TRACK.CF_FEAT:
        outs__.append(outs_[vout])
    outs_ = outs__
    outs = []
    for i, out_ in enumerate(outs_):
        if i<1:
            out_ = out_.detach().cpu().squeeze(0)
            out_ = out_.permute(1,2,0)
            outs.append(out_)

    outs = torch.cat(outs,dim=2).numpy()

    return outs


def extract_cn_feature_byw2c(patch, w2c):
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255 - 0.5
    gray = gray[:, :, np.newaxis]

    if np.all(patch[:,:,0]==patch[:,:,1]) and np.all(patch[:,:,0]==patch[:,:,2]):
        return gray

    b, g, r = cv2.split(patch)
    index_im = ( r//8 + 32 * g//8 + 32 * 32 * b//8)
    h, w = patch.shape[:2]
    w2c=np.array(w2c)
    out=w2c.T[index_im.flatten(order='F')].reshape((h,w,w2c.shape[0]))
    out=np.concatenate((gray,out),axis=2)
    return out






