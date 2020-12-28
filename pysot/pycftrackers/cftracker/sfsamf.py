"""
Python re-implementation of "A Scale Adaptive Kernel Correlation Filter Tracker with Feature Integration"
@inproceedings{li2014scale,
  title={A scale adaptive kernel correlation filter tracker with feature integration},
  author={Li, Yang and Zhu, Jianke},
  booktitle={European conference on computer vision},
  pages={254--265},
  year={2014},
  organization={Springer}
}
"""
import numpy as np
import cv2
from scipy.ndimage import map_coordinates
from pysot.pycftrackers.lib.utils import cos_window,gaussian2d_rolled_labels,gaussian2d_labels
from pysot.pycftrackers.lib.fft_tools import fft2,ifft2
from .base import BaseCF
from .feature import extract_hog_feature,extract_cn_feature
from pysot.utils.siamf import Trans,CalTrans
import torch
import torch.nn.functional as F
import visdom

class SFSAMF(BaseCF):
    def __init__(self,lr_u=0.2,lr_v=0.2,lambda_u=0.1,lambda_v=0.1, kernel='gaussian',x_padding=0.5,z_ratio=1.2):
        super(SFSAMF).__init__()
        self.x_padding = x_padding
        self.z_padding = z_ratio*x_padding
        self.lambda_ = 1e-4
        self.output_sigma_factor=0.1
        self.interp_factor=0.01
        self.kernel_sigma=0.5
        self.cell_size=4
        self.kernel=kernel
        self.resize=False

        self.U = None
        self.V = None
        self.lr_u = lr_u
        self.lr_v = lr_v
        self.lambda_v = lambda_v
        self.lambda_u = lambda_u
        self.z_padding = z_ratio*x_padding
        #self.vis = None
        self.vis = visdom.Visdom()

    def init(self,first_frame,bbox):
        assert len(first_frame.shape)==3 and first_frame.shape[2]==3
        bbox = np.array(bbox).astype(np.int64)
        x0, y0, w, h = tuple(bbox)
        if w*h>=100**2:
            self.resize=True
            x0,y0,w,h=x0/2,y0/2,w/2,h/2
            first_frame=cv2.resize(first_frame,dsize=None,fx=0.5,fy=0.5).astype(np.uint8)

        self.x_crop_siz = (int(np.floor(w * (1 + self.x_padding))), int(np.floor(h * (1 + self.x_padding))))# for vis
        self._center = (x0 + w / 2,y0 + h / 2)
        self.w, self.h = w, h
        self.x_window_size=(int(np.floor(w*(1+self.x_padding)))//self.cell_size,int(np.floor(h*(1+self.x_padding)))//self.cell_size)
        self.x_cos_window = cos_window(self.x_window_size)


        self.search_size=np.linspace(0.985,1.015,7)
        s=np.sqrt(w*h)*self.output_sigma_factor/self.cell_size
        self.x_gaus = gaussian2d_labels(self.x_window_size, s)
        self.target_sz=(w,h)

        patch = cv2.getRectSubPix(first_frame,self.x_crop_siz, self._center)
        patch = cv2.resize(patch, dsize=self.x_crop_siz)
        hc_features=self.get_features(patch,self.cell_size)
        hc_features=hc_features*self.x_cos_window[:,:,None]

        self.x1 = torch.from_numpy(hc_features.astype(np.float32)).cuda() #

        if self.vis is not None:
            self.vis.image(self.x1.permute(2,0,1)[0:3,:,:],win='template')

    def update(self,current_frame):
        if self.resize:
            current_frame=cv2.resize(current_frame,dsize=None,fx=0.5,fy=0.5).astype(np.uint8)
        response=None

        # Conduct V transformation over target template:
        if self.V is not None:
            x_ = Trans(self.x1,self.V,self.lr_v)
        else:
            self.V = CalTrans(self.x1, self.x1, self.lambda_v)
            x_ = self.x1

        self.z_crop_siz = np.round((self.target_sz[1]*(1+self.z_padding),
                    self.target_sz[0]*(1+self.z_padding)))
        for i in range(len(self.search_size)):
            tmp_sz=(self.target_sz[0]*(1+self.z_padding)*self.search_size[i],
                    self.target_sz[1]*(1+self.z_padding)*self.search_size[i])
            patch=cv2.getRectSubPix(current_frame,(int(np.round(tmp_sz[0])),int(np.round(tmp_sz[1]))),self._center)
            patch = cv2.resize(patch, self.z_crop_siz)
            hc_features=self.get_features(patch,self.cell_size)
            self.z_cos_window = cos_window(np.round(tmp_sz)//self.cell_size)
            hc_features=hc_features*self.z_cos_window[:,:,None]

            z = torch.from_numpy(hc_features.astype(np.float32)).cuda()

            # Conduct U transformation over search region:
            if self.U is not None:
                if z.size() != self.U.size()[:-1]:
                    raise NotImplementedError
                z_ = Trans(z, self.U, self.lr_u)
            else:
                z_ = z

            padding = [int(np.ceil((self.x_window_size[1]-1) / 2)), int(np.ceil((self.x_window_size[0]-1) / 2))]
            if response is None:
                response = F.conv2d(z_.permute(2,0,1).unsqueeze(0),x_.permute(2,0,1).unsqueeze(0),padding=padding).squeeze(0).squeeze(0)
                response = response[:,:,np.newaxis]
            else:
                response_=F.conv2d(z_.permute(2,0,1).unsqueeze(0),x_.permute(2,0,1).unsqueeze(0),padding=padding).squeeze(0).squeeze(0)
                response = np.concatenate((response,response_[:,:,np.newaxis]),
                                        axis=2)

        if self.vis is not None:
            self.vis.image(x_.permute(2,0,1)[0:3,:,:], win='updated_template')
            self.vis.image(z.permute(2,0,1)[0:3,:,:], win='search region')
            self.vis.image(z_.permute(2,0,1)[0:3,:,:], win='updated search region')

        delta_y,delta_x,sz_id = np.unravel_index(np.argmax(response, axis=None), response.shape)
        self.sz_id=sz_id

        if delta_y+1>self.tmp_sz[1]/2:
            delta_y=delta_y-self.tmp_sz[1]
        if delta_x+1>self.tmp_sz[0]/2:
            delta_x=delta_x-self.tmp_sz[0]

        self.target_sz = (self.target_sz[0] * self.search_size[self.sz_id],
                          self.target_sz[1] * self.search_size[self.sz_id])
        tmp_sz=(self.target_sz[0]*(1+self.x_padding),
                self.target_sz[1]*(1+self.x_padding))
        current_size_factor=tmp_sz[0]/self.x_crop_siz[0]
        x,y=self._center
        x+=current_size_factor*self.cell_size*delta_x
        y+=current_size_factor*self.cell_size*delta_y
        self._center=(x,y)

        patch = cv2.getRectSubPix(current_frame, (int(np.round(tmp_sz[0])), int(np.round(tmp_sz[1]))), self._center)
        patch=cv2.resize(patch,self.x_crop_siz)
        hc_features=self.get_features(patch, self.cell_size)
        new_x=self.x_cos_window[:,:,None]*hc_features
        new_x_ = torch.from_numpy(new_x.astype(np.float32)).cuda()
        self.V = CalTrans(self.x1,new_x_, self.lambda_v)

        # new_z
        tmp_sz = (self.target_sz[0] * (1 + self.z_padding),
                  self.target_sz[1] * (1 + self.z_padding))
        self.z_window_size=(int(np.round(tmp_sz[0]))//self.cell_size,int(np.round(tmp_sz[1]))//self.cell_size)
        self.z_cos_window = cos_window(self.z_window_size)

        s = np.sqrt(self.target_sz[0] * self.target_sz[1]) * self.output_sigma_factor // self.cell_size
        self.z_gaus = gaussian2d_labels(self.z_window_size, s)

        patch = cv2.getRectSubPix(current_frame, (int(np.round(tmp_sz[0])), int(np.round(tmp_sz[1]))), self._center)
        hc_features = self.get_features(patch, self.cell_size)
        hc_features = hc_features * self.x_cos_window[:, :, None]
        new_z = torch.from_numpy(hc_features.astype(np.float32)).cuda()
        new_z_ = np.multiply(np.repeat(self.z_gaus[:,:,np.newaxis],new_z.shape[2],axis=2),new_z)
        new_z_ = torch.from_numpy(new_z_).cuda()
        self.U = CalTrans(new_z, new_z_, self.lambda_u)

        bbox=[(self._center[0] - self.target_sz[0] / 2), (self._center[1] - self.target_sz[1] / 2),
                self.target_sz[0], self.target_sz[1]]
        if self.resize is True:
            bbox=[ele*2 for ele in bbox]

        max_score = response.max()

        return bbox,max_score

    def _kernel_correlation(self, xf, yf, kernel='gaussian'):
        if kernel== 'gaussian':
            N=xf.shape[0]*xf.shape[1]
            xx=(np.dot(xf.flatten().conj().T,xf.flatten())/N)
            yy=(np.dot(yf.flatten().conj().T,yf.flatten())/N)
            xyf=xf*np.conj(yf)
            xy=np.sum(np.real(ifft2(xyf)),axis=2)
            kf = fft2(np.exp(-1 / self.kernel_sigma ** 2 * np.clip(xx+yy-2*xy,a_min=0,a_max=None) / np.size(xf)))
        elif kernel== 'linear':
            kf= np.sum(xf*np.conj(yf),axis=2)/np.size(xf)
        else:
            raise NotImplementedError
        return kf

    def get_features(self,img,cell_size):
        hog_feature=extract_hog_feature(img,cell_size)
        cn_feature=extract_cn_feature(img,cell_size)
        return np.concatenate((hog_feature,cn_feature),axis=2)

    """
     def warpimg(self,img,p,sz):
        w,h=sz
        x,y=np.meshgrid(np.arange(w)-w/2+0.5,np.arange(h)-h/2)
        pos=np.reshape(np.concatenate((np.ones((w*h,1)),x.ravel()[:,np.newaxis],y.ravel()[:,np.newaxis]),axis=1).dot(
            np.array([[p[0],p[1]],[p[2],p[4]],[p[3],p[5]]])),(h,w,2),order='C')
        c=img.shape[2]
        wimg=np.zeros((h,w,c))
        for i in range(c):
            wimg[:,:,i]=self.interp2(img[:,:,i],pos[:,:,1],pos[:,:,0])
        wimg[np.isnan(wimg)]=0
        return wimg

    def interp2(self,img,Xq,Yq):
        wimg=map_coordinates(img,[Xq.ravel(),Yq.ravel()],order=1,mode='constant')
        wimg=wimg.reshape(Xq.shape)
        return wimg

    def affparam2mat(self,p):
        
        # converts 6 affine parameters to a 2x3 matrix
        # :param p [dx,dy,sc,th,sr,phi]'
        # :return: q [q(1),q(3),q(4);q(2),q(5),q(6)]
        _,_,s,th,r,phi=p
        cth,sth=np.cos(th),np.sin(th)
        cph,sph=np.cos(phi),np.sin(phi)
        ccc=cth*cph*cph
        ccs=cth*cph*sph
        css=cth*sph*sph
        scc=sth*cph*cph
        scs=sth*cph*sph
        sss=sth*sph*sph
        q0=p[0]
        q1=p[1]
        q2=s*(ccc+scs+r*(css-scs))
        q3=s*(r*(ccs-scc)-ccs-sss)
        q4=s*(scc-ccs+r*(ccs+sss))
        q5=s*(r*(ccc+scs)-scs+css)
        return [q0,q1,q2,q3,q4,q5]

    def mex_resize(self, img, sz,method='auto'):
        sz = (int(sz[0]), int(sz[1]))
        src_sz = (img.shape[1], img.shape[0])
        if method=='antialias':
            interpolation=cv2.INTER_AREA
        elif method=='linear':
            interpolation=cv2.INTER_LINEAR
        else:
            if sz[1] > src_sz[1]:
                interpolation = cv2.INTER_LINEAR
            else:
                interpolation = cv2.INTER_AREA
        img = cv2.resize(img, sz, interpolation=interpolation)
        return img
    """




