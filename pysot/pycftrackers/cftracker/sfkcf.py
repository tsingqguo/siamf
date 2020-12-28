
import numpy as np
import cv2
from pysot.pycftrackers.lib.utils import cos_window,gaussian2d_labels
from pysot.pycftrackers.lib.fft_tools import fft2,ifft2
from .base import BaseCF
from .feature import extract_hog_feature,extract_cn_feature,extract_sfres50_feature
from pysot.utils.siamf import Trans,CalTrans
from scipy import signal
import torch
import torch.nn.functional as F
import visdom
from pysot.utils.model_load import load_pretrain
from pysot.models.model_builder import ModelBuilder
from pysot.core.config import cfg


class SFKCF(BaseCF):
    def __init__(self, lr_u=0.2,lr_v=0.2,lambda_u=0.1,lambda_v=10.0,x_padding=0.5, z_ratio=1.2,features='gray', kernel='gaussian'):
        super(SFKCF).__init__()
        self.x_padding = x_padding
        self.lambda_ = 1e-4
        self.features = features
        self.w2c=None
        if self.features=='hog':
            self.interp_factor = 0.02
            self.sigma = 0.5
            self.cell_size=4
            self.output_sigma_factor=0.1

        elif self.features=='sfres50':

            self.interp_factor = 0.02
            self.sigma = 0.5
            self.cell_size=8.0
            self.output_sigma_factor=0.1
            model = ModelBuilder()
            model = load_pretrain(model, cfg.BACKBONE.PRETRAINED).backbone
            self.model = model.cuda().eval()

        elif self.features=='gray' or self.features=='color':

            self.interp_factor=0.075
            self.sigma=0.2
            self.cell_size=1
            self.output_sigma_factor=0.1

        elif self.features=='cn':
            self.interp_factor=0.075
            self.sigma=0.2
            self.cell_size=1
            self.output_sigma_factor=1./16
            self.padding=1

        else:
            raise NotImplementedError

        self.kernel=kernel
        self.U = None
        self.V = None
        self.lr_u = lr_u
        self.lr_v = lr_v
        self.lambda_v = lambda_v
        self.lambda_u = lambda_u
        self.z_padding = z_ratio*x_padding
        self.vis = None
        #self.vis = visdom.Visdom()

    def init(self,first_frame,bbox):

        assert len(first_frame.shape)==3 and first_frame.shape[2]==3
        self.U = None
        self.V = None
        if self.features=='gray':
            first_frame=cv2.cvtColor(first_frame,cv2.COLOR_BGR2GRAY)
        bbox = np.array(bbox).astype(np.int64)
        x0, y0, w, h = tuple(bbox)
        self.crop_size = (int(np.floor(w * (1 + self.x_padding))), int(np.floor(h * (1 + self.x_padding))))# for vis
        self._center = (np.floor(x0 + w / 2),np.floor(y0 + h / 2))
        self.w, self.h = w, h

        if self.features=='sfres50':
            self.x_window_size=(np.ceil(int(np.floor(w*(1+self.x_padding)))/self.cell_size),np.ceil(int(np.floor(h*(1+self.x_padding)))/self.cell_size))
        else:
            self.x_window_size = (int(np.floor(w * (1 + self.x_padding))) // self.cell_size,
                                  int(np.floor(h * (1 + self.x_padding))) // self.cell_size)

        self.x_cos_window = cos_window(self.x_window_size)

        if self.features == 'sfres50':
            self.z_window_size = (np.ceil(int(np.floor(w * (1 + self.z_padding))) / self.cell_size),
                                  np.ceil(int(np.floor(h * (1 + self.z_padding))) / self.cell_size))
        else:

            self.z_window_size=(int(np.floor(w*(1+self.z_padding)))//self.cell_size,int(np.floor(h*(1+self.z_padding)))//self.cell_size)

        self.z_cos_window = cos_window(self.z_window_size)

        s=np.sqrt(w*h)*self.output_sigma_factor/self.cell_size

        self.x_gaus = gaussian2d_labels(self.x_window_size, s)
        self.z_gaus = gaussian2d_labels(self.z_window_size, s)

        if self.features=='gray' or self.features=='color':
            first_frame = first_frame.astype(np.float32) / 255
            x=self._crop(first_frame,self._center,(w,h),self.x_padding)
            x=x-np.mean(x)
        elif self.features=='hog':
            x=self._crop(first_frame,self._center,(w,h),self.x_padding)
            x=cv2.resize(x,(self.x_window_size[0]*self.cell_size,self.x_window_size[1]*self.cell_size))
            x=extract_hog_feature(x, cell_size=self.cell_size)

        elif self.features=='cn':
            x = cv2.resize(first_frame, (self.x_window_size[0] * self.cell_size, self.x_window_size[1] * self.cell_size))
            x=extract_cn_feature(x,self.cell_size)

        elif self.features=='sfres50':

            x=self._crop(first_frame,self._center,(w,h),self.x_padding)

            desired_sz = (int((self.x_window_size[0]+1) * self.cell_size), \
                          int((self.x_window_size[1]+1) * self.cell_size))

            x = cv2.resize(x, desired_sz)
            x=extract_sfres50_feature(self.model,x,self.cell_size)

        else:
            raise NotImplementedError

        self.init_response_center = (0,0)

        x = self._get_windowed(x, self.x_cos_window)

        self.x1 = torch.from_numpy(x.astype(np.float32)).cuda() #
        if self.vis is not None:
            self.vis.image(self.x1.permute(2,0,1)[0:3,:,:],win='template')


    def update(self,current_frame):

        assert len(current_frame.shape) == 3 and current_frame.shape[2] == 3
        imgh,imgw,imgc = current_frame.shape
        if self.features == 'gray':
            current_frame = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
        if self.features=='color' or self.features=='gray':
            current_frame = current_frame.astype(np.float32) / 255
            z = self._crop(current_frame, self._center, (self.w, self.h),self.z_padding)
            z=z-np.mean(z)
        elif self.features=='hog':
            z = self._crop(current_frame, self._center, (self.w, self.h),self.z_padding)
            z = cv2.resize(z, (self.z_window_size[0] * self.cell_size, self.z_window_size[1] * self.cell_size))
            z = extract_hog_feature(z, cell_size=self.cell_size)
        elif self.features=='cn':
            z = self._crop(current_frame, self._center, (self.w, self.h),self.z_padding)
            z = cv2.resize(z, (self.z_window_size[0] * self.cell_size, self.z_window_size[1] * self.cell_size))
            z = extract_cn_feature(z, cell_size=self.cell_size)

        elif self.features=='sfres50':

            z=self._crop(current_frame,self._center,(self.w,self.h),self.z_padding)

            desired_sz = (int((self.z_window_size[0]+1) * self.cell_size), \
                          int((self.z_window_size[1]+1) * self.cell_size))

            z = cv2.resize(z, desired_sz)
            z=extract_sfres50_feature(self.model,z,self.cell_size)

        else:
            raise NotImplementedError

        z = torch.from_numpy(z.astype(np.float32)).cuda()
        # original operations
        # zf = fft2(self._get_windowed(z, self._window))

        # Conduct U transformation over search region:
        if self.U is not None:
            if z.size() != self.U.size()[:-1]:
                raise NotImplementedError
            z_ = Trans(z, self.U, self.lr_u)
        else:
            z_ = z

        # Conduct V transformation over target template:
        if self.V is not None:
            x_ = Trans(self.x1,self.V,self.lr_v)
        else:
            self.V = CalTrans(self.x1, self.x1, self.lambda_v)
            x_ = self.x1

        if self.vis is not None:
            self.vis.image(x_.permute(2,0,1)[0:3,:,:], win='updated_template')
            self.vis.image(z.permute(2,0,1)[0:3,:,:], win='search region')
            self.vis.image(z_.permute(2,0,1)[0:3,:,:], win='updated search region')

        # Get response maps:
        # responses=[]
        # for ci in range(3):
        #     response = signal.convolve2d(z_[:,:,ci],x_[:,:,ci],"same","symm")
        #     responses.append(response)
        padding = [int(np.ceil((self.x_window_size[1]-1) / 2)), int(np.ceil((self.x_window_size[0]-1) / 2))]
        responses = F.conv2d(z_.permute(2,0,1).unsqueeze(0),x_.permute(2,0,1).unsqueeze(0),padding=padding).squeeze(0).squeeze(0)
        responses = ((responses-responses.min())/(responses.max()-responses.min()+1e-20)).detach().cpu().numpy()
        curr =np.unravel_index(np.argmax(responses, axis=None),responses.shape)

        if self.vis is not None:
            self.vis.image(responses,win="response")

        dy=curr[0]-self.z_window_size[1]/2
        dx=curr[1]-self.z_window_size[0]/2
        dy,dx=dy*self.cell_size,dx*self.cell_size
        x_c, y_c = self._center
        x_c+= dx
        y_c+= dy
        self._center = (np.floor(x_c), np.floor(y_c))

        if self.features=='color' or self.features=='gray':
            new_x = self._crop(current_frame, self._center, (self.w, self.h),self.x_padding)
        elif self.features=='hog':
            new_x = self._crop(current_frame, self._center, (self.w, self.h),self.x_padding)
            new_x = cv2.resize(new_x, (self.x_window_size[0] * self.cell_size, self.x_window_size[1] * self.cell_size))
            new_x= extract_hog_feature(new_x, cell_size=self.cell_size)
        elif self.features=='cn':
            new_x = self._crop(current_frame, self._center, (self.w, self.h),self.x_padding)
            new_x = cv2.resize(new_x, (self.x_window_size[0] * self.cell_size, self.x_window_size[1] * self.cell_size))
            new_x = extract_cn_feature(new_x,cell_size=self.cell_size)

        elif self.features=='sfres50':

            new_x=self._crop(current_frame,self._center,(self.w,self.h),self.x_padding)

            desired_sz = (int((self.x_window_size[0]+1) * self.cell_size), \
                          int((self.x_window_size[1]+1) * self.cell_size))

            new_x = cv2.resize(new_x, desired_sz)
            new_x=extract_sfres50_feature(self.model,new_x,self.cell_size)

        else:
            raise NotImplementedError

        max_score = responses.max()

        # update U and V transformations
        new_x = self._get_windowed(new_x, self.x_cos_window)
        new_x_ = torch.from_numpy(new_x.astype(np.float32)).cuda()
        self.V = CalTrans(self.x1,new_x_, self.lambda_v)

        # extract new search region
        if self.features == 'gray':
            current_frame = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
        if self.features=='color' or self.features=='gray':
            current_frame = current_frame.astype(np.float32) / 255
            new_z = self._crop(current_frame, self._center, (self.w, self.h),self.z_padding)
            new_z=new_z-np.mean(new_z)
        elif self.features=='hog':
            new_z = self._crop(current_frame, self._center, (self.w, self.h),self.z_padding)
            new_z = cv2.resize(new_z, (self.z_window_size[0] * self.cell_size, self.z_window_size[1] * self.cell_size))
            new_z = extract_hog_feature(new_z, cell_size=self.cell_size)
        elif self.features=='cn':
            new_z = self._crop(current_frame, self._center, (self.w, self.h),self.z_padding)
            new_z = cv2.resize(new_z, (self.z_window_size[0] * self.cell_size, self.z_window_size[1] * self.cell_size))
            new_z = extract_cn_feature(new_z, cell_size=self.cell_size)

        elif self.features=='sfres50':

            new_z=self._crop(current_frame,self._center,(self.w,self.h),self.z_padding)

            desired_sz = (int((self.z_window_size[0]+1) * self.cell_size), \
                          int((self.z_window_size[1]+1) * self.cell_size))

            new_z = cv2.resize(new_z, desired_sz)
            new_z=extract_sfres50_feature(self.model,new_z,self.cell_size)

        else:
            raise NotImplementedError

        new_z_ = np.multiply(np.repeat(self.z_cos_window[:,:,np.newaxis],new_z.shape[2],axis=2),new_z)
        new_z_ = torch.from_numpy(new_z_.astype(np.float32)).cuda()
        new_z = torch.from_numpy(new_z.astype(np.float32)).cuda()
        self.U = CalTrans(new_z,new_z_, self.lambda_u)

        if (self._center[0] - self.w / 2)>imgw or (self._center[0] + self.w/2 )<0:
            self._center = (np.floor(imgw / 2 - self.w / 2), self._center[1])

        if (self._center[1] - self.h / 2)>imgh or (self._center[1] + self.h/2)<0:
            self._center = (self._center[0],np.floor(imgh / 2 - self.h/ 2))

        return [(self._center[0] - self.w / 2), (self._center[1] - self.h / 2), self.w, self.h],max_score

    def _kernel_correlation(self, xf, yf, kernel='gaussian'):
        if kernel== 'gaussian':
            N=xf.shape[0]*xf.shape[1]
            xx=(np.dot(xf.flatten().conj().T,xf.flatten())/N)
            yy=(np.dot(yf.flatten().conj().T,yf.flatten())/N)
            xyf=xf*np.conj(yf)
            xy=np.sum(np.real(ifft2(xyf)),axis=2)
            kf = fft2(np.exp(-1 / self.sigma ** 2 * np.clip(xx+yy-2*xy,a_min=0,a_max=None) / np.size(xf)))
        elif kernel== 'linear':
            kf= np.sum(xf*np.conj(yf),axis=2)/np.size(xf)
        else:
            raise NotImplementedError
        return kf

    def _training(self, xf, yf, kernel='gaussian'):
        kf = self._kernel_correlation(xf, xf, kernel)
        alphaf = yf/(kf+self.lambda_)
        return alphaf

    def _detection(self, alphaf, xf, zf, kernel='gaussian'):
        kzf = self._kernel_correlation(zf, xf, kernel)
        responses = np.real(ifft2(alphaf * kzf))
        return responses

    def _crop(self,img,center,target_sz,padding):
        if len(img.shape)==2:
            img=img[:,:,np.newaxis]
        w,h=target_sz

        # the same as matlab code
        w=int(np.floor((1+padding)*w))
        h=int(np.floor((1+padding)*h))
        cropped = np.zeros([h, w, 3])
        xs=(np.floor(center[0])+np.arange(w)-np.floor(w/2)).astype(np.int64)
        ys=(np.floor(center[1])+np.arange(h)-np.floor(h/2)).astype(np.int64)
        x0=np.arange(w).astype(np.int64)
        y0=np.arange(h).astype(np.int64)
        xs[xs<0]=0
        ys[ys<0]=0
        x0[xs<0]=0
        y0[ys<0]=0

        xs[xs>=img.shape[1]]=img.shape[1]-1
        ys[ys>=img.shape[0]]=img.shape[0]-1
        x0[xs>=img.shape[1]]=h-1
        y0[ys>=img.shape[0]]=w-1

        x0,y0=np.meshgrid(x0, y0)
        xs, ys = np.meshgrid(xs, ys)
        cropped[y0,x0]=img[ys,xs]
        # cropped=cv2.getRectSubPix(img,(int(np.floor((1+padding)*w)),int(np.floor((1+padding)*h))),center)

        return cropped

    def _get_windowed(self,img,cos_window):
        if len(img.shape)==2:
            img=img[:,:,np.newaxis]
        windowed = cos_window[:,:,None] * img
        return windowed
