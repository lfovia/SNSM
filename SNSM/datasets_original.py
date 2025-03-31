
import os

import torchvision.transforms as transforms
import torch.utils.data as data

import numpy as np
from PIL import Image
from sklearn.neighbors import NearestNeighbors
import lpips
import torch
import scipy.signal
#from patchnetvlad.tools import PATCHNETVLAD_ROOT_DIR
import cv2

def input_transform(resize=(224, 224)):
    if resize[0] > 0 and resize[1] > 0:
        return transforms.Compose([
            transforms.Resize(resize),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
    else:
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])


class ScalingLayer(torch.nn.Module):
    def __init__(self):
        super(ScalingLayer, self).__init__()
        self.register_buffer('shift', torch.Tensor([-.030,-.088,-.188])[None,:,None,None])
        self.register_buffer('scale', torch.Tensor([.458,.448,.450])[None,:,None,None])

    def forward(self, inp):
        return (inp - self.shift) / self.scale
        
class PlaceDataset(data.Dataset):
    #def __init__(self, query_file_path, index_file_path, dataset_root_dir, mask_root_dir, ground_truth_path, config, posDistThr=None):
    def __init__(self, query_file_path, index_file_path, dataset_root_dir, ground_truth_path, config):
        super().__init__()

        self.queries, self.database, self.numQ, self.numDb, self.utmQ, self.utmDb = None, None, None, None, None, None
        if query_file_path is not None:
            self.queries, self.numQ = self.parse_text_file(query_file_path)
        if index_file_path is not None:
            self.database, self.numDb = self.parse_text_file(index_file_path)
        if ground_truth_path is not None:
            print('ground_truth_path:', ground_truth_path, ground_truth_path==None)
            print('P:', P, P==None); print(jo)
            self.utmQ, self.utmDb, self.posDistThr = self.parse_gt_file(ground_truth_path)

        if self.queries is not None:
            self.image_paths = self.database + self.queries
        else:
            self.image_paths = self.database
        
        
        self.images = [os.path.join(dataset_root_dir, image) for image in self.image_paths]
        
        self.positives = None
        self.distances = None
        self.mytransform = input_transform()
        self.scaling_layer = ScalingLayer()
        
    def __getitem__(self, index):
        img = Image.open(self.images[index]).convert('RGB')
        img = self.mytransform(img)
        '''img = cv2.imread(self.images[index])
        img = cv2.resize(img, (224, 224), interpolation = cv2.INTER_LINEAR)
        if len(img.shape)==2:
            img = np.repeat(img[..., np.newaxis], 3, axis=-1)
        img = self.RMSCN(np.float32(img))
        img = self.multi_chn(img)
        img = np.uint8(img) 
        
        img = Image.fromarray(img)
        img = self.mytransform(img)'''
        
        return img, index
        

    def __len__(self):
        return len(self.images)

    def get_positives(self):
        # positives for evaluation are those within trivial threshold range
        '''if  self.positives is None:
            knn = NearestNeighbors(n_jobs=-1)
            knn.fit(self.utmDb)
            self.distances, self.positives = knn.radius_neighbors(self.utmQ,
                    radius=self.posDistThr)'''
                    
        self.positives = [i for i in range(0,self.numDb)] #@@ OSU Strict
        #self.positives = [np.arange(max(i-25,0),i+25,1) for i in range(0,self.numDb)]
        return self.positives

    @staticmethod
    def parse_text_file(textfile):
        print('Parsing dataset...')

        with open(textfile, 'r') as f:
            image_list = f.read().splitlines()

        if 'robotcar' in image_list[0].lower():
            image_list = [os.path.splitext('/'.join(q_im.split('/')[-3:]))[0] for q_im in image_list]

        num_images = len(image_list)

        print('Done! Found %d images' % num_images)

        return image_list, num_images

    @staticmethod
    def parse_gt_file(gtfile):
        print('Parsing ground truth data file...')
        gtdata = np.load(gtfile)
        return gtdata['utmQ'], gtdata['utmDb'], gtdata['posDistThr']
        
    @staticmethod
    def RMSCN(image1):
          kernal_size = 5
          ker=np.ones([kernal_size,kernal_size])
          mscn_3ch = np.empty((image1.shape), dtype=np.float32)
          chns = [image1[:,:,0], image1[:,:,1], image1[:,:,2]]
          imgs = []
          for i, ch in enumerate(chns): 
             image2 = ch
             sqr_img=image2**2       
             mean_img=scipy.signal.convolve2d(image2,ker,mode="same")
             meansqr_image = scipy.signal.convolve2d(sqr_img,ker,mode="same")
             mean_image1 = mean_img/(kernal_size**2)
             meansqr_image1 = meansqr_image/(kernal_size**2)
             std_image = np.sqrt(np.abs(meansqr_image1 - mean_image1**2))
             imgg = (image2 - mean_image1) / (std_image + 1)
             imgs.append(((imgg - np.min(imgg))/(np.max(imgg) - np.min(imgg)))*255)
             del image2
          return imgs  
        
    @staticmethod    
    def multi_chn(proc_img1):
          proc_img = np.concatenate([np.expand_dims(proc_img1[0], axis=2), np.expand_dims(proc_img1[1], axis=2), np.expand_dims(proc_img1[2], axis=2)], axis=2)
          return proc_img        
        
