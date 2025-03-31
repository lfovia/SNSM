
from os.path import join, exists, isfile
from os import makedirs
import torch
import cv2
from PIL import Image
from torchvision import transforms
import argparse
import os
from os.path import join, exists, isfile
from tqdm.auto import tqdm
import configparser
from os.path import join, exists, isfile
from os import makedirs
from torch.utils.data import DataLoader
import numpy as np
import faiss
import lpips
import h5py
import torchvision
from math import ceil
from torch.utils.data import DataLoader, SubsetRandomSampler
from einops.layers.torch import Rearrange, Reduce
import sys
sys.path.append('./..')
from datasets_original import PlaceDataset
from sklearn.decomposition import PCA
import utils
import vision_transformer as vits

class ResNet50(torch.nn.Module):
    def __init__(self,
                 model_name='resnet50',
                 layers_to_freeze=3,
                 layers_to_crop = [0],
                 pretrained=True,
                 ):
       
        super().__init__()
        self.model_name = model_name.lower()
        self.layers_to_freeze = layers_to_freeze
        self.layers_to_crop = layers_to_crop
        
         
        if pretrained:
            # the new naming of pretrained weights, you can change to V2 if desired.
            weights = 'IMAGENET1K_V1'
        else:
            weights = None
        self.model = torchvision.models.resnet50(weights=weights)
        # freeze only if the model is pretrained
        if pretrained:
            if layers_to_freeze >= 0:
                self.model.conv1.requires_grad_(False)
                self.model.bn1.requires_grad_(False)
            if layers_to_freeze >= 1:
                self.model.layer1.requires_grad_(False)
            if layers_to_freeze >= 2:
                self.model.layer2.requires_grad_(False)
            if layers_to_freeze >= 3:
                self.model.layer3.requires_grad_(False)

        # remove the avgpool and most importantly the fc layer
        self.model.avgpool = None
        self.model.fc = None

        if 4 in layers_to_crop:
            self.model.layer4 = None
        if 3 in layers_to_crop:
            self.model.layer3 = None
        #out_channels = 2048
        #if '34' in model_name or '18' in model_name:
        #    out_channels = 512
       # self.out_channels = out_channels // 2 if self.model.layer4 is None else out_channels
       # self.out_channels = self.out_channels // 2 if self.model.layer3 is None else self.out_channels
        
    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        y1 = self.model.layer1(x)
        y2 = self.model.layer2(y1)
        y3 = self.model.layer3(y2)
        #y4 = self.model.layer4(y3)
        return y3
        
def Model(args):
    model = vits.__dict__[args.arch](patch_size=args.patch_size, num_classes=0)
    for name, p in model.named_parameters():
        p.requires_grad = False
    model.eval()
    model.to(args.device)
    url = None
    if args.arch == "vit_small" and args.patch_size == 16:
        url = "dino_deitsmall16_pretrain/dino_deitsmall16_pretrain.pth"
    elif args.arch == "vit_small" and args.patch_size == 8:
        url = "dino_deitsmall8_300ep_pretrain/dino_deitsmall8_300ep_pretrain.pth"  # model used for visualizations in our paper
    elif args.arch == "vit_base" and args.patch_size == 16:
        url = "dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth"
    elif args.arch == "vit_base" and args.patch_size == 8:
        url = "dino_vitbase8_pretrain/dino_vitbase8_pretrain.pth"
    if url is not None:
        print("Since no pretrained weights have been provided, we load the reference pretrained DINO weights.")
        state_dict = torch.hub.load_state_dict_from_url(url="https://dl.fbaipublicfiles.com/dino/" + url)
        model.load_state_dict(state_dict, strict=True)
    else:
        print("There is no reference weights available for this model => We use random weights.")
    
    return model


def dino_feature_extract(eval_set, model, device, opt, config, file_name, pool_size):
    if not exists(opt.result_save_folder):
        makedirs(opt.result_save_folder)
    #output_global_features_filename = join(opt.output_features_dir, file_name)#'qry_globalfeats_pool.npy')
    pool_size_loc = 151296; pool_size_glob = 768

    test_data_loader = DataLoader(dataset=eval_set, num_workers=int(config['global_params']['threads']),
                                  batch_size=int(config['feature_extract']['cacheBatchSize']),
                                  shuffle=False, pin_memory=(not opt.nocuda))

    with torch.no_grad():
        tqdm.write('====> Extracting Features')
        
        db_feat_loc = np.empty((len(eval_set), pool_size_loc), dtype=np.float32)               
        
        for iteration, (input_data, indices) in \
                enumerate(tqdm(test_data_loader, position=1, leave=False, desc='Test Iter'.rjust(15)), 1):
            indices_np = indices.detach().numpy()
            input_data = input_data.to(device)
            image_encoding = model.get_intermediate_layers(input_data)
            image_encoding = lpips.normalize_tensor(image_encoding[0]).flatten(0)
            db_feat_loc[indices_np, :] = image_encoding.detach().cpu().numpy()
    return db_feat_loc  


def get_clusters(cluster_set, model, device, opt, config, file_name):
    nDescriptors = 50000
    nPerImage = 50
    nIm = ceil(nDescriptors/nPerImage)
    encoder_dim = 768
    opt.num_clusters = 5
    
    sampler = SubsetRandomSampler(np.random.choice(len(cluster_set), nIm, replace=False))
    data_loader = DataLoader(dataset=cluster_set, 
                num_workers=int(config['global_params']['threads']), batch_size=int(config['feature_extract']['cacheBatchSize']), shuffle=False, 
                pin_memory=opt.cuda,
                sampler=sampler)
    
    opt.catcheBatchSize = int(config['feature_extract']['cacheBatchSize'])
    
    if not exists(join(opt.result_save_folder, 'centroids')):
        makedirs(join(opt.result_save_folder, 'centroids'))

    initcache = join(opt.result_save_folder, 'centroids', opt.arch + '_' +file_name+ '_' + str(opt.num_clusters) + '_desc_cen.hdf5')
    with h5py.File(initcache, mode='w') as h5: 
        with torch.no_grad():
            model.eval()
            print('====> Extracting Descriptors')
            dbFeat = h5.create_dataset("descriptors", 
                        [nDescriptors, encoder_dim], 
                        dtype=np.float32)
            for iteration, (input_data, indices) in enumerate(tqdm(data_loader, position=1, leave=False, desc='Test Iter'.rjust(15)), 1):
                indices_np = indices.detach().numpy()
                input_data = input_data.to(device)
                #image_descriptors = model.forward(input_data).view(input_data.size(0), encoder_dim, -1).permute(0, 2, 1)
                image_descriptors = model.get_intermediate_layers(input_data, 9)[0]#.view(input_data.size(0), encoder_dim, -1).permute(0, 2, 1)  
                #print('image_descriptors:\t', image_descriptors[0].shape); print(hi)
                image_descriptors = lpips.normalize_tensor(image_descriptors)#[0]
                batchix = (iteration-1)*opt.catcheBatchSize*nPerImage
                for ix in range(image_descriptors.size(0)):
                    # sample different location for each image in batch
                    sample = np.random.choice(image_descriptors.size(1), nPerImage, replace=False)
                    startix = batchix + ix*nPerImage
                    dbFeat[startix:startix+nPerImage, :] = image_descriptors[ix, sample, :].detach().cpu().numpy()
                del input_data, image_descriptors 
        print('====> Clustering..')
        niter = 100
        kmeans = faiss.Kmeans(encoder_dim, opt.num_clusters, niter=niter, verbose=False)
        kmeans.train(dbFeat[...])

        print('====> Storing centroids', kmeans.centroids.shape)
        h5.create_dataset('centroids', data=kmeans.centroids)
        print('====> Done!')



def main():
    parser = argparse.ArgumentParser(description='Patch-NetVLAD-Feature-Match')
    parser.add_argument('--config_path', type=str, default='performance_bb.ini',
                        help='File name (with extension) to an ini file that stores most of the configuration data for patch-netvlad')
    parser.add_argument('--dataset_root_dir', type=str, default='/home/anuradha/MyResearch@Sindhu/DataSets/Multi-RGBT/All_Datasets/Datasets/Full-Length/kaist-cvpr15/images/',
                        help='If the files in query_file_path and index_file_path are relative, use dataset_root_dir as prefix.')
    parser.add_argument('--query_file_path', type=str, default='/home/anuradha/MyResearch@Sindhu/DataSets/Multi-RGBT/All_Datasets/Datasets/Full-Length/kaist-cvpr15/images/set01_v002_thr.txt',
                        help='Path (with extension) to a text file that stores the save location and name of all query images in the dataset')
    parser.add_argument('--index_file_path', type=str, default='/home/anuradha/MyResearch@Sindhu/DataSets/Multi-RGBT/All_Datasets/Datasets/Full-Length/kaist-cvpr15/images/set01_v002_rgb.txt',
                        help='Path (with extension) to a text file that stores the save location and name of all database images in the dataset')
    parser.add_argument('--ground_truth_path', type=str, default=None,
                        help='Path (with extension) to a file that stores the ground-truth data')      
    parser.add_argument('--arch', default='vit_base', type=str,
        choices=['vit_tiny', 'vit_small', 'vit_base'], help='Architecture (support only ViT atm).')   
    parser.add_argument('--patch_size', default=8, type=int, help='Patch resolution of the model.')                                   
    parser.add_argument('--result_save_folder', type=str, default='./KAIST/VLAD/')
    parser.add_argument('--posDistThr', type=int, default=None, help='Manually set ground truth threshold')
    parser.add_argument('--nocuda', action='store_true', help='If true, use CPU only. Else use GPU.')
    
    opt = parser.parse_args()
    print(opt)
    
    configfile = opt.config_path
    assert os.path.isfile(configfile)
    config = configparser.ConfigParser()
    config.read(configfile)
    
    cuda = not opt.nocuda
    if cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, please run with --nocuda")    
    opt.device = torch.device("cuda" if cuda else "cpu")
    opt.cuda = cuda
    model=Model(opt)
    #model = ResNet50()
    model = model.to(opt.device)
    model.eval()   
    
    #opt.query_file_path = opt.dataset_root_dir+'kaist_set10_v001_lwir.txt'
    #opt.index_file_path = opt.dataset_root_dir+'kaist_set10_v001_vis.txt'
    
    
    dataset = PlaceDataset(opt.query_file_path, opt.index_file_path, opt.dataset_root_dir, opt.ground_truth_path,
                           config['feature_extract'])
   
    #qImgs = PlaceDataset(None, opt.query_file_path, opt.dataset_root_dir, None, config['feature_extract'])
    dbImgs = PlaceDataset(None, opt.index_file_path, opt.dataset_root_dir, None, config['feature_extract'])
    
    #for i, k in dbImgs:
    #    print(i.shape)
    
    #get_clusters(qImgs, model, opt.device, opt, config, 'Nordland-winter')
    get_clusters(dbImgs, model, opt.device, opt, config, 'p30k-mscn')
    
    #qfeats_loc = dino_feature_extract(qImgs, model, opt.device, opt, config, 'globalfeats_pool.npy', None) #opt.pool_size)#4096)#
    #dbfeats_loc = dino_feature_extract(dbImgs, model, opt.device, opt, config, 'globalfeats_pool.npy', None) #opt.pool_size)#4096)#200704)
    
    #qfeats_loc = np.load(opt.result_save_folder+'nir_left_patchfeats.npy')
    #dbfeats_loc = np.load(opt.result_save_folder+'rgb_left_patchfeats.npy')
    
    #feature_match(qfeats_loc, dbfeats_loc, dataset, opt.device, opt, config, 'bl_dino_recalls.txt')
    
    
    '''if not exists(opt.result_save_folder):
        makedirs(opt.result_save_folder)
    np.save(opt.result_save_folder+'nir_left_patchfeats.npy', qfeats_loc)
    np.save(opt.result_save_folder+'rgb_left_patchfeats.npy', dbfeats_loc)
    '''
   
    
if __name__ == "__main__":
    main()
        
