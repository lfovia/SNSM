
# Imports
import pytorch_lightning as pl
from einops.layers.torch import Rearrange, Reduce
from typing import Union, List, Tuple, Literal
from torch.nn import functional as F
from datasets_original import PlaceDataset
from sklearn.decomposition import PCA
#from vlad_utilities_pt import VLAD
from vlad_utilities_hdf5 import VLAD
from torch.utils.data import DataLoader
from os.path import join, exists, isfile
from os import makedirs
from tqdm.auto import tqdm
#import vision_transformer as vits
import configparser
import argparse
import torchvision
import numpy as np
import torch
import cv2
import os
import faiss
import lpips
#import utils
import vision_transformer as vits
from scipy.spatial.distance import cdist

# ResNet50 Backbone
class ResNet50(torch.nn.Module):
    def __init__(self,
                 pretrained=True,
                 ):
       
        super().__init__()
        #self.model_name = model_name.lower()
        
        if pretrained:
            # the new naming of pretrained weights, you can change to V2 if desired.
            weights = 'IMAGENET1K_V1'
        else:
            weights = None
        self.model = torchvision.models.resnet50(weights=weights)
        self.model.avgpool = None
        self.model.fc = None

    def get_intermediate_layers(self, x, n):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        y1 = self.model.layer1(x)
        y2 = self.model.layer2(y1)
        y3 = self.model.layer3(y2)
        y4 = self.model.layer4(y3)
        if n==1: y=y1;
        elif n==2: y=y2;     
        elif n==3: y=y3;
        elif n==4: y=y4;
        #y = y.permute(0,3,1,2)#.flatten(-2)
        y = y.flatten(-2).permute(0,2,1)#
        return [y]

# DINOv2 Model from SALAD: Optimal Transport Aggregation for Visual Place Recognition
DINOV2_ARCHS = {
    'dinov2_vits14': 384,
    'dinov2_vitb14': 768,
    'dinov2_vitl14': 1024,
    'dinov2_vitg14': 1536,
}

_DINO_V2_MODELS = Literal["dinov2_vits14", "dinov2_vitb14", \
                        "dinov2_vitl14", "dinov2_vitg14"]
_DINO_FACETS = Literal["query", "key", "value", "token"]
class DinoV2ExtractFeatures:
    """
        Extract features from an intermediate layer in Dino-v2
    """
    def __init__(self, dino_model: _DINO_V2_MODELS, layer: int, 
                facet: _DINO_FACETS="token", use_cls=False, 
                norm_descs=True, device: str = "cpu") -> None:

        self.vit_type: str = dino_model
        self.dino_model: nn.Module = torch.hub.load(
                'facebookresearch/dinov2', dino_model)
        self.device = torch.device(device)
        self.dino_model = self.dino_model.eval().to(self.device)
        self.layer: int = layer
        self.facet = facet
        if self.facet == "token":
            self.fh_handle = self.dino_model.blocks[self.layer].\
                    register_forward_hook(
                            self._generate_forward_hook())
        else:
            self.fh_handle = self.dino_model.blocks[self.layer].\
                    attn.qkv.register_forward_hook(
                            self._generate_forward_hook())
        self.use_cls = use_cls
        self.norm_descs = norm_descs
        # Hook data
        self._hook_out = None
    
    def _generate_forward_hook(self):
        def _forward_hook(module, inputs, output):
            self._hook_out = output
        return _forward_hook
    
    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        """
            Parameters:
            - img:   The input image
        """
        with torch.no_grad():
            res = self.dino_model(img)
            if self.use_cls:
                res = self._hook_out
            else:
                res = self._hook_out[:, 1:, ...]
            if self.facet in ["query", "key", "value"]:
                d_len = res.shape[2] // 3
                if self.facet == "query":
                    res = res[:, :, :d_len]
                elif self.facet == "key":
                    res = res[:, :, d_len:2*d_len]
                else:
                    res = res[:, :, 2*d_len:]
        if self.norm_descs:
            res = F.normalize(res, dim=-1)
        self._hook_out = None   # Reset the hook
        return res
    
    def __del__(self):
        self.fh_handle.remove()

class DINOv2(torch.nn.Module):
    def __init__(
            self,
            model_name='dinov2_vitg14',
            num_trainable_blocks=0,
            norm_layer=False,
            return_token=False
        ):
        super().__init__()

        assert model_name in DINOV2_ARCHS.keys(), f'Unknown model name {model_name}'
        self.model = torch.hub.load('facebookresearch/dinov2', model_name)
        self.num_channels = DINOV2_ARCHS[model_name]
        self.num_trainable_blocks = num_trainable_blocks
        self.norm_layer = norm_layer
        self.return_token = return_token

    def get_intermediate_layers(self, x, n):
        B, C, H, W = x.shape
        x = self.model.prepare_tokens_with_masks(x)
        output = []
        with torch.no_grad():
            for i, blk in enumerate(self.model.blocks):
               x = blk(x)
               if len(self.model.blocks) - i <= n:
                  output.append(x[:, 1:])        
                
        #if self.norm_layer:
        #    x = self.model.norm(x)
        
        #t = x[:, 0]
        #f = x[:, 1:]
        
        # Reshape to (B, C, H, W)
        #f = f.reshape((B, H // 14, W // 14, self.num_channels)).permute(0, 3, 1, 2)

        #if self.return_token:
        #    return f, t
        return output

# Model selection
def Model(args):
    if args.backbone=='DINOv1':
       model = vits.__dict__[args.arch](patch_size=args.patch_size, num_classes=0)
       for name, p in model.named_parameters():
           p.requires_grad = False
       model.eval()
       model.to(args.device)
       url = None 
       if args.arch == "vit_small" and args.patch_size == 16:
           url = "dino_deitsmall16_pretrain/dino_deitsmall16_pretrain.pth"
           args.split_factor = 14
       elif args.arch == "vit_small" and args.patch_size == 8:
           url = "dino_deitsmall8_300ep_pretrain/dino_deitsmall8_300ep_pretrain.pth"  # model used for visualizations in our paper
       elif args.arch == "vit_base" and args.patch_size == 16:
           url = "dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth"
           args.split_factor = 14
       elif args.arch == "vit_base" and args.patch_size == 8:
           url = "dino_vitbase8_pretrain/dino_vitbase8_pretrain.pth"
           args.split_factor = 28
       if url is not None:
           print("Since no pretrained weights have been provided, we load the reference pretrained DINO weights.")
           state_dict = torch.hub.load_state_dict_from_url(url="https://dl.fbaipublicfiles.com/dino/" + url)
           model.load_state_dict(state_dict, strict=True)
       else:
           print("There is no reference weights available for this model => We use random weights.")
           
    elif args.backbone=='DINOv2':
       model = DINOv2()
       model = model.to(args.device)
       model.eval()
       args.split_factor = 16
    elif args.backbone=='ResNet50':
       model = ResNet50()
       model = model.to(args.device)
       model.eval()   
       args.split_factor = 14
    elif args.backbone == 'DINOv2g':
       model = DinoV2ExtractFeatures(dino_model = 'dinov2_vitg14', layer=31, facet="value", use_cls=False, norm_descs=True, device=args.device)
       args.split_factor = 16
    return model

# GAP and GMP Descriptors
def get_gp_descriptors(db_feat_loc, opt):
    Re_arrange = Rearrange('b (h w c) -> b c (h w)', w=opt.split_factor, h=opt.split_factor) 
    db_feat_loc = Re_arrange(db_feat_loc)
    #print('db_feat_loc:', db_feat_loc.shape)
    if opt.gap:
        ret = torch.mean(db_feat_loc, dim=2)
        print('GAP Descs:', ret.shape)
    elif opt.gmp:
        ret = torch.max(db_feat_loc, dim=2)[0]
        print('GMP Descs:', ret.shape)
    if opt.backbone=='DINOv1' or opt.backbone=='DINOv2': assert ret.shape[-1]==768
    elif opt.backbone=='ResNet50': assert ret.shape[-1]==1024    
    return ret
       
# GeM descriptors         
def get_gem_descriptors(patch_descs, opt):
    Re_arrange = Rearrange('b (h w c) -> b c (h w)', w=opt.split_factor, h=opt.split_factor) 
    patch_descs = Re_arrange(patch_descs)
    assert len(patch_descs.shape) == len(("N", "n_p", "d_dim"))
    g_res = None
    if opt.gem_use_abs:
       g_res = torch.mean(torch.abs(patch_descs)**opt.gem_p, dim=-1) ** (1/opt.gem_p)
    else:
       if opt.gem_elem_by_elem:
          g_res_all = []
          for patch_desc in patch_descs:
              x = torch.mean(patch_desc**opt.gem_p, dim=-1)
              g_res = x.to(torch.complex64) ** (1/opt.gem_p)
              g_res = torch.abs(g_res) * torch.sign(x)
              g_res_all.append(g_res)
              g_res = torch.stack(g_res_all)
       else:
              x = torch.mean(patch_descs**opt.gem_p, dim=-1)
              g_res = x.to(torch.complex64) ** (1/opt.gem_p)
              g_res = torch.abs(g_res) * torch.sign(x)
    print('GeM Descs:', g_res.shape)
    if opt.backbone=='DINOv1' or opt.backbone=='DINOv2': assert g_res.shape[-1]==768
    elif opt.backbone=='ResNet50': assert g_res.shape[-1]==1024 
    return g_res

# VLAD Descriptors 
def VLAD_descs(db_feat_loc, opt, cache_dir):
    
    vlad = VLAD(num_clusters=opt.num_clusters, cache_dir=cache_dir) 
    Re_arrange2 = Rearrange('b (w c) -> b w c', w=opt.split_factor**2) 
    db_feat_loc = Re_arrange2(db_feat_loc)
    if cache_dir is None:
       vlad.fit(db_feat_loc.view(-1, db_feat_loc.shape[-1])) 
    else:
       vlad.fit(None)   
    vlad_descs = vlad.generate_multi(db_feat_loc)#.detach().cpu().numpy()) #@@ If cache file is a torch .pt file
    return vlad_descs

# Cosine similarity between the selected patch and neighboring patches
class Cosine_Similarity(torch.nn.Module):
    def __init__ (self,
                  similarity_measure = torch.nn.CosineSimilarity()):

        super().__init__()
        self.similarity_measure = similarity_measure
        
    def forward(self, input_image):
        self.input_image_orig = input_image
        H = self.input_image_orig.shape[0]
        W = self.input_image_orig.shape[1]
        self.input_image = torch.zeros([H + 1, W + 1, 1, self.input_image_orig.shape[-1]])
        self.input_image[:H, :W] = self.input_image_orig
        self.similarity_map = np.empty((H, W), dtype=np.float32)
        for m in range(H):
           for n in range(W):
               
               self.similarity_map[m,n] = np.sum(
                                     [ [ int((n+l)>=0 and (m+k)>=0) * int(m+k<H and n+l<W) * (
                                     self.similarity_measure(self.input_image[m,n], self.input_image[m+k, n+l])
                                     ).detach().cpu().numpy() 
                                     for k in [-1,1] ] 
                                     for l in [-1,1] ]  
                                      )
               
        return self.similarity_map

# Upscale feature maps
def UpScale(image_encoding, att_scale, opt):
    
    if att_scale>1: 
        image_encoding = image_encoding.permute(2, 0, 1)#@@DINO 
        image_encoding = (torch.nn.functional.interpolate(
                        image_encoding.unsqueeze(0),
                        scale_factor=att_scale,
                        mode="nearest",
                    )[0]#.cpu().numpy()
                )
        image_encoding = image_encoding.permute(1, 2, 0)#@@DINO  
    return image_encoding#.permute(1,2,0)#@@@ features: DINO                 

# SNSM Feature maps
def N_support(image_encoding, neigh_patch_size, CosineSim):
    Re_arrange_P1 = Rearrange('(h m) (w n) c -> h w (m n c)', m=neigh_patch_size, n=neigh_patch_size)    
    image_encodingP1 = Re_arrange_P1(image_encoding);
    similarity_mapP1 = CosineSim.forward(image_encodingP1.unsqueeze(2))#.flatten()
    return similarity_mapP1

# SNSM Feature maps
def SNSM(image_encoding, opt):
    CosineSim = Cosine_Similarity()
    Arrange = Rearrange('(h w) c-> h w c', h=opt.split_factor)  
    image_encoding = Arrange(image_encoding)
    image_encoding1 = UpScale(image_encoding, opt.upscale_factor, opt)
    SimMap_P2 = N_support(image_encoding1, opt.window_size, CosineSim)
    return torch.tensor(SimMap_P2).to(image_encoding.get_device())
              

# Feature extraction module                
def feature_extraction(eval_set, model, device, opt, config, file_name, pool_size, cache_dir):
    if not exists(opt.result_save_folder):
        makedirs(opt.result_save_folder)
    pool_size_loc = opt.emb_dim
    Re_arrange = Rearrange('h w -> (h w)')
    test_data_loader = DataLoader(dataset=eval_set, num_workers=int(config['global_params']['threads']),
                                  batch_size=int(config['feature_extract']['cacheBatchSize']),
                                  shuffle=False, pin_memory=(not opt.nocuda))
    with torch.no_grad():
        tqdm.write('====> Extracting Features')
        
        #if not opt.vlad:
        #db_feat_loc = np.empty((len(eval_set), pool_size_loc), dtype=np.float32)               
        #elif opt.vlad:
        db_feat_loc = torch.empty((len(eval_set), pool_size_loc), dtype=torch.float32).to(opt.device)
        
        for iteration, (input_data, indices) in \
                enumerate(tqdm(test_data_loader, position=1, leave=False, desc='Test Iter'.rjust(15)), 1):
            indices_np = indices.detach().numpy()
            input_data = input_data.to(device)
            if not opt.vlad_api:
               if opt.backbone=='DINOv2g':  
                  image_encoding = model(input_data)[0]
               else:
                  image_encoding = model.get_intermediate_layers(input_data, opt.encoder)[0]
                  image_encoding = lpips.normalize_tensor(image_encoding)[0]
               
            if opt.vlad_api:
               image_encoding = model(input_data)#.detach().cpu().numpy().flatten()
            if opt.snsm:
               image_encoding = SNSM(image_encoding, opt)
            #print('db_feat_loc:', image_encoding.shape, opt.split_factor); print(hi)   
            image_encoding = Re_arrange(image_encoding)
            db_feat_loc[indices_np, :] = image_encoding#.detach().cpu().numpy()
    
    if opt.vlad:
        print('===>',opt.backbone,'-VLAD Activated...') 
        db_feat_loc = VLAD_descs(db_feat_loc, opt, cache_dir)
    elif opt.gem: 
        print('===>',opt.backbone,'-GeM Activated...')
        db_feat_loc = get_gem_descriptors(db_feat_loc, opt)
    elif opt.gap: 
        print('===>',opt.backbone,'-GAP Activated...')
        print('db_feat_loc:', db_feat_loc.shape)
        db_feat_loc = get_gp_descriptors(db_feat_loc, opt)  
    elif opt.gmp: 
        print('===>',opt.backbone,'-GMP Activated...')
        db_feat_loc = get_gp_descriptors(db_feat_loc, opt)       
    db_feat_loc = db_feat_loc.detach().cpu().numpy()
    #np.save(file_name, db_feat_loc)
    return db_feat_loc   

# Compute recalls 
def compute_recall(gt, predictions, numQ, n_values, recall_str=''):
    correct_at_n = np.zeros(len(n_values))
    for qIx, pred in enumerate(predictions):
        for i, n in enumerate(n_values):
            # if in top N then also in top NN, where NN > N
            if np.any(np.in1d(pred[:n], gt[qIx])):
                correct_at_n[i:] += 1
                break
    recall_at_n = correct_at_n / numQ
    all_recalls = {}  # make dict for output
    for i, n in enumerate(n_values):
        all_recalls[n] = recall_at_n[i]
        tqdm.write("====> Recall {}@{}: {:.4f}".format(recall_str, n, recall_at_n[i]))
    return all_recalls

# Writing predictions to txt file
def write_kapture_output(opt, eval_set, predictions, distances, outfile_name):
    if not exists(opt.result_save_folder):
        os.mkdir(opt.result_save_folder)
    outfile = join(opt.result_save_folder, outfile_name)
    print('Writing results to', outfile)
    with open(outfile, 'w') as kap_out:
        kap_out.write('# kapture format: 1.0\n')
        kap_out.write('# query_image, map_image\n')
        image_list_array = np.array(eval_set.images)
        for q_idx in range(len(predictions)):
            full_paths = image_list_array[predictions[q_idx]]
            dist = distances[q_idx]
            query_full_path = image_list_array[eval_set.numDb + q_idx]
            for c, ref_image_name in enumerate(full_paths):
                kap_out.write(query_full_path + ', ' + ref_image_name + ', ' + str(dist[c]) + '\n')
            del dist    

# Writing recalls to txt file
def write_recalls_output(opt, recalls_netvlad, n_values, file_name):
    if not exists(opt.result_save_folder):
        os.mkdir(opt.result_save_folder)
    outfile = join(opt.result_save_folder, file_name)
    print('Writing recalls to', outfile)
    with open(outfile, 'w') as rec_out:
        for n in n_values:
            rec_out.write("Recall {}@{}: {:.4f}\n".format('DINOV2', n, recalls_netvlad[n]))

# Feature match module
def feature_match(qFeat, dbFeat, eval_set, device, opt, config, file_name):

    #qFeat = np.load(input_query_global_features_prefix)
    pool_size = qFeat.shape[1]
    #dbFeat = np.load(input_index_global_features_prefix)

    if dbFeat.dtype != np.float32:
        qFeat = qFeat.astype('float32')
        dbFeat = dbFeat.astype('float32')

    tqdm.write('====> Building faiss index')
    faiss_index = faiss.IndexFlatL2(pool_size)
    # noinspection PyArgumentList
    faiss_index.add(dbFeat)

    n_values = []
    for n_value in config['feature_match']['n_values_all'].split(","):  # remove all instances of n that are bigger than maxK
        n_values.append(int(n_value))

    distances, predictions = faiss_index.search(qFeat, min(len(qFeat), max(n_values)))
    print('Saving qry-pred-and-gt features to ', opt.result_save_folder)
    if not exists(opt.result_save_folder):
        makedirs(opt.result_save_folder)
    
    write_kapture_output(opt, eval_set, predictions, distances, 'predictions.txt')

    print('Finished matching features.')

    # for each query get those within threshold distance
    if opt.ground_truth_path is None:
        print('Calculating recalls using ground truth.')
        gt = eval_set.get_positives()
        print(gt[:20])
        global_recalls = compute_recall(gt, predictions, eval_set.numQ, n_values, 'DINOv1')
        write_recalls_output(opt, global_recalls, n_values, file_name)
    else:
        print('No ground truth was provided; not calculating recalls.')


def main():
    parser = argparse.ArgumentParser(description='SNSM-Feature-Maps')
    ################# Datsets and save directory
    parser.add_argument('--config_path', type=str, default='performance_bb.ini',
                        help='File name (with extension) to an ini file that stores most of the configuration data for SMSM')
    parser.add_argument('--data', type=str, required=True, help='Dataset name')                    
    parser.add_argument('--dataset_root_dir', type=str, default='/data/anuradha/Multi-RGBT/All_Datasets/Datasets/',
                        help='Dataset_root_dir as prefix.')
    parser.add_argument('--query_file_path', type=str, default='/data/anuradha/Multi-RGBT/All_Datasets/dataset_names/',
                        help='Path (with extension) to a text file that stores the save location and name of all query images in the THR dataset')
    parser.add_argument('--index_file_path', type=str, default='/data/anuradha/Multi-RGBT/All_Datasets/dataset_names/',
                        help='Path (with extension) to a text file that stores the save location and name of all database images in the RGB dataset')
    parser.add_argument('--ground_truth_path', type=str, default=None,
                        help='Path (with extension) to a file that stores the ground-truth data')      
    parser.add_argument('--result_save_folder', type=str, default='./ICASSP-Video-15-03-25/', help='Directory to save features and recalls')
    
    ################# Architectures
    parser.add_argument('--arch', default='vit_base', type=str,
        choices=['vit_tiny', 'vit_small', 'vit_base', 'vit_giant'], help='Architecture (support only ViT atm).')   
    parser.add_argument('--patch_size', default=8, type=int, help='Patch resolution of the model.')                                   
    parser.add_argument('--backbone', type=str, default='DINOv1', help='DINOv1, DINOv2, DINOv2g, ResNet50')
    parser.add_argument('--baseline', action='store_true', help='DINOv1, DINOv2, ResNet50')
    parser.add_argument('--emb_dim', type=int, required=True, help='Manually set ground truth threshold')    
    parser.add_argument('--encoder', default=1, type=int, help='Last layer of the model')
    
    ################# SNSM details
    parser.add_argument('--snsm', action='store_true', help='VLAD')
    parser.add_argument('--upscale_factor', type=int, default=4, help='Manually set ground truth threshold')
    parser.add_argument('--window_size', type=int, default=2, help='Manually set ground truth threshold')
    
    ################# VLAD details
    parser.add_argument('--cache_dir_thr', type=str, default=None, help='cache file path to centroids')
    parser.add_argument('--cache_dir_rgb', type=str, default=None, help='cache file path to centroids')
    parser.add_argument('--vocab_type', type=str, default=None, help='cache file path to centroids')
    parser.add_argument('--vlad', action='store_true', help='VLAD')
    parser.add_argument('--vlad_api', action='store_true', help='VLAD API developed by AnyLoc Authors')
    parser.add_argument('--api_vocab', type=str, default=None, help='VLAD API developed by AnyLoc Authors')
    parser.add_argument('--num_clusters', type=int, default=5, help='K value')
    parser.add_argument('--split_factor', type=int, default=5, help='Reshaping factor for VLAD input')
    parser.add_argument('--PlotWhatVLADSees', action='store_true', help='Projecting soft-weights on image: [This command in inactive]') 
    
    ################# GeM Pooling Layers details    
    parser.add_argument('--gem', action='store_true', help='Gem')
    parser.add_argument('--gem_p', type=float, default=3, help='Norm power')
    parser.add_argument('--gem_use_abs', action='store_true', help='Avoid negatives in the Gem features')
    parser.add_argument('--gem_elem_by_elem', action='store_true', help='Avoid negatives in the Gem features')
    
    ################# GeM Pooling Layers details
    parser.add_argument('--gap', action='store_true', help='VLAD')
    parser.add_argument('--gmp', action='store_true', help='VLAD')
    
    ################# Compute device
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
    
    #if opt.backbone=='ResNet50':
    #    model=ResNet50()#Model(opt)
    #    model.eval()
    #    model=model.to(opt.device)
    #elif opt.backbone=='DINO':
    if not opt.vlad_api:
       model=Model(opt)
    else:
       if opt.api_vocab == 'I':
          model = torch.hub.load("AnyLoc/DINO", "get_vlad_model", backbone="DINOv2", domain="indoor", device="cuda")   
       elif opt.api_vocab == 'U':
          model = torch.hub.load("AnyLoc/DINO", "get_vlad_model", backbone="DINOv2", domain="urban", device="cuda") 
       elif opt.api_vocab == 'A':
          model = torch.hub.load("AnyLoc/DINO", "get_vlad_model", backbone="DINOv2", domain="aerial", device="cuda")           
    
    opt.query_file_path = opt.query_file_path + opt.data.lower()+'_thr.txt'
    opt.index_file_path = opt.index_file_path + opt.data.lower()+'_rgb.txt'
    opt.dataset_root_dir = opt.dataset_root_dir + opt.data
    if opt.baseline:
       if opt.backbone=='DINOv1' or opt.backbone=='DINOv2' or opt.backbone=='DINOv2g':  
          opt.result_save_folder = opt.result_save_folder+opt.backbone + 'b' +str(opt.patch_size)+ '/'+'Baseline'+'/'+opt.data+'/'
       
       else:
          opt.result_save_folder = opt.result_save_folder+opt.backbone+'/'+'Baseline'+'/'+opt.data+'/'
    
    elif opt.vlad:
       if opt.backbone=='DINOv1' or opt.backbone=='DINOv2' or opt.backbone=='DINOv2g':
          opt.centres_root = opt.result_save_folder+opt.backbone+ 'b' +str(opt.patch_size)+'/'+'VLAD'+'/'+opt.data+'/' + '-K-'+ str(opt.num_clusters)+'/'
          opt.result_save_folder = opt.result_save_folder+opt.backbone+ 'b' +str(opt.patch_size)+'/'+'VLAD'+'/'+opt.data+'/' + '-K-'+ str(opt.num_clusters)+'/'
       else: 
          opt.result_save_folder = opt.result_save_folder+opt.backbone+'/'+'VLAD'+'/'+opt.data+'/'+'-K-'+ str(opt.num_clusters)+'/'   
    elif opt.vlad_api:
       opt.result_save_folder = opt.result_save_folder+'VLAD-API'+'/'+opt.data+'/'   
    elif opt.snsm:
       if opt.backbone=='DINOv1' or opt.backbone=='DINOv2' or opt.backbone=='DINOv2g': 
          opt.result_save_folder = opt.result_save_folder + opt.backbone+ 'b' + str(opt.patch_size)+'/' + 'SNSM' + '/' + opt.data+'/' + 'scale-' + str(opt.upscale_factor) + '_patchSize_' + str(opt.window_size)+'/'
       else:
          opt.result_save_folder = opt.result_save_folder+opt.backbone+ '/'+'SNSM'+'/'+opt.data+'/'+'scale-'+str(opt.upscale_factor) + '_patchSize_'+str(opt.window_size)+'/'
    elif opt.gem:
       if opt.backbone=='DINOv1' or opt.backbone=='DINOv2' or opt.backbone=='DINOv2g': 
          opt.result_save_folder = opt.result_save_folder+opt.backbone+ 'b' +str(opt.patch_size)+'/'+'GeM'+'/'+opt.data+'/'   
       else: 
          opt.result_save_folder = opt.result_save_folder+opt.backbone+'/'+'GeM'+'/'+opt.data+'/'    
    elif opt.gmp:
       if opt.backbone=='DINOv1' or opt.backbone=='DINOv2' or opt.backbone=='DINOv2g': 
          opt.result_save_folder = opt.result_save_folder+opt.backbone+ 'b' +str(opt.patch_size)+'/'+'GMP'+'/'+opt.data+'/'
       else:
          opt.result_save_folder = opt.result_save_folder+opt.backbone+ '/'+'GMP'+'/'+opt.data+'/'   
    elif opt.gap:
       if opt.backbone=='DINOv1' or opt.backbone=='DINOv2' or opt.backbone=='DINOv2g':
          opt.result_save_folder = opt.result_save_folder+opt.backbone+ 'b' +str(opt.patch_size)+'/'+'GAP'+'/'+opt.data+'/'   
       else:
          opt.result_save_folder = opt.result_save_folder+opt.backbone+ '/'+'GAP'+'/'+opt.data+'/'   
          
    if opt.backbone=='DINOv1' or opt.backbone=='DINOv2' or opt.backbone=='DINOv2g': 
       opt.recall_file = 'blk-'+str(12-opt.encoder)+'_scale_'+str(opt.upscale_factor)+'_patchSize_'+str(opt.window_size)+'x'+str(opt.window_size)+'.txt'
    elif opt.backbone=='ResNet50':
       if not opt.baseline:
          opt.recall_file = 'lay-'+str(opt.encoder)+'_scale_'+str(opt.upscale_factor)+'_patchSize_'+str(opt.window_size)+'x'+str(opt.window_size)+'.txt'    
       else:   
          opt.recall_file = 'lay-'+str(opt.encoder)+'.txt'     
    elif opt.vlad_api:
       opt.recall_file = 'recalls.txt'   
    dataset = PlaceDataset(opt.query_file_path, opt.index_file_path, opt.dataset_root_dir, None,
                           config['feature_extract'])
   
    qImgs = PlaceDataset(None, opt.query_file_path, opt.dataset_root_dir, None, config['feature_extract'])
    dbImgs = PlaceDataset(None, opt.index_file_path, opt.dataset_root_dir, None, config['feature_extract'])
    
    #for i, k in dbImgs:
    #    print(i.shape)
    if opt.vlad:
       
       if opt.vocab_type=='rgbt':
          opt.cache_dir_thr = opt.centres_root + 'thr-vocab/'
          opt.cache_dir_rgb = opt.centres_root + 'rgb-vocab/'
       elif opt.vocab_type=='rgb':
          opt.cache_dir_thr = opt.centres_root + 'rgb-vocab/'
          opt.cache_dir_rgb = opt.centres_root + 'rgb-vocab/'
       elif opt.vocab_type=='thr':
          opt.cache_dir_thr = opt.centres_root + 'thr-vocab/'
          opt.cache_dir_rgb = opt.centres_root + 'thr-vocab/'
       else: 
          opt.cache_dir_thr = opt.cache_dir_thr 
          opt.cache_dir_rgb = opt.cache_dir_rgb                 

    
    #if not exists(opt.cache_dir_rgb) or not exists(opt.cache_dir_thr):
    #    makedirs(opt.cache_dir_rgb)
    #    makedirs(opt.cache_dir_thr)
        
    # Comment if features are already avaliable 
    qfeats_loc = feature_extraction(qImgs, model, opt.device, opt, config, 'qfeats_dino_gem.npy' , None, opt.cache_dir_thr) 
    dbfeats_loc = feature_extraction(dbImgs, model, opt.device, opt, config, 'dbfeats_dino_gem.npy', None, opt.cache_dir_rgb) 
    
    # Uncomment to read features from the saved files 
    #qfeats_loc = np.load('qfeats_dino_snsm.npy')
    #dbfeats_loc = np.load('dbfeats_dino_snsm.npy')
    
    # Uncomment if saving features is not necessary 
    if not exists(opt.result_save_folder):
        makedirs(opt.result_save_folder)
    np.save(opt.result_save_folder+'qfeats_dino_gem.npy.npy', qfeats_loc)
    np.save(opt.result_save_folder+'dbfeats_dino_gem.npy', dbfeats_loc)

    # Match features and compute recalls 
    feature_match(qfeats_loc, dbfeats_loc, dataset, opt.device, opt, config, opt.recall_file)
    
    #Match(qfeats, dbfeats, config)
    #[print(q,'\t',p) for q, p in enumerate(preds)]
    
if __name__ == "__main__":
    pl.utilities.seed.seed_everything(seed=1, workers=True)
    main()
        
