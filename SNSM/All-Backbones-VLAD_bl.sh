
for dataS in 'M3FD' 'Boson' 'FLIR' 'INO' 'LLVIP' 'VOT' 'VTUAV' #
do     
   python All-Backbones-VLAD_bl.py  --upscale_factor=4 --window_size=2 --emb_dim=3136 --backbone=DINOv1 --snsm --encoder=1 --data=$dataS
 
done




#for dataS in 'M3FD' #'Boson' 'FLIR' 'INO' 'LLVIP' 'M3FD' 'VOT' 'VTUAV' #
'''do

   ###### VLAD on DINOv1      
   #python All-Backbones-VLAD_bl.py  --emb_dim=602112 --backbone=DINOv1 --vlad --arch=vit_base --patch_size=8 --encoder=1 --data=Boson --dataset_root_dir=/home/anuradha/MyResearch@Sindhu/DataSets/Multi-RGBT/All_Datasets/Datasets-MSCN/ --query_file_path=/home/anuradha/MyResearch@Sindhu/DataSets/Multi-RGBT/All_Datasets/dataset_names/ --index_file_path=/home/anuradha/MyResearch@Sindhu/DataSets/Multi-RGBT/All_Datasets/dataset_names/ --result_save_folder=./P30k/MSCN/ --ground_truth_path=None --cache_dir_thr=./P30k/MSCN/centroids/vit_base_p30k-mscn_16_desc_cen.hdf5 --cache_dir_rgb=./P30k/MSCN/centroids/vit_base_p30k-mscn_16_desc_cen.hdf5
   #python All-Backbones-VLAD_bl.py  --emb_dim=49152 --vlad_api --encoder=1 --data=$dataS --api_vocab=U --result_save_folder=./RGB-T-VLAD-API/Indoor/ --ground_truth_path=None
   ###### SNSM on ResNet50      
   python All-Backbones-VLAD_bl.py  --upscale_factor=4 --window_size=2 --emb_dim=3136 --backbone=DINOv1 --snsm --encoder=1 --data=$dataS 
   #python All-Backbones-VLAD_bl.py  --upscale_factor=4 --window_size=4 --emb_dim=784 --backbone=DINOv1 --snsm --encoder=1 --data=$dataS 
   #python All-Backbones-VLAD_bl.py  --upscale_factor=4 --window_size=7 --emb_dim=256 --backbone=DINOv1 --snsm --encoder=1 --data=$dataS 
   #python All-Backbones-VLAD_bl.py  --upscale_factor=4 --window_size=8 --emb_dim=196 --backbone=DINOv1 --snsm --encoder=1 --data=$dataS 
   #python All-Backbones-VLAD_bl.py  --upscale_factor=4 --window_size=14 --emb_dim=64 --backbone=DINOv1 --snsm --encoder=1 --data=$dataS
   #python All-Backbones-VLAD_bl.py  --upscale_factor=4 --window_size=16 --emb_dim=49 --backbone=DINOv1 --snsm --encoder=1 --data=$dataS
   #python All-Backbones-VLAD_bl.py  --upscale_factor=4 --window_size=28 --emb_dim=16 --backbone=DINOv1 --snsm --encoder=1 --data=$dataS
   #python All-Backbones-VLAD_bl.py  --upscale_factor=4 --window_size=56 --emb_dim=4 --backbone=DINOv1 --snsm --encoder=1 --data=$dataS

done    
'''




###### SNSM on ResNet50      
#python All-Backbones-VLAD_bl.py  --emb_dim=1024 --backbone=DINOv2 --snsm --encoder=1 --data=None --dataset_root_dir=/home/anuradha/MyResearch@Sindhu/DataSets/ --query_file_path=/home/anuradha/MyResearch@Sindhu/NV_GSV_TokyoV3/dataset_imagenames/nordland_imageNames_query.txt --index_file_path=/home/anuradha/MyResearch@Sindhu/NV_GSV_TokyoV3/dataset_imagenames/nordland_imageNames_index.txt --result_save_folder=./nordland/SNSM/DINOv2g14/ --ground_truth_path=/home/anuradha/MyResearch@Sindhu/NV_GSV_TokyoV3/dataset_imagenames/gts/nordland.npz    

###### SNSM on ResNet50      
#python All-Backbones-VLAD_bl.py  --emb_dim=784 --backbone=ResNet50 --snsm --encoder=3 --data=None --dataset_root_dir=/home/anuradha/MyResearch@Sindhu/DataSets/ --query_file_path=/home/anuradha/MyResearch@Sindhu/NV_GSV_TokyoV3/dataset_imagenames/nordland_imageNames_query.txt --index_file_path=/home/anuradha/MyResearch@Sindhu/NV_GSV_TokyoV3/dataset_imagenames/nordland_imageNames_index.txt --result_save_folder=./nordland/ --ground_truth_path=/home/anuradha/MyResearch@Sindhu/NV_GSV_TokyoV3/dataset_imagenames/gts/nordland.npz      


###### VLAD_API on DINO      
#python All-Backbones-VLAD_bl.py  --emb_dim=49152 --vlad_api --encoder=3 --data=None --dataset_root_dir=/home/anuradha/MyResearch@Sindhu/DataSets/ --query_file_path=/home/anuradha/MyResearch@Sindhu/NV_GSV_TokyoV3/dataset_imagenames/nordland_imageNames_query.txt --index_file_path=/home/anuradha/MyResearch@Sindhu/NV_GSV_TokyoV3/dataset_imagenames/nordland_imageNames_index.txt --result_save_folder=./nordland/VLAD_API/Indoor/ --ground_truth_path=/home/anuradha/MyResearch@Sindhu/NV_GSV_TokyoV3/dataset_imagenames/gts/nordland.npz 
#--cache_dir_thr=./nordland/VLAD/centroids-db-summer/ResNet50_Nordland-M-Lay3_16_desc_cen.hdf5 --cache_dir_rgb=./nordland/VLAD/centroids-db-summer/ResNet50_Nordland-M-Lay3_16_desc_cen.hdf5

###### VLAD on DINOv1      
#python All-Backbones-VLAD_bl.py  --emb_dim=75264 --backbone=DINOv1 --vlad --arch=vit_small --patch_size=16 --encoder=1 --data=None --dataset_root_dir=/home/anuradha/MyResearch@Sindhu/DataSets/ --query_file_path=/home/anuradha/MyResearch@Sindhu/NV_GSV_TokyoV3/dataset_imagenames/nordland_imageNames_query.txt --index_file_path=/home/anuradha/MyResearch@Sindhu/NV_GSV_TokyoV3/dataset_imagenames/nordland_imageNames_index.txt --result_save_folder=./nordland/DINO/ --ground_truth_path=/home/anuradha/MyResearch@Sindhu/NV_GSV_TokyoV3/dataset_imagenames/gts/nordland.npz --cache_dir_thr=./nordland/DINO/centroids/vit_small_Nordland-summer_16_desc_cen.hdf5 --cache_dir_rgb=./nordland/DINO/centroids/vit_small_Nordland-summer_16_desc_cen.hdf5
      

###### VLAD on ResNet50      
#python All-Backbones-VLAD_bl.py  --emb_dim=200704 --backbone=ResNet50 --vlad --encoder=3 --data=None --dataset_root_dir=/home/anuradha/MyResearch@Sindhu/DataSets/ --query_file_path=/home/anuradha/MyResearch@Sindhu/NV_GSV_TokyoV3/dataset_imagenames/nordland_imageNames_query.txt --index_file_path=/home/anuradha/MyResearch@Sindhu/NV_GSV_TokyoV3/dataset_imagenames/nordland_imageNames_index.txt --result_save_folder=./nordland/VLAD/ --ground_truth_path=/home/anuradha/MyResearch@Sindhu/NV_GSV_TokyoV3/dataset_imagenames/gts/nordland.npz --cache_dir_thr=./nordland/VLAD/centroids-db-summer/ResNet50_Nordland-M-Lay3_16_desc_cen.hdf5 --cache_dir_rgb=./nordland/VLAD/centroids-db-summer/ResNet50_Nordland-M-Lay3_16_desc_cen.hdf5


###### Baseline ResNet50
#python All-Backbones-VLAD_bl.py  --emb_dim=200704 --backbone=ResNet50 --encoder=3 --data=None --dataset_root_dir=/home/anuradha/MyResearch@Sindhu/DataSets/Pittsburgh-Masks/Masked-Images/ --query_file_path=/home/anuradha/MyResearch@Sindhu/NV_GSV_TokyoV3/dataset_imagenames/pitts30k_imageNames_query.txt --index_file_path=/home/anuradha/MyResearch@Sindhu/NV_GSV_TokyoV3/dataset_imagenames/pitts30k_imageNames_index.txt --result_save_folder=./P30k/Masked-Images/ --ground_truth_path=/home/anuradha/MyResearch@Sindhu/NV_GSV_TokyoV3/dataset_imagenames/gts/pitts30k_test.npz


       
       
