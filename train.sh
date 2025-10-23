
#python train.py -s data/ref_nerf/coffee --eval --white_background   
#python train.py -s data/ref_nerf/helmet --eval  --white_background  --lambda_normal_smooth 1.0
#python train.py -s data/ref_nerf/ball --eval  --white_background --lambda_normal_smooth 1.0 
#python train.py -s data/ref_nerf/teapot --eval  --white_background 
#python train.py -s data/ref_nerf/toaster --eval  --white_background   
#python train.py -s data/ref_nerf/car --eval  --white_background 

python train.py -s ../3DGS-DR/data/GlossySynthetic/angel_blender --eval --white_background --lambda_hybrid 0.3 --uncertainty_lr 0.001 --uncertainty_from_iter 35000
python train.py -s ../3DGS-DR/data/GlossySynthetic/potion_blender --eval  --white_background   --lambda_hybrid 0.3 --uncertainty_lr 0.001 --uncertainty_from_iter 35000
python train.py -s ../3DGS-DR/data/GlossySynthetic/horse_blender --eval  --white_background   --lambda_hybrid 0.3 --uncertainty_lr 0.001 --uncertainty_from_iter 35000
python train.py -s ../3DGS-DR/data/GlossySynthetic/luyu_blender --eval  --white_background    --lambda_hybrid 0.3 --uncertainty_lr 0.001 --uncertainty_from_iter 35000
python train.py -s ../3DGS-DR/data/GlossySynthetic/teapot_blender --eval  --white_background --lambda_hybrid 0.3 --uncertainty_lr 0.001 --uncertainty_from_iter 35000
python train.py -s ../3DGS-DR/data/GlossySynthetic/bell_blender --eval  --white_background   --lambda_hybrid 0.3 --uncertainty_lr 0.001 --uncertainty_from_iter 35000
python train.py -s ../3DGS-DR/data/GlossySynthetic/tbell_blender --eval  --white_background  --lambda_normal_smooth 1.0 --lambda_hybrid 0.3 --uncertainty_lr 0.001 --uncertainty_from_iter 35000
python train.py -s ../3DGS-DR/data/GlossySynthetic/cat_blender --eval  --white_background --lambda_hybrid 0.3 --uncertainty_lr 0.001 --uncertainty_from_iter 35000


#python train.py -s ../3DGS-DR/data/ref_real/gardenspheres -m ./output_abc/sedan/sedan-0711_0646 --eval --iterations 20000 --indirect_from_iter 10000 --volume_render_until_iter 0 --initial 1 --init_until_iter 3000 --lambda_normal_smooth 0.45 -r 4 
#python train.py -s ../3DGS-DR/data/ref_real/toycar --eval --iterations 20000 --indirect_from_iter 10000 --volume_render_until_iter 0  --initial 1 --init_until_iter 3000  -r 4
#python train.py -s ../3DGS-DR/data/ref_real/sedan --eval --iterations 20000 --indirect_from_iter 10000 --volume_render_until_iter 0  --initial 1 --init_until_iter 3000  -r 4
#python train.py -s ../3DGS-DR/data/ref_real/sedan --eval --iterations 20000 --indirect_from_iter 10000  -r 4
#python train.py -s data/ref_nerf/helmet --iterations 50000 --uncertainty_from_iter 31000 --eval --white_background  --lambda_normal_smooth 1.0