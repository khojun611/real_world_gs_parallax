
#python train.py -s ../../ref_gaussian/data/ref_nerf/coffee --eval --white_background   
python train.py -s ../ref_gaussian/data/ref_nerf/helmet --eval  --white_background  --lambda_normal_smooth 1.0 --lambda_hybrid 0.1 --uncertainty_lr 0.001 --uncertainty_from_iter 5000
python train.py -s ../ref_gaussian/data/ref_nerf/ball --eval  --white_background --lambda_normal_smooth 1.0 
python train.py -s ../ref_gaussian/data/ref_nerf/teapot --eval  --white_background 
python train.py -s ../ref_gaussian/data/ref_nerf/toaster --eval  --white_background   
python train.py -s ../ref_gaussian/data/ref_nerf/car --eval  --white_background 

# python train.py -s data/GlossySynthetic/angel_blender --eval --white_background   
# python train.py -s data/GlossySynthetic/potion_blender --eval  --white_background   
# python train.py -s data/GlossySynthetic/horse_blender --eval  --white_background   
# python train.py -s data/GlossySynthetic/luyu_blender --eval  --white_background    
# python train.py -s data/GlossySynthetic/teapot_blender --eval  --white_background 
# python train.py -s data/GlossySynthetic/bell_blender --eval  --white_background   
# python train.py -s data/GlossySynthetic/tbell_blender --eval  --white_background  --lambda_normal_smooth 1.0
# python train.py -s data/GlossySynthetic/cat_blender --eval  --white_background 


python train.py -s ../3DGS-DR/data/ref_real/gardenspheres --eval --iterations 20000 --indirect_from_iter 10000 --volume_render_until_iter 0 --initial 1 --init_until_iter 3000 --lambda_normal_smooth 0.45 -r 4 
python train.py -s ../3DGS-DR/data/ref_real/toycar --eval --iterations 20000 --indirect_from_iter 10000 --volume_render_until_iter 0  --initial 1 --init_until_iter 3000  -r 4
python train.py -s ../3DGS-DR/data/ref_real/sedan --eval --iterations 20000 --indirect_from_iter 10000 --volume_render_until_iter 0  --initial 1 --init_until_iter 3000  -r 4
python train.py -s ../3DGS-DR/data/ref_real/sedan --eval --iterations 20000 --indirect_from_iter 10000  -r 4