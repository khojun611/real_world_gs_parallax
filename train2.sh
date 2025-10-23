
#python train.py -s data/ref_nerf/coffee --eval --white_background   
#python train.py -s data/ref_nerf/helmet --eval  --white_background  --lambda_normal_smooth 1.0
#python train.py -s data/ref_nerf/ball --eval  --white_background --lambda_normal_smooth 1.0 
#python train.py -s data/ref_nerf/teapot --eval  --white_background 
#python train.py -s data/ref_nerf/toaster --eval  --white_background   
#python train.py -s data/ref_nerf/car --eval  --white_background 

#python train.py -s ../../3DGS-DR/data/GlossySynthetic/angel_blender --eval -indirect_from_iter 10000 --initial 1 --init_until_iter 3000  -r 1 --volume_render_until_iter 0 --white_background 
#python train.py -s ../../3DGS-DR/data/GlossySynthetic/potion_blender --eval  --white_background   
#python train.py -s ../../3DGS-DR/data/GlossySynthetic/horse_blender --eval  --white_background   
#python train.py -s ../../3DGS-DR/data/GlossySynthetic/luyu_blender --eval  --white_background    
#python train.py -s ../../3DGS-DR/data/GlossySynthetic/teapot_blender --eval  --white_background 
#python train.py -s ../../3DGS-DR/data/GlossySynthetic/bell_blender --eval  --white_background   
#python train.py -s ../../3DGS-DR/data/GlossySynthetic/tbell_blender --eval  --white_background  --lambda_normal_smooth 1.0 
#python train.py -s ../../3DGS-DR/data/GlossySynthetic/cat_blender --eval  --white_background 


#python train.py -s ../3DGS-DR/data/ref_real/gardenspheres -m ./output_abc/sedan/sedan-0711_0646 --eval --iterations 20000 --indirect_from_iter 10000 --volume_render_until_iter 0 --initial 1 --init_until_iter 3000 --lambda_normal_smooth 0.45 -r 4 
#python train.py -s ../3DGS-DR/data/ref_real/toycar --eval --iterations 20000 --indirect_from_iter 10000 --volume_render_until_iter 0  --initial 1 --init_until_iter 3000  -r 4
#python train.py -s ../3DGS-DR/data/ref_real/sedan --eval --iterations 20000 --indirect_from_iter 10000 --volume_render_until_iter 0  --initial 1 --init_until_iter 3000  -r 4
#python train.py -s ../3DGS-DR/data/ref_real/sedan --eval --iterations 20000 --indirect_from_iter 10000  -r 4
#python train.py -s data/ref_nerf/helmet --iterations 50000 --uncertainty_from_iter 31000 --eval --white_background  --lambda_normal_smooth 1.0



#python train.py -s ../../ref-gaussian/data/realworld_scenario/chrome6 --eval --iterations 50000 --indirect_from_iter 10000 --initial 1 --init_until_iter 3000  
#python train.py -s ../../ref-gaussian/data/realworld_scenario/namby1 -r 2 --eval --iterations 50000 --indirect_from_iter 10000 --initial 1 --init_until_iter 3000  


#python train.py -s data/real50/bell -m output/metallic/bell --eval --iterations 30000 --indirect_from_iter 10000 --initial 1 --init_until_iter 3000 --volume_render_until_iter 0
#python train.py -s data/real50/chrome -m output/metallic2/chrome --eval --iterations 30000 --indirect_from_iter 10000 --initial 1 --init_until_iter 3000 
#python train.py -s data/real50/pot -m output/metallic2/pot --eval --iterations 30000 --indirect_from_iter 10000 --initial 1 --init_until_iter 3000 
#python train.py -s data/real50/tray -m output/metallic2/tray --eval --iterations 30000 --indirect_from_iter 10000 --initial 1 --init_until_iter 3000 
#python train.py -s data/real50/blackcar -m output/metallic/blackcar --eval --iterations 30000 --indirect_from_iter 10000 --initial 1 --init_until_iter 3000 
#python train.py -s data/real50/graycar -m output/metallic/graycar --eval --iterations 30000 --indirect_from_iter 10000 --initial 1 --init_until_iter 3000 
#python train.py -s data/real50/helmet -m output/metallic/helmet --eval --iterations 30000 --indirect_from_iter 10000 --initial 1 --init_until_iter 3000 
#python train.py -s data/real_colmap/bell2 -m output/metallic_new/bell2 --eval --iterations 30000 --indirect_from_iter 10000 --initial 1 --init_until_iter 3000
#python train.py -s data/real_colmap/bell -m output/metallic_new/bell --eval --iterations 30000 --indirect_from_iter 10000 --initial 1 --init_until_iter 3000
#python train.py -s data/real_colmap/chrome -m output/metallic_new/chrome --eval --iterations 30000 --indirect_from_iter 10000 --initial 1 --init_until_iter 3000

#python train.py -s data/real_play/bell -m output/real_play/bell --eval --iterations 30000 --indirect_from_iter 10000 --initial 1 --init_until_iter 3000  -r 1 --volume_render_until_iter 0
#python train.py -s data/real_play/chrome -m output/real_play/chrome --eval --iterations 30000 --indirect_from_iter 10000 --initial 1 --init_until_iter 3000  -r 1 --volume_render_until_iter 0
#python train.py -s data/real_play/pot -m output/real_play/pot --eval --iterations 30000 -indirect_from_iter 10000 --initial 1 --init_until_iter 3000  -r 1 --volume_render_until_iter 0
python train.py -s data/real50/bell -m output/real50_rgb_uc2/bell --eval --iterations 30000 --indirect_from_iter 10000 --initial 1 --init_until_iter 3000  
python train.py -s data/real50/chrome -m output/real50_rgb_uc2/chrome --eval --iterations 30000 --indirect_from_iter 10000 --initial 1 --init_until_iter 3000   
python train.py -s data/real50/pot -m output/real50_rgb_uc2/pot --eval --iterations 30000 --indirect_from_iter 10000 --initial 1 --init_until_iter 3000   




#python train.py -s data/real20/tray -m output/real20/tray --eval --iterations 30000 --indirect_from_iter 10000 --initial 1 --init_until_iter 3000 

#python train.py -s ../../ref-gaussian/data/ref_nerf/ball --iterations 50000  --eval --white_background  --lambda_normal_smooth 1.0
#python train.py -s ../../ref-gaussian/data/ref_nerf/car --iterations 50000  --eval --white_background  --lambda_normal_smooth 1.0 
#python train.py -s ../../ref-gaussian/data/ref_nerf/teapot --iterations 50000  --eval --white_background  --lambda_normal_smooth 1.0
#python train.py -s ../../ref-gaussian/data/ref_nerf/toaster --iterations 50000  --eval --white_background  --lambda_normal_smooth 1.0


#python train.py -s ../../ref-gaussian/data/realworld_scenario/chrome5 --eval --iterations 50000 --indirect_from_iter 10000 
#python train.py -s ../../ref-gaussian/data/realworld_scenario/car_gear --eval --iterations 50000 --indirect_from_iter 10000 
#python train.py -s ../../ref-gaussian/data/realworld_scenario/white_car --eval --iterations 50000 --indirect_from_iter 10000 
#python train.py -s ../../ref-gaussian/data/realworld_scenario/tv2 --eval --iterations 50000 --indirect_from_iter 10000 
#python train.py -s ../../ref-gaussian/data/realworld_scenario/car_segmented --eval --iterations 50000 --indirect_from_iter 10000 
#python train.py -s ../../ref-gaussian/data/realworld_scenario/disk --eval --iterations 50000 --indirect_from_iter 10000 
#python train.py -s ../../ref-gaussian/data/realworld_scenario/postcase1 --eval --iterations 50000 --indirect_from_iter 10000 

