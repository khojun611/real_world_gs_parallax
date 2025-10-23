# python train.py -s ../gaussian-splatting/artifacts/02 --eval -r 2 --white_background 
# python train.py -s ../gaussian-splatting/artifacts/ceramic --eval -r 2
# python train.py -s ../mip-splatting/artifacts/yeonjeok --eval
# python train.py -s ../mip-splatting/artifacts/pouch --eval 
# python train.py -s ../mip-splatting/artifacts/mirror --eval 
# python train.py -s ../mip-splatting/artifacts/ceramic --eval 
# python train.py -s ../mip-splatting/artifacts/cup --eval 
# python train.py -s ../mip-splatting/artifacts/hinge --eval -r 8
# python train.py -s ../3DGS-DR/data/chrome_back --eval -r 4
# python train.py -s ./data/chrome_table --eval -r 2
# python train.py -s ./data/custom/ball_back --eval 



#python train.py -s ../3DGS-DR/data/GlossyReal/bear --eval --iterations 50000 --indirect_from_iter 10000 --initial 1 --init_until_iter 3000  -r 1 --volume_render_until_iter 0 #--lambda_hybrid 0.1 --uncertainty_lr 0.001 --uncertainty_from_iter 10000
#python train.py -s ../3DGS-DR/data/GlossyReal/bunny --eval --iterations 50000 --indirect_from_iter 10000 --initial 1 --init_until_iter 3000  -r 1 --volume_render_until_iter 0 #--lambda_hybrid 0.1 --uncertainty_lr 0.001 --uncertainty_from_iter 10000
#python train.py -s ../3DGS-DR/data/GlossyReal/coral --eval --iterations 50000 --indirect_from_iter 10000 --initial 1 --init_until_iter 3000  -r 1 --volume_render_until_iter 0 #--lambda_hybrid 0.1 --uncertainty_lr 0.001 --uncertainty_from_iter 10000
#python train.py -s ../3DGS-DR/data/GlossyReal/maneki --eval --iterations 50000 --indirect_from_iter 10000 --initial 1 --init_until_iter 3000  -r 1 --volume_render_until_iter 0 #--lambda_hybrid 0.1 --uncertainty_lr 0.001 --uncertainty_from_iter 10000
#python train.py -s ../3DGS-DR/data/GlossyReal/vase --eval --iterations 50000 --indirect_from_iter 10000 --initial 1 --init_until_iter 3000  -r 1 --volume_render_until_iter 0 #--lambda_hybrid 0.1 --uncertainty_lr 0.001 --uncertainty_from_iter 10000
#python train.py -s data/realworld_scenario/car_segmented --eval --iterations 50000 --indirect_from_iter 10000 --initial 1 --init_until_iter 3000  -r 1 --volume_render_until_iter 0 #--lambda_hybrid 0.1 --uncertainty_lr 0.001 --uncertainty_from_iter 10000



# python train.py -s ../3DGS-DR/data/ref_real/sedan -m output/sedan4_newcon --eval --iterations 20000 --indirect_from_iter 10000 --initial 1 --init_until_iter 3000  -r 8 --volume_render_until_iter 0 --lambda_hybrid 0.1 --uncertainty_lr 0.001 --uncertainty_from_iter 10000
#python train.py -s ../3DGS-DR/data/ref_real/sedan -m output/uncer/sedan --eval --iterations 20000 --indirect_from_iter 10000 --initial 1 --init_until_iter 3000  -r 8 --volume_render_until_iter 0 #--lambda_hybrid 0.1 --uncertainty_lr 0.001 --uncertainty_from_iter 10000




 #--lambda_hybrid 0.1 --uncertainty_lr 0.001 --uncertainty_from_iter 10000
#python train.py -s ../ref-gaussian/data/real2/chrome -m output/chrome_dual --eval --iterations 20000 --indirect_from_iter 10000 --initial 1 --init_until_iter 3000  --volume_render_until_iter 0 #--lambda_hybrid 0.1 --uncertainty_lr 0.001 --uncertainty_from_iter 10000
#python train.py -s ../ref-gaussian/data/real2/mirror --eval --iterations 20000 --indirect_from_iter 10000 --initial 1 --init_until_iter 3000  --volume_render_until_iter 0 #--lambda_hybrid 0.1 --uncertainty_lr 0.001 --uncertainty_from_iter 10000
#python train.py -s ../ref-gaussian/data/real2/pot --eval --iterations 20000 --indirect_from_iter 10000 --initial 1 --init_until_iter 3000  --volume_render_until_iter 0 #--lambda_hybrid 0.1 --uncertainty_lr 0.001 --uncertainty_from_iter 10000
##python train.py -s ../ref-gaussian/data/real2/tray --eval --iterations 20000 --indirect_from_iter 10000 --initial 1 --init_until_iter 3000  --volume_render_until_iter 0 #--lambda_hybrid 0.1 --uncertainty_lr 0.001 --uncertainty_from_iter 10000
#python train.py -s ../ref-gaussian/data/real2/helmet --eval --iterations 20000 --indirect_from_iter 10000 --initial 1 --init_until_iter 3000  --volume_render_until_iter 0
#python train.py -s ../ref-gaussian/data/real2/car_night --eval --iterations 20000 --indirect_from_iter 10000 --initial 1 --init_until_iter 3000  --volume_render_until_iter 0
#python train.py -s ../ref-gaussian/data/real2/car_figure --eval --iterations 20000 --indirect_from_iter 10000 --initial 1 --init_until_iter 3000  --volume_render_until_iter 0

python train.py -s ../ref-gaussian/data/real50/bell --eval --iterations 30000 --indirect_from_iter 10000 --initial 1 --init_until_iter 3000  --volume_render_until_iter 0 




#python train.py -s ../mount1/ref_real/sedan --eval --iterations 20000 --indirect_from_iter 10000 --initial 1 --init_until_iter 3000  -r 8 --volume_render_until_iter 0 #--lambda_hybrid 0.1 --uncertainty_lr 0.001 --uncertainty_from_iter 10000
#python train.py -s ../mount1/ref_real/toycar --eval --iterations 20000 --indirect_from_iter 10000 --initial 1 --init_until_iter 3000  -r 4 --volume_render_until_iter 0 #--lambda_hybrid 0.1 --uncertainty_lr 0.001 --uncertainty_from_iter 10000
#python train.py -s ../mount1/ref_real/gardenspheres --eval --iterations 20000 --indirect_from_iter 10000 --initial 1 --init_until_iter 3000  -r 4 --volume_render_until_iter 0 --lambda_normal_smooth 0.45 #--lambda_hybrid 0.1 --uncertainty_lr 0.001 --uncertainty_from_iter 10000
# python train.py -s ../3DGS-DR/data/chrome6 --eval --iterations 50000 --indirect_from_iter 10000 --initial 1 --init_until_iter 3000 --volume_render_until_iter 0 --lambda_normal_smooth 0.45 -r 2 #--lambda_hybrid 0.1 --uncertainty_lr 0.001 --uncertainty_from_iter 10000






#python train.py -s data/ref_nerf/helmet --iterations 50000  --eval --white_background  --lambda_normal_smooth 1.0 
#python train.py -s data/ref_nerf/coffee --iterations 50000  --eval --white_background  --lambda_normal_smooth 1.0 
##python train.py -s data/ref_nerf/ball --iterations 50000  --eval --white_background  --lambda_normal_smooth 1.0 
#python train.py -s data/ref_nerf/car --iterations 50000  --eval --white_background  --lambda_normal_smooth 1.0 
#python train.py -s data/ref_nerf/teapot --iterations 50000  --eval --white_background  --lambda_normal_smooth 1.0 
#python train.py -s data/ref_nerf/toaster --iterations 50000  --eval --white_background  --lambda_normal_smooth 1.0

#python train.py -s data/realworld_scenario/white_car --eval --iterations 50000 --indirect_from_iter 10000 
#python train.py -s data/realworld_scenario/tv2 --eval --iterations 50000 --indirect_from_iter 10000 --lambda_normal_smooth 1.0 #--lambda_hybrid 0.1 --uncertainty_lr 0.001 --uncertainty_from_iter 5000 --lambda_normal_smooth 1.0 

#python train.py -s data/realworld_scenario/disk --eval --iterations 50000 --indirect_from_iter 10000 --lambda_normal_smooth 1.0  #--lambda_hybrid 0.1 --uncertainty_lr 0.001 --uncertainty_from_iter 5000 --lambda_normal_smooth 1.0 
#python train.py -s data/realworld_scenario/postcase1 --eval --iterations 50000 --indirect_from_iter 10000 --lambda_normal_smooth 1.0  #--lambda_hybrid 0.1 --uncertainty_lr 0.001 --uncertainty_from_iter 5000 --lambda_normal_smooth 1.0 
#python train.py -s data/realworld_scenario/chrome5 --eval --iterations 50000 --indirect_from_iter 10000 --lambda_normal_smooth 1.0  #--lambda_hybrid 0.1 --uncertainty_lr 0.001 --uncertainty_from_iter 5000 --lambda_normal_smooth 1.0 
#python train.py -s data/realworld_scenario/car_gear --eval --iterations 50000 --indirect_from_iter 10000 --lambda_normal_smooth 1.0  #--lambda_hybrid 0.1 --uncertainty_lr 0.001 --uncertainty_from_iter 5000 --lambda_normal_smooth 1.0 

# python train.py -s data/realworld_scenario/car_segmented --eval --iterations 50000 --indirect_from_iter 10000 --lambda_normal_smooth 1.0 #--lambda_hybrid 0.1 --uncertainty_lr 0.001 --uncertainty_from_iter 5000 --lambda_normal_smooth 1.0 


#python train.py -s ../ref-gaussian/data/realworld_scenario/chrome6 --eval --iterations 50000 --indirect_from_iter 10000 --initial 1 --init_until_iter 3000  
#python train.py -s ../ref-gaussian/data/realworld_scenario/namby1 -r 2 --eval --iterations 50000 --indirect_from_iter 10000 --initial 1 --init_until_iter 3000  
#python train.py -s ../ref-gaussian/data/realworld_scenario/namby2 -r 2 --eval --iterations 30000 --indirect_from_iter 10000 --initial 1 --init_until_iter 3000  
#python train.py -s ../ref-gaussian/data/realworld_scenario/dog -r 2 --eval --iterations 30000 --indirect_from_iter 10000 --initial 1 --init_until_iter 3000  