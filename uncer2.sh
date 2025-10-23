#python train.py -s data/ref_nerf/helmet --iterations 50000 --uncertainty_from_iter 20000 --eval --white_background  --lambda_normal_smooth 1.0
#python train.py -s ./data/disk --iterations 50000 --uncertainty_from_iter 45000 --eval --lambda_normal_smooth 1.0 --lambda_hybrid 0.3 --uncertainty_lr 0.001
#python train.py -s ./data/car_gear --iterations 50000 --uncertainty_from_iter 45000 --eval --lambda_normal_smooth 1.0 --lambda_hybrid 0.3 --uncertainty_lr 0.001
#python train.py -s .//data/realworld_scenario/tv --iterations 50000 --uncertainty_from_iter 45000 --eval --lambda_normal_smooth 1.0 --lambda_hybrid 0.3 --uncertainty_lr 0.001
#python train.py -s ./data/chrome5 --iterations 50000 --uncertainty_from_iter 45000 --eval  --lambda_normal_smooth 1.0 --lambda_hybrid 0.3 --uncertainty_lr 0.001
#python train.py -s ./data/realworld_scenario/postcase1 --iterations 50000 --uncertainty_from_iter 45000 --eval --lambda_normal_smooth 1.0 --lambda_hybrid 0.3 --uncertainty_lr 0.001
#python train.py -s ./data/realworld_scenario/car_segmented --iterations 50000 --uncertainty_from_iter 45000 --eval --lambda_normal_smooth 1.0 --lambda_hybrid 0.3 --uncertainty_lr 0.001




python train.py -s ../3DGS-DR/data/ref_real/toycar --eval --iterations 50000 
python train.py -s ../3DGS-DR/data/ref_real/sedan --eval --iterations 28000 --uncertainty_from_iter 20000 --indirect_from_iter 10000 --volume_render_until_iter 0  --initial 1 --init_until_iter 3000  -r 4 --lambda_hybrid 0.5 --uncertainty_lr 0.001 
python train.py -s ../3DGS-DR/data/ref_real/gardenspheres --eval --iterations 28000 --uncertainty_from_iter 20000 --indirect_from_iter 10000 --volume_render_until_iter 0 --initial 1 --init_until_iter 3000 --lambda_normal_smooth 0.45 -r 4 --lambda_hybrid 0.5 --uncertainty_lr 0.001 


#python train.py -s data/ref_nerf/helmet --iterations 50000 --uncertainty_from_iter 20000 --eval --white_background  --lambda_normal_smooth 1.0
#python train.py -s ./data/chrome5 --iterations 50000 --uncertainty_from_iter 35000 --eval  --lambda_normal_smooth 1.0 --lambda_hybrid 0.2 --uncertainty_lr 0.001 
#python train.py -s ./data/disk --iterations 50000 --uncertainty_from_iter 35000 --eval --lambda_normal_smooth 1.0 --lambda_hybrid 0.2 --uncertainty_lr 0.001 
#python train.py -s ./data/car_gear --iterations 50000 --uncertainty_from_iter 35000 --eval --lambda_normal_smooth 1.0 --lambda_hybrid 0.2 --uncertainty_lr 0.001 
#python train.py -s ./data/realworld_scenario/car_segmented --iterations 50000 --uncertainty_from_iter 35000 --eval --lambda_normal_smooth 1.0 --lambda_hybrid 0.2 --uncertainty_lr 0.001 

#python train.py -s .//data/realworld_scenario/tv --iterations 50000 --uncertainty_from_iter 35000 --eval --lambda_normal_smooth 1.0 --lambda_hybrid 0.2 --uncertainty_lr 0.001 

#python train.py -s ./data/realworld_scenario/postcase1 --iterations 50000 --uncertainty_from_iter 35000 --eval --lambda_normal_smooth 1.0 --lambda_hybrid 0.2 --uncertainty_lr 0.001 


#python train.py -s ../3DGS-DR/data/ref_real/gardenspheres --eval --iterations 20000 --uncertainty_from_iter 15000 --indirect_from_iter 10000 --volume_render_until_iter 0 --initial 1 --init_until_iter 3000 --lambda_normal_smooth 0.45 -r 4 
#python train.py -s ../3DGS-DR/data/ref_real/toycar --eval --iterations 20000 --uncertainty_from_iter 15000 --indirect_from_iter 10000 --volume_render_until_iter 0  --initial 1 --init_until_iter 3000  -r 4
#python train.py -s ../3DGS-DR/data/ref_real/sedan --eval --iterations 20000 --uncertainty_from_iter 15000 --indirect_from_iter 10000 --volume_render_until_iter 0  --initial 1 --init_until_iter 3000  -r 4