#python train.py -s data/ref_nerf/helmet --iterations 50000 --uncertainty_from_iter 45000 --eval --white_background  --lambda_normal_smooth 1.0 --lambda_hybrid 0.3 --uncertainty_lr 0.001
#python train.py -s data/ref_nerf/coffee --iterations 50000 --uncertainty_from_iter 45000 --eval --white_background  --lambda_normal_smooth 1.0 --lambda_hybrid 0.3 --uncertainty_lr 0.001
#python train.py -s data/ref_nerf/ball --iterations 50000 --uncertainty_from_iter 45000 --eval --white_background  --lambda_normal_smooth 1.0 --lambda_hybrid 0.3 --uncertainty_lr 0.001
#python train.py -s data/ref_nerf/car --iterations 50000 --uncertainty_from_iter 45000 --eval --white_background  --lambda_normal_smooth 1.0 --lambda_hybrid 0.3 --uncertainty_lr 0.001
#python train.py -s data/ref_nerf/teapot --iterations 50000 --uncertainty_from_iter 45000 --eval --white_background  --lambda_normal_smooth 1.0 --lambda_hybrid 0.3 --uncertainty_lr 0.001
#python train.py -s data/ref_nerf/toaster --iterations 50000 --uncertainty_from_iter 45000 --eval --white_background  --lambda_normal_smooth 1.0 --lambda_hybrid 0.3 --uncertainty_lr 0.001


#python train.py -s ../3DGS-DR/data/ref_real/gardenspheres --eval --iterations 20000 --uncertainty_from_iter 17000 --indirect_from_iter 10000 --volume_render_until_iter 0 --initial 1 --init_until_iter 3000 --lambda_normal_smooth 0.45 -r 4 --lambda_hybrid 0.3 --uncertainty_lr 0.001
#python train.py -s ../3DGS-DR/data/ref_real/toycar --eval --iterations 20000 --uncertainty_from_iter 17000 --indirect_from_iter 10000 --volume_render_until_iter 0  --initial 1 --init_until_iter 3000  -r 4 --lambda_hybrid 0.3 --uncertainty_lr 0.001
#python train.py -s ../3DGS-DR/data/ref_real/sedan --eval --iterations 20000 --uncertainty_from_iter 17000 --indirect_from_iter 10000 --volume_render_until_iter 0  --initial 1 --init_until_iter 3000  -r 4 --lambda_hybrid 0.3 --uncertainty_lr 0.001





# python train.py -s ../3DGS-DR/data/ref_real/gardenspheres --eval --iterations 28000 --uncertainty_from_iter 20000 --indirect_from_iter 10000 --volume_render_until_iter 0 --initial 1 --init_until_iter 3000 --lambda_normal_smooth 0.45 -r 4 --lambda_hybrid 0.5 --uncertainty_lr 0.001
#python train.py -s ../3DGS-DR/data/ref_real/toycar --eval --iterations 28000 --uncertainty_from_iter 20000 --indirect_from_iter 10000 --volume_render_until_iter 0  --initial 1  -r 4 --lambda_hybrid 0.5 --uncertainty_lr 0.001
#python train.py -s ../3DGS-DR/data/ref_real/sedan --eval --iterations 28000 --uncertainty_from_iter 20000 --indirect_from_iter 10000 --volume_render_until_iter 0  --initial 1 --init_until_iter 3000  -r 4 --lambda_hybrid 0.5 --uncertainty_lr 0.001


python train.py -s ../3DGS-DR/data/ref_real/toycar --eval --iterations 50000
python train.py -s ../3DGS-DR/data/ref_real/gardenspheres --eval --iterations 50000
python train.py -s ../3DGS-DR/data/ref_real/sedan --eval --iterations 50000

# python train.py -s data/ref_nerf/helmet --iterations 50000 --uncertainty_from_iter 25000 --eval --white_background  --lambda_normal_smooth 1.0 --lambda_hybrid 0.2 --uncertainty_lr 0.001
python train.py -s data/ref_nerf/coffee --iterations 50000 --uncertainty_from_iter 25000 --eval --white_background  --lambda_normal_smooth 1.0 --lambda_hybrid 0.2 --uncertainty_lr 0.001
python train.py -s data/ref_nerf/ball --iterations 50000 --uncertainty_from_iter 25000 --eval --white_background  --lambda_normal_smooth 1.0 --lambda_hybrid 0.2 --uncertainty_lr 0.001
python train.py -s data/ref_nerf/car --iterations 50000 --uncertainty_from_iter 25000 --eval --white_background  --lambda_normal_smooth 1.0 --lambda_hybrid 0.2 --uncertainty_lr 0.001
python train.py -s data/ref_nerf/teapot --iterations 50000 --uncertainty_from_iter 25000 --eval --white_background  --lambda_normal_smooth 1.0 --lambda_hybrid 0.2 --uncertainty_lr 0.001
python train.py -s data/ref_nerf/toaster --iterations 50000 --uncertainty_from_iter 25000 --eval --white_background  --lambda_normal_smooth 1.0 --lambda_hybrid 0.2 --uncertainty_lr 0.001