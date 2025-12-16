
#python train.py -s ../../../3DGS-DR/data/GlossyReal/bear -m output/learnable_test/bear --eval --iterations 50000 --initial 1 --init_until_iter 3000  -r 1  --train_on_all #--lambda_hybrid 0.1 --uncertainty_lr 0.001 --uncertainty_from_iter 10000
#python train.py -s ../../../3DGS-DR/data/GlossyReal/coral -m output/learnable_test/coral --eval --iterations 50000 --initial 1 --init_until_iter 3000  -r 1 --train_on_all #--lambda_hybrid 0.1 --uncertainty_lr 0.001 --uncertainty_from_iter 10000
python train.py -s ../../../3DGS-DR/data/GlossyReal/maneki -m output/maneki --eval --iterations 40000 --initial 1 --init_until_iter 3000  -r 1 --use_parallax_correction #--train_on_all #--lambda_hybrid 0.1 --uncertainty_lr 0.001 --uncertainty_from_iter 10000
python train.py -s ../../../3DGS-DR/data/GlossyReal/bunny -m output/bunny --eval --iterations 40000 --initial 1 --init_until_iter 3000  -r 1 --use_parallax_correction #--train_on_all #--lambda_hybrid 0.1 --uncertainty_lr 0.001 --uncertainty_from_iter 10000

python train.py -s ../../../3DGS-DR/data/GlossyReal/bear -m output/bear --eval --iterations 40000 --initial 1 --init_until_iter 3000  -r 1  --use_parallax_correction #--train_on_all #--lambda_hybrid 0.1 --uncertainty_lr 0.001 --uncertainty_from_iter 10000
python train.py -s ../../../3DGS-DR/data/GlossyReal/coral -m output/coral --eval --iterations 40000 --initial 1 --init_until_iter 3000  -r 1 --use_parallax_correction #--train_on_all #--lambda_hybrid 0.1 --uncertainty_lr 0.001 --uncertainty_from_iter 10000
python train.py -s ../../../3DGS-DR/data/GlossyReal/vase -m output/vase --eval --iterations 40000 --initial 1 --init_until_iter 3000  -r 1 --lambda_diffuse_metal 0.05 --train_on_all #--lambda_hybrid 0.1 --uncertainty_lr 0.001 --uncertainty_from_iter 10000
