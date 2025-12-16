#python train.py -s ../../marigold/pcc_monkey -m output/pcc_monkey_learnale --eval --iterations 50000 --initial 1 --init_until_iter 3000  -r 1 --use_parallax_correction
#python train.py -s ../../marigold/pcc_monkey -m output/pcc_monkey_wopcc --eval --iterations 50000 --initial 1 --init_until_iter 3000  -r 1 --envmap_max_res 128
python train.py -s ../../marigold/pcc_pot5_150 -m output/pcc_pot5_freeze --eval --iterations 50000 --initial 1 --init_until_iter 3000  -r 1 --use_parallax_correction
python train.py -s ../../marigold/pcc_pot5_150 -m output/pcc_pot5_wopcc --eval --iterations 50000 --initial 1 --init_until_iter 3000  -r 1 --envmap_max_res 128
