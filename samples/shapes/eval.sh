NUM=0
#CUDA_VISIBLE_DEVICES=$NUM python eval_rgb.py
#CUDA_VISIBLE_DEVICES=$NUM python eval_depth.py
#CUDA_VISIBLE_DEVICES=$NUM python eval_combined.py
CUDA_VISIBLE_DEVICES=$NUM python eval_rgbd.py

