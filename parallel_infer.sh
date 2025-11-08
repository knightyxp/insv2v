#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2,3

torchrun --nproc_per_node=4 /home/xianyang/Data/code/instruct-video-to-video/insv2v_parallel_json.py \
  --test_json /scratch3/yan204/yxp/VideoX_Fun/data/test_json/4tasks_rem_add_swap_local-style_test.json \
  --output_dir ./results/insv2v_on_our_bench_test \
  --ckpt_path /instruct-v2v-5b.pth \
  --config_path /home/xianyang/Data/code/instruct-video-to-video/configs/instruct_v2v.yaml \
  --seed 0 \
  --num_frames 33 \
  --text_cfg 7.5 \
  --video_cfg 1.8 \
  --frames_in_batch 16 \
  --num_ref_frames 4 \
  --output_fps 8