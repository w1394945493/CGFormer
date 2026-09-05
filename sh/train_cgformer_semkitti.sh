CUDA_VISIBLE_DEVICES=0,1,2,3 \
PYTHONPATH="$(pwd):$(pwd)/packages/DFA3D" \
python /vepfs-mlp2/c20250502/haoce/wangyushen/CGFormer/main.py \
      --config_path /vepfs-mlp2/c20250502/haoce/wangyushen/CGFormer/configs/customs/CGFormer-Efficient-Swin-SemanticKITTI.py \
      --log_folder /c20250502/wangyushen/Outputs/cgformer/cgformer/sem_kitti/train \
      --seed 7240 \
      --log_every_n_steps 50