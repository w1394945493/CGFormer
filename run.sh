
# ========================================#
# 火山服务器
# 生成semantic kitti所需深度图 tmux 153

cd /vepfs-mlp2/c20250502/haoce/wangyushen/CGFormer/preprocess
. /root/miniconda3/bin/activate
conda activate /vepfs-mlp2/c20250502/haoce/conda_env/wys_temp_2
bash image2depth_semantickitti.sh

# 生成sscbench kitti 360所需深度图 tmux 154
cd /vepfs-mlp2/c20250502/haoce/wangyushen/CGFormer/preprocess
. /root/miniconda3/bin/activate
conda activate /vepfs-mlp2/c20250502/haoce/conda_env/wys_temp_2
bash image2depth_kitti360.sh

cd /vepfs-mlp2/c20250502/haoce/wangyushen/CGFormer/packages/DFA3D
python3 setup.py build_ext --inplace
# 允许覆盖注册
grep -n "CONV_LAYERS.register_module" \
  packages/mmdetection3d/mmdet3d/ops/spconv/conv.py
sed -i.bak \
  's/@CONV_LAYERS\.register_module()/@CONV_LAYERS.register_module(force=True)/g' \
  packages/mmdetection3d/mmdet3d/ops/spconv/conv.py

grep -n "NORM_LAYERS.register_module" \
  packages/mmdetection3d/mmdet3d/ops/norm.py
sed -i.bak \
  "s/@NORM_LAYERS\.register_module(name='\([^']*\)')/@NORM_LAYERS.register_module(name='\1', force=True)/g" \
  packages/mmdetection3d/mmdet3d/ops/norm.py

cd /vepfs-mlp2/c20250502/haoce/wangyushen/CGFormer/packages/mmdetection3d
python3 setup.py build_ext --inplace
pip install . --no-build-isolation
python -m pip install . --no-deps --no-build-isolation --force-reinstall

(/vepfs-mlp2/c20250502/haoce/conda_env/wys_temp_2) root@di-20260307214017-v7hl8:/vepfs-mlp2/c20250502/haoce/wangyushen/CGFormer# grep -RniE \
"from mmengine|import mmengine" \
main.py mmdet3d_plugin packages \
--include="*.py"


python -m pip install \
  --no-deps \
  '/vepfs-mlp2/c20250502/haoce/wangyushen/CGFormer/data/natten-0.15.0+torch210cu121-cp310-cp310-linux_x86_64.whl'

python -m pip install "mmengine==0.10.7" \
  -i https://pypi.org/simple
  
# evaluate 
CUDA_VISIBLE_DEVICES=0 \
python /vepfs-mlp2/c20250502/haoce/wangyushen/CGFormer/main.py \
  --eval \
  --ckpt_path /c20250502/wangyushen/Weights/cgformer/CGFormer-Efficient-Swin-SemanticKITTI.ckpt \
  --config_path /vepfs-mlp2/c20250502/haoce/wangyushen/CGFormer/configs/customs/CGFormer-Efficient-Swin-SemanticKITTI.py \
  --log_folder /vepfs-mlp2/c20250502/haoce/wangyushen/Outputs/gcformer/CGFormer-Efficient-Swin-SemanticKITTI-eval \
  --seed 7240 \
  --log_every_n_steps 50

# train
CUDA_VISIBLE_DEVICES=0,1,2,3 \
PYTHONPATH="$(pwd):$(pwd)/packages/DFA3D" \
python /vepfs-mlp2/c20250502/haoce/wangyushen/CGFormer/main.py \
  --config_path /vepfs-mlp2/c20250502/haoce/wangyushen/CGFormer/configs/customs/CGFormer-Efficient-Swin-SemanticKITTI.py \
  --log_folder /c20250502/wangyushen/Outputs/cgformer/cgformer/sem_kitti/train \
  --seed 7240 \
  --log_every_n_steps 50

cd /vepfs-mlp2/c20250502/haoce/wangyushen/CGFormer
. /root/miniconda3/bin/activate
conda activate /vepfs-mlp2/c20250502/haoce/conda_env/wys_temp_2
bash sh/train_cgformer_semkitti.sh