# 生成semantic kitti所需深度图 tmux 153
conda activate /vepfs-mlp2/c20250502/haoce/conda_env/wys_temp_2
cd /vepfs-mlp2/c20250502/haoce/wangyushen/CGFormer/preprocess
bash image2depth_semantickitti.sh

tmux attach -t 153

# 生成sscbench kitti 360所需深度图 tmux 153
conda activate /vepfs-mlp2/c20250502/haoce/conda_env/wys_temp_2
cd /vepfs-mlp2/c20250502/haoce/wangyushen/CGFormer/preprocess
bash image2depth_kitti360.sh