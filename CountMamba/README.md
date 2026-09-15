# 手动执行：
cd /home/shy/Project/CountMamba-WF/CountMamba

python convert_to_npz.py --dataset tor_fiber_acup3
python convert_to_npz.py --dataset tor_starlink_acup3

for dataset in tor_fiber_front-t1 tor_starlink_front-t1 tor_fiber_front-t2 tor_starlink_front-t2 tor_fiber_wtfpad tor_starlink_wtfpad
do
  python dataset_split.py --dataset ${dataset}
done

## 可选：先检查任务矩阵，不启动训练
/home/shy/anaconda3/envs/Countmamba/bin/python \
  run_domain_experiments.py --dry-run --seeds 2024 --gpu 0

## 正式单种子实验
/home/shy/anaconda3/envs/Countmamba/bin/python \
  run_domain_experiments.py --seeds 2024 --gpu 0

# 三种子实验：
/home/shy/anaconda3/envs/Countmamba/bin/python \
  run_domain_experiments.py --seeds 42 2024 2025 --gpu 0

# 结果会写入：
/home/shy/Project/CountMamba-WF/logs/CountMamba_domain

cd /home/shy/Project/CountMamba-WF/CountMamba

/home/shy/anaconda3/envs/Countmamba/bin/python \
  run_domain_experiments.py \
  --datasets tor_fiber_wtfpad tor_starlink_wtfpad \
  --datasets tor_fiber_front-t1 tor_starlink_front-t1 \
  --datasets tor_fiber_front-t2 tor_starlink_front-t2 \
  --datasets tor_fiber_regulator tor_starlink_regulator \
  --seeds 42 2024 2025 \
  --gpu 2