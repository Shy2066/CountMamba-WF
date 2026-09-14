# 手动执行：
cd /home/shy/Project/CountMamba-WF/CountMamba

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