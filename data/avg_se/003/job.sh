#!/bin/bash
#SBATCH -J zigzag             # 作业名是 
#SBATCH -p defq              # 提交到 默认的defq 队列
#SBATCH -N 1                # 使用1个节点
#SBATCH --ntasks-per-node=8 # 每个节点开启90个进程
#SBATCH --cpus-per-task=1    # 每个进程占用一个 cpu 核心
#SBATCH -t 50000:00            # 任务最大运行时间是 500 分钟
#SBATCH --mem=4G           #申请100G内存

module load cuda11.1/toolkit/11.1.1
module load lammps/2022.06.23



mpirun lmp_oneapi -in 1.in
