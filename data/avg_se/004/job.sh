#!/bin/bash
#SBATCH -J zigzag             # ��ҵ���� 
#SBATCH -p defq              # �ύ�� Ĭ�ϵ�defq ����
#SBATCH -N 1                # ʹ��1���ڵ�
#SBATCH --ntasks-per-node=8 # ÿ���ڵ㿪��90������
#SBATCH --cpus-per-task=1    # ÿ������ռ��һ�� cpu ����
#SBATCH -t 50000:00            # �����������ʱ���� 500 ����
#SBATCH --mem=4G           #����100G�ڴ�

module load cuda11.1/toolkit/11.1.1
module load lammps/2022.06.23



mpirun lmp_oneapi -in 1.in
