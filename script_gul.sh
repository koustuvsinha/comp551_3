#!/bin/bash
#PBS -N Comp551
#PBS -A eeg-641-aa
#PBS -l walltime=03:00:00
#PBS -l nodes=1:ppn=1:gpu=1
cd "${PBS_O_WORKDIR}"
module load Python/2.7.10 CUDA/7.5.18 cuDNN/5.0-ga
source /home/koustuvs/project/bin/activate
cd "/home/koustuvs/comp551_3"
THEANO_FLAGS='floatX=float32,device=gpu' python runner_lasagne.py
