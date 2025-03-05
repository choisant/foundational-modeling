## Choose the node to run on
#PBS -l nodes=atlas3.driftslab.hib.no-0
## Name the analysis
#PBS -N DNN ensemble
## Choose queue
#PBS -q unlimited
## Concat output files
#PBS -j oe
## -o bnn.out

. /home/agrefsru/.bashrc
cd foundational-modeling/toy_experiment/src
conda activate imcal
python train_bnn.py
