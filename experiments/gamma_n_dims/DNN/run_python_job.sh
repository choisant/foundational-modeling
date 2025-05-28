## Choose the node to run on
#PBS -l nodes=atlas3.driftslab.hib.no-0
## Name the analysis
#PBS -N DNN job
## Choose queue
#PBS -q unlimited
## Concat output files
#PBS -j oe
## -o bnn.out

. /home/agrefsru/.bashrc
cd foundational-modeling/experiments/gamma_n_dims/DNN
conda activate imcal
python DNN_CL.py
