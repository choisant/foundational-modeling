## Choose the node to run on
#PBS -l nodes=atlas3.driftslab.hib.no-0
## Name the analysis
#PBS -N hyperparams
## Choose queue
#PBS -q unlimited
## Concat output files
#PBS -j oe


. /home/agrefsru/.bashrc
cd foundational-modeling/experiments/gamma_n_dims/DNN
conda activate imcal
python DNN_evidential.py
python MCD.py
python DNN_ensemble.py
python DNN_CL.py
