## Choose the node to run on
#PBS -l nodes=atlas3.driftslab.hib.no-0:ppn=10
## Name the analysis
#PBS -N InfernoJobCalibrate
## Choose queue
#PBS -q unlimited
## Concat output files
#PBS -j oe
## Array of jobs
#PBS -t 1-4

#nlist=(10 40 80 120 150 200 250 300 500 900 1200 1500 1800 2100 2400 2700 3000 3300 3600)
#nlist=(2 4 8 16 32 64 128 256 300 500 512 900 1024 1200 1500 1800 2048 2100 2400 2700 3000 4096)
nlist=(900 2100 2700 3600)

cd foundational-modeling

apptainer run apptainer/updated_env.sif toy_experiment/inferno/inferno_calibrate.R 1000 20 10 1200 toy_experiment/inferno/metadata_DNN.csv toy_experiment/DNN/predictions/val_n_5000_kr7_kg3_s1_vary_a1_False_predicted_SequentialNet_ndata-${nlist[${PBS_ARRAYID}-1]}.csv toy_experiment/data/x1_x2_grid.csv
apptainer run apptainer/updated_env.sif toy_experiment/inferno/inferno_test_calibrate.R 1000 20 10 1200 toy_experiment/inferno/metadata_DNN.csv toy_experiment/DNN/predictions/val_n_5000_kr7_kg3_s1_vary_a1_False_predicted_SequentialNet_ndata-${nlist[${PBS_ARRAYID}-1]}.csv toy_experiment/DNN/predictions/val_n_5000_kr7_kg3_s1_vary_a1_False_predicted_SequentialNet_ndata-${nlist[${PBS_ARRAYID}-1]}.csv
apptainer run apptainer/updated_env.sif toy_experiment/inferno/inferno_test_calibrate.R 1000 20 10 1200 toy_experiment/inferno/metadata_DNN.csv toy_experiment/DNN/predictions/val_n_5000_kr7_kg3_s1_vary_a1_False_predicted_SequentialNet_ndata-${nlist[${PBS_ARRAYID}-1]}.csv toy_experiment/DNN/predictions/grid_predicted_SequentialNet_ndata-${nlist[${PBS_ARRAYID}-1]}.csv
