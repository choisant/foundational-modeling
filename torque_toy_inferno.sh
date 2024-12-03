## Choose the node to run on
#PBS -l nodes=atlas3.driftslab.hib.no-0:ppn=5
## Name the analysis
#PBS -N InfernoJobTest
## Choose queue
#PBS -q unlimited
## Concat output files
#PBS -j oe
## Array of jobs
#PBS -t 1-3

#nlist=(10 40 80 120 150 200 250 300 500 900 1200 1500 1800 2100 2400 2700 3000 3300 3600)
#nlist=(2 4 8 16 32 64 128 256 300 500 512 900 1024 1200 1500 1800 2048 2100 2400 2700 3000 4096)
nlist=(250 2100 3600)

cd foundational-modeling
#apptainer run env_test.sif ${nlist[${PBS_ARRAYID}-1]} 20 5 1200 metadata.csv data/train_n_5000_kr7_kg3_s1_vary_a1_False.csv data/val_n_5000_kr7_kg3_s1_vary_a1_False.csv
apptainer run apptainer/updated_env.sif toy_experiment/inferno/inferno_learn.R ${nlist[${PBS_ARRAYID}-1]} 20 5 800 toy_experiment/inferno/metadata_r_a_x.csv toy_experiment/data/train_n_5000_kr9_kb3_s1_vary_a1_False.csv toy_experiment/data/x1_x2_grid.csv
