## Choose the node to run on
#PBS -l nodes=atlas3.driftslab.hib.no-0:ppn=20
## Name the analysis
#PBS -N InfernoJobNdata
## Choose queue
#PBS -q unlimited
## Concat output files
#PBS -j oe
## Array of jobs
#PBS -t 1-4

#nlist=(10 40 80 120 150 200 250 300 500 900 1200 1500 1800 2100 2400 2700 3000 3300 3600)
#nlist=(2 4 8 16 32 64 128 256 300 500 512 900 1024 1200 1500 1800 2048 2100 2400 2700 3000 4096)
nlist=(250 2100 3600 4500)

script="toy_experiment/inferno/inferno.R"
metadata="toy_experiment/inferno/metadata_x1_x2.csv"
testfile="toy_experiment/data/x1_x2_grid.csv"
nchains=20
ncores=20
nsamples=1200
vary_a1=False
kr=7
kb=3
runLearn=TRUE
trainfile="toy_experiment/data/train_n_50000_kr${kr}_kb${kb}_s1_vary_a1_${vary_a1}.csv"
valfile="toy_experiment/data/val_n_5000_kr${kr}_kb${kb}_s1_vary_a1_${vary_a1}.csv"
infernolib="/disk/atlas2/users/agrefsru/inferno_renegade"

cd foundational-modeling
apptainer run --bind ${infernolib} apptainer/updated_env.sif ${script} ${nlist[${PBS_ARRAYID}-1]} ${nchains} ${ncores} ${nsamples} ${runLearn} ${valfile} ${metadata} ${trainfile}
apptainer run --bind ${infernolib} apptainer/updated_env.sif ${script} ${nlist[${PBS_ARRAYID}-1]} ${nchains} ${ncores} ${nsamples} FALSE ${testfile} ${metadata} ${trainfile}
