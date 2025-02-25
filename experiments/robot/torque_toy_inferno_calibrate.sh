## Choose the node to run on
#PBS -l nodes=atlas2.driftslab.hib.no-0:ppn=10
## Name the analysis
#PBS -N InfernoJobNchains
## Choose queue
#PBS -q unlimited
## Concat output files
#PBS -j oe
## Array of jobs
#PBS -t 1-4

#nlist=(10 40 80 120 150 200 250 300 500 900 1200 1500 1800 2100 2400 2700 3000 3300 3600)
#nlist=(2 4 8 16 32 64 128 256 300 500 512 900 1024 1200 1500 1800 2048 2100 2400 2700 3000 4096)
#nlist=(250 1000 2000 3000 4000 5000)
nlist=(500 1000 2000 5000)
#nlist=(1 2 4 8 16)

script="toy_experiment/inferno/inferno_calibrate.R"
metadata="toy_experiment/inferno/metadata_calibrate.csv"
gridfile="toy_experiment/data/x1_x2_grid.csv"
nchains=10
ncores=10
ndata=1000
ncal=${nlist[${PBS_ARRAYID}-1]}
nsamples=1200
runLearn=TRUE #R variable TRUE FALSE

R2=3
k_red=7
k_blue=3
R1_min=6
scale=1
vary_a1=False #Python variable
vary_R2=False #Python variable
p_red=0.5

tag="r2_${R2}_kr${k_red}_kb${k_blue}_r1min${R1_min}_s${scale}_vary_r2_${vary_R2}_vary_a1_${vary_a1}_pRed_${p_red}"
folder="train_n_50000_${tag}"
calfile="toy_experiment/DNN/predictions/${folder}/val_n_5000_${tag}_predicted_SequentialNet_10ensembles_ndata-${ndata}.csv"
testfile="toy_experiment/DNN/predictions/${folder}/test_n_10000_${tag}_predicted_SequentialNet_10ensembles_ndata-${ndata}.csv"
gridfile="toy_experiment/DNN/predictions/${folder}/grid_${tag}_predicted_SequentialNet_10ensembles_ndata-${ndata}.csv"
infernolib="/disk/atlas2/users/agrefsru/inferno_renegade"

cd foundational-modeling
apptainer run --bind ${infernolib} apptainer/updated_env.sif ${script} ${ncal} ${nchains} ${ncores} ${nsamples} ${runLearn} ${testfile} ${metadata} ${calfile}
apptainer run --bind ${infernolib} apptainer/updated_env.sif ${script} ${ncal} ${nchains} ${ncores} ${nsamples} FALSE ${gridfile} ${metadata} ${calfile}
