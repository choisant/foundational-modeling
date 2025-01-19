## Choose the node to run on
#PBS -l nodes=atlas3.driftslab.hib.no-0
## Name the analysis
#PBS -N InfernoJobNchains
## Choose queue
#PBS -q unlimited
## Concat output files
#PBS -j oe
## Array of jobs
#PBS -t 1-5

#nlist=(10 40 80 120 150 200 250 300 500 900 1200 1500 1800 2100 2400 2700 3000 3300 3600)
#nlist=(2 4 8 16 32 64 128 256 300 500 512 900 1024 1200 1500 1800 2048 2100 2400 2700 3000 4096)
nlist=(250 500 1000 2000 5000)
#nlist=(25 50 100 125 150 200 225)

script="toy_experiment/inferno/inferno.R"
metadata="toy_experiment/inferno/metadata_x1_x2.csv"
gridfile="toy_experiment/data/x1_x2_grid.csv"
nchains=10
ncores=10
#ndata=2000
nsamples=2400

R2=3
k_red=7
k_blue=3
R1_min=6
scale=1
vary_a1=False #Python variable
vary_R2=False #Python variable
p_red=0.5

tag="r2_${R2}_kr${k_red}_kb${k_blue}_r1min${R1_min}_s${scale}_vary_r2_${vary_R2}_vary_a1_${vary_a1}_pRed_${p_red}"
runLearn=TRUE #R variable TRUE FALSE
trainfile="toy_experiment/data/train_n_50000_${tag}.csv"
testfile="toy_experiment/data/test_n_10000_${tag}.csv"
infernolib="/disk/atlas2/users/agrefsru/inferno_renegade"

cd foundational-modeling
apptainer run --bind ${infernolib} apptainer/updated_env.sif ${script} ${nlist[${PBS_ARRAYID}-1]} ${nchains} ${ncores} ${nsamples} ${runLearn} ${testfile} ${metadata} ${trainfile}
apptainer run --bind ${infernolib} apptainer/updated_env.sif ${script} ${nlist[${PBS_ARRAYID}-1]} ${nchains} ${ncores} ${nsamples} FALSE ${gridfile} ${metadata} ${trainfile}
