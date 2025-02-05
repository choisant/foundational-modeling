## Choose the node to run on
#PBS -l nodes=atlas2.driftslab.hib.no-0:ppn=10
## Name the analysis
#PBS -N InfernoJobNchains
## Choose queue
#PBS -q unlimited
## Concat output files
#PBS -j oe
## Array of jobs
#PBS -t 1

#nlist=(10 40 80 120 150 200 250 300 500 900 1200 1500 1800 2100 2400 2700 3000 3300 3600)
#nlist=(2 4 8 16 32 64 128 256 300 500 512 900 1024 1200 1500 1800 2048 2100 2400 2700 3000 4096)
nlist=(250 1000 2000 3000 4000 5000)
#nlist=(750 1000 1250 1500 1750 2000)
#nlist=(1 2 4 8 16)

script="toy_experiment/inferno/inferno_calibrate_truth.R"
metadata="toy_experiment/inferno/metadata_calibrate_truth.csv"

nchains=10
ncores=10
#ndata=2000
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
#calfile="toy_experiment/analytical/results/analytical_solution_cal_n_2000_${tag}_nxMC_200_nr1MC_100.csv"
calfile="toy_experiment/analytical/results/analytical_solution_test_n_10000_${tag}_nxMC_200_nr1MC_100.csv"
testfile="toy_experiment/analytical/results/analytical_solution_test_n_10000_${tag}_nxMC_200_nr1MC_100.csv"
gridfile="toy_experiment/analytical/results/analytical_solution_x1_x2_grid_${tag}_nxMC_200_nr1MC_100.csv"
infernolib="/disk/atlas2/users/agrefsru/inferno_renegade"

cd foundational-modeling
apptainer run --bind ${infernolib} apptainer/updated_env.sif ${script} ${nlist[${PBS_ARRAYID}-1]} ${nchains} ${ncores} ${nsamples} ${runLearn} ${testfile} ${metadata} ${calfile}
apptainer run --bind ${infernolib} apptainer/updated_env.sif ${script} ${nlist[${PBS_ARRAYID}-1]} ${nchains} ${ncores} ${nsamples} FALSE ${gridfile} ${metadata} ${calfile}
