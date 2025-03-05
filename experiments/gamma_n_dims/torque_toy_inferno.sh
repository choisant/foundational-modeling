## Choose the node to run on
#PBS -l nodes=atlas3.driftslab.hib.no-0:ppn=5
## Name the analysis
#PBS -N InfernoJobNchains
## Choose queue
#PBS -q unlimited
## Concat output files
#PBS -j oe
## Array of jobs
#PBS -t 1

nlist=(10000)

script="experiments/gamma_n_dims/inferno/inferno.R"
metadata="experiments/gamma_n_dims/inferno/metadata_x1_x2.csv"

nchains=5
ncores=5
ndata=${nlist[${PBS_ARRAYID}-1]}
nsamples=1200

tag="k_2_d2_shapes[2,6]_scales[5,3]_pc[0.5,0.5]"
runLearn=TRUE #R variable TRUE FALSE
trainfile="experiments/gamma_n_dims/data/train_n_50000_${tag}.csv"
testfile="experiments/gamma_n_dims/data/test_n_10000_${tag}.csv"
gridfile="experiments/gamma_n_dims/data/grid_x1_x2_10000_${tag}.csv"

cd foundational-modeling
apptainer run apptainer/robot_inferno.sif ${script} ${ndata} ${nchains} ${ncores} ${nsamples} ${runLearn} ${testfile} ${metadata} ${trainfile}
apptainer run apptainer/robot_inferno.sif ${script} ${ndata} ${nchains} ${ncores} ${nsamples} FALSE ${gridfile} ${metadata} ${trainfile}
