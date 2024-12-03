## Choose the node to run on
#PBS -l nodes=atlas3.driftslab.hib.no-0
## Name the analysis
#PBS -N MonteCarloComputations
## Choose queue
#PBS -q unlimited
## Concat output files
#PBS -j oe
## -o foundational-modeling/toy_experiment/analytical/jobarrays/MCoutput.out
## Array of jobs
#PBS -t 1-25

start=$((400 * $(( ${PBS_ARRAYID}-1 )) ))
stop=$((400*${PBS_ARRAYID}))
nX_MC=200
nr1_MC=100
N_JOBS=25

. /home/agrefsru/.bashrc
cd foundational-modeling/toy_experiment/analytical
conda activate imcal
python analytical.py -f "../data/x1_x2_grid.csv" -s "jobarrays" --job_nr ${PBS_ARRAYID} --i_start $start --i_stop $stop --n_x_mc ${nX_MC} --n_r1_mc ${nr1_MC}
