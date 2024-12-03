## Choose the node to run on
#PBS -l nodes=atlas3.driftslab.hib.no-0
## Name the analysis
#PBS -N MonteCarloComputations
## Choose queue
#PBS -q unlimited
## Concat output files
#PBS -j oe
## Array of jobs
#PBS -t 1-100

start=$((100 * $(( ${PBS_ARRAYID}-1 )) ))
stop=$((100*${PBS_ARRAYID}))

. /home/agrefsru/.bashrc
cd foundational-modeling/toy_experiment/analytical
conda activate imcal
python analytical.py -f "../data/x1_x2_grid.csv" -s "jobarrays" --job_nr ${PBS_ARRAYID} --i_start $start --i_stop $stop