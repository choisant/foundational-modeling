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
#PBS -t 1-10

#start=$((400 * $(( ${PBS_ARRAYID}-1 )) ))
#stop=$((400*${PBS_ARRAYID}))
N_JOBS=10
N_DATA=10000
N_PER_JOB=$((${N_DATA}/${N_JOBS}))
start=$((${N_PER_JOB} * $(( ${PBS_ARRAYID}-1 )) ))
stop=$((${N_PER_JOB}*${PBS_ARRAYID}))
nr1_MC=4000

R2=3
k_red=7
k_blue=3
R1_min=6
scale=1
vary_a1=False #Python variable
vary_R2=False #Python variable
p_red=0.5

tag="r2_${R2}_kr${k_red}_kb${k_blue}_r1min${R1_min}_s${scale}_vary_r2_${vary_R2}_vary_a1_${vary_a1}_pRed_${p_red}"
#testfile="../data/cal_n_${N_DATA}_${tag}.csv"
gridfile="../data/x1_x2_grid.csv"

. /home/agrefsru/.bashrc
cd foundational-modeling/toy_experiment/analytical
conda activate imcal
python analytical.py -f ${gridfile} -s "jobarrays" --job_nr ${PBS_ARRAYID} --i_start $start --i_stop $stop --n_r1_mc ${nr1_MC} --R2 ${R2} --R1_min ${R1_min} --kr ${k_red} --kb ${k_blue}
