## Choose the node to run on
#PBS -l nodes=atlas2.driftslab.hib.no-0
## Name the analysis
#PBS -N NN_models
## Choose queue
#PBS -q unlimited
## Concat output files
#PBS -j oe
#PBS -t 3

gridSearch=false
activation="tanh"

#Dataset A
#shape1=2
#shape2=6
#scale1=5
#scale2=3

#Dataset B
shape1=2
shape2=4
scale1=3
scale2=3

#Dataset C
#shape1=6
#shape2=6
#scale1=3
#scale2=3


. /home/agrefsru/.bashrc
cd /disk/atlas3/users/agrefsru/foundational-modeling/experiments/gamma_n_dims/DNN
conda activate imcal

if $gridSearch
then
    if [[ ${PBS_ARRAYID} -eq 1 ]]
    then
        python DNN_evidential.py --shape1 ${shape1} --shape2 ${shape2} --scale1 ${scale1} --scale2 ${scale2} --activation ${activation} -g
    elif [[ ${PBS_ARRAYID} -eq 2 ]]
    then
        python MCD.py --shape1 ${shape1} --shape2 ${shape2} --scale1 ${scale1} --scale2 ${scale2} --activation ${activation} -g
    elif [[ ${PBS_ARRAYID} -eq 3 ]]
    then
        python DNN_ensemble.py --shape1 ${shape1} --shape2 ${shape2} --scale1 ${scale1} --scale2 ${scale2} --activation ${activation} -g
    else [[ ${PBS_ARRAYID} -eq 4 ]]
        python DNN_CL.py --shape1 ${shape1} --shape2 ${shape2} --scale1 ${scale1} --scale2 ${scale2} --activation ${activation} -g
    fi
else
    if [[ ${PBS_ARRAYID} -eq 1 ]]
    then
        python DNN_evidential.py --shape1 ${shape1} --shape2 ${shape2} --scale1 ${scale1} --scale2 ${scale2} --activation ${activation}
    elif [[ ${PBS_ARRAYID} -eq 2 ]]
    then
        python MCD.py --shape1 ${shape1} --shape2 ${shape2} --scale1 ${scale1} --scale2 ${scale2} --activation ${activation}
    elif [[ ${PBS_ARRAYID} -eq 3 ]]
    then
        python DNN_ensemble.py --shape1 ${shape1} --shape2 ${shape2} --scale1 ${scale1} --scale2 ${scale2} --activation ${activation}
    else [[ ${PBS_ARRAYID} -eq 4 ]]
        python DNN_CL.py --shape1 ${shape1} --shape2 ${shape2} --scale1 ${scale1} --scale2 ${scale2} --activation ${activation}
    fi
fi
