#!/usr/bin/env bash

nX_MC=200
nr1_MC=100
N_JOBS=25

R2=3
k_red=7
k_blue=3
R1_min=6
scale=1
vary_a1=False #Python variable
vary_R2=False #Python variable
p_red=0.5

tag="r2_${R2}_kr${k_red}_kb${k_blue}_r1min${R1_min}_s${scale}_vary_r2_${vary_R2}_vary_a1_${vary_a1}_pRed_${p_red}"
file="cal_n_2000"
#file="data/x1_x2_grid"

sed -n 1p jobarrays/analytical_solution_${file}__${tag}_nxMC_${nX_MC}_nr1MC_${nr1_MC}_jobnr1.csv > analytical_solution_${file}_${tag}_nxMC_${nX_MC}_nr1MC_${nr1_MC}.csv

for i in $(seq 1 ${N_JOBS});
do
    if [ -f "jobarrays/analytical_solution_${file}__${tag}_nxMC_${nX_MC}_nr1MC_${nr1_MC}_jobnr${i}.csv" ]; then
        sed 1d jobarrays/analytical_solution_${file}__${tag}_nxMC_${nX_MC}_nr1MC_${nr1_MC}_jobnr${i}.csv >> analytical_solution_${file}_${tag}_nxMC_${nX_MC}_nr1MC_${nr1_MC}.csv
    fi
done