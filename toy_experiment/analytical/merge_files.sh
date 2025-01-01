#!/usr/bin/env bash

nX_MC=20
nr1_MC=20
N_JOBS=25
specs="r2_7_kr15_kb5_r1min0_s1_vary_r2_False_vary_a1_False_pRed_0.5"

sed -n 1p jobarrays/analytical_solution_x1_x2_grid_${specs}_nxMC_${nX_MC}_nr1MC_${nr1_MC}_jobnr1.csv > analytical_solution_x1_x2_grid_${specs}_nxMC_${nX_MC}_nr1MC_${nr1_MC}.csv

for i in $(seq 1 ${N_JOBS});
do
    if [ -f "jobarrays/analytical_solution_x1_x2_grid_${specs}_nxMC_${nX_MC}_nr1MC_${nr1_MC}_jobnr${i}.csv" ]; then
        sed 1d jobarrays/analytical_solution_x1_x2_grid_${specs}_nxMC_${nX_MC}_nr1MC_${nr1_MC}_jobnr${i}.csv >> analytical_solution_x1_x2_grid_${specs}_nxMC_${nX_MC}_nr1MC_${nr1_MC}.csv
    fi
done