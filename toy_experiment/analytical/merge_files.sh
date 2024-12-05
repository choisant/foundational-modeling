#!/usr/bin/env bash

nX_MC=100
nr1_MC=50
N_JOBS=25
specs="kr7_kb3_vary_a1_True"

sed -n 1p jobarrays/analytical_solution_x1_x2_grid_${specs}_nxMC_${nX_MC}_nr1MC_${nr1_MC}_jobnr1.csv > analytical_solution_x1_x2_grid_${specs}_nxMC_${nX_MC}_nr1MC_${nr1_MC}.csv

for i in $(seq 1 ${N_JOBS});
do
    if [ -f "jobarrays/analytical_solution_x1_x2_grid_${specs}_nxMC_${nX_MC}_nr1MC_${nr1_MC}_jobnr${i}.csv" ]; then
        sed 1d jobarrays/analytical_solution_x1_x2_grid_${specs}_nxMC_${nX_MC}_nr1MC_${nr1_MC}_jobnr${i}.csv >> analytical_solution_x1_x2_grid_${specs}_nxMC_${nX_MC}_nr1MC_${nr1_MC}.csv
    fi
done