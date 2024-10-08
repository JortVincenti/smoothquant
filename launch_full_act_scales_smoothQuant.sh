#!/bin/bash 


#list of strings to evaluate
QUANT_MODE=("mode_1" "mode_2" "mode_3" "mode_4" "mode_5")

for QUANT in "${QUANT_MODE[@]}"
do
    echo "Evaluating $QUANT"
    sbatch --job-name "full-act-scales-${QUANT}" full_act_scales_smoothQuant.job --model "ALMA-7B/" --num-samples 512 --seq-len 512 --mode $QUANT
done