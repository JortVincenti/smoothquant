#!/bin/bash 


#list of strings to evaluate
QUANT_LIST=("None" "smooth" "naive")

for QUANT in "${QUANT_LIST[@]}"
do
    echo "Evaluating $QUANT"
    if [ "$QUANT" == "None" ]; then
        sbatch --job-name "eval-fakequant-${QUANT}" smoothquant.job --model "ALMA-7B/" --src "cs" --tgt "en" --dtype "float16" --beam 5
    else
        sbatch --job-name "eval-fakequant-${QUANT}" smoothquant.job --model "ALMA-7B/" --src "cs" --tgt "en" --dtype "float16" --beam 5 --quant $QUANT
    fi
done