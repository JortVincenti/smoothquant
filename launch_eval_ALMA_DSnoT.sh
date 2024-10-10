#!/bin/bash 

LANG_PAIR=("hf://datasets/haoranxu/WMT22-Test/cs-en/test-00000-of-00001-1a83a591805d9178.parquet")

LANG=$(echo "$LANG_PAIR" | awk -F 'hf://datasets/haoranxu/WMT22-Test/' '{print $2}' | awk -F '/' '{print $1}')
SRC_LANG=$(echo "$LANG" | awk -F '-' '{print $1}')
TGT_LANG=$(echo "$LANG" | awk -F '-' '{print $2}')

sbatch --job-name "eval-ALMA-DSnoT-${LANG}" smoothquant.job --tokenizer "DSnoT/" --model "DSnoT/" --src "${SRC_LANG}" --data_path "${LANG_PAIR}" --tgt "${TGT_LANG}" --dtype "float16" --beam 5