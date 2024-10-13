# SmoothQuant \& Q-LoRA: Model Compression for Machine Translation in Large Language Models

## Findings

In this work, we explored the application of various compression techniques to ALMA. Our study focused on five methods: GPTQ, Q-LoRA, SmoothQuant, Wanda, and DSnot. The results show that quantization techniques offer reductions in memory usage with minimal impact on translation quality. Pruning methods like Wanda provided faster inference times but at the cost of translation performance. The findings highlight the trade-offs between model efficiency and performance, demonstrating that compression methods can make large models like ALMA more accessible for deployment without sacrificing translation quality.

## Installation

### Clone the repo
```bash
git clone https://github.com/JortVincenti/smoothquant.git
```

### Install Environment

```bash
sbatch install_env.job
```
### Install Git LFS from source (needed to store the weights in the sratch directory)
```bash
source install_lfs.sh
```


### Download ALMA-7B weights
```bash
sbatch download_alma_weights.job
```

## Running SmoothQuant

### Getting the Activation Scales
```bash
source launch_full_act_scales_smoothQuant.sh
```

### Running SmoothQuant Experiments with Q-LoRA like quantization for ALMA-7B
```bash
source launch_eval_smoothquant_all_lang_pairs.sh 

```

### Running Q-LoRA like quantization for ALMA-7B
```bash
source launch_eval_ALMA_LLM_int8_all_lang_pairs.sh
```

### Running baseline ALMA-7B
```bash
source launch_eval_plain_ALMA_all_lang_pairs.sh
```
