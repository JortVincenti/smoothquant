import torch
import os

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    LlamaTokenizer
)
import argparse

from smoothquant.alma_calibration import get_act_scales


def build_model_and_tokenizer(model_name):
    #tokenizer = AutoTokenizer.from_pretrained(model_name, model_max_length=512)
    tokenizer = LlamaTokenizer.from_pretrained(model_name)
    tokenizer.pad_token_id = tokenizer.eos_token_id

    kwargs = {"torch_dtype": torch.float16, "device_map": "sequential"}
    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    return model, tokenizer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-name", type=str, default="facebook/opt-1.3b", help="model name"
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="full_scale_activation_models/",
        help="dir to save the act scales",
    )
   
    parser.add_argument("--num-samples", type=int, default=512)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument(
        "--mode",
        type=str,
        default="mode_1",
        help="Controls the model type for act scales. Options: mode_1, mode_2, mode_3, mode_4, mode_5",
    )
    args = parser.parse_args()
    return args


@torch.no_grad()
def main():
    args = parse_args()
    model, tokenizer = build_model_and_tokenizer(args.model_name)

    # if not os.path.exists(args.dataset_path):
    #     print(f"Cannot find the dataset at {args.dataset_path}")
    #     print("Please download the Pile dataset and put the validation set at the path")
    #     print(
    #         "You can download the validation dataset of the Pile at https://huggingface.co/datasets/mit-han-lab/pile-val-backup/resolve/main/val.jsonl.zst"
    #     )
    #     raise FileNotFoundError
    
    act_scales = get_act_scales(
        model, tokenizer, args.mode, args.num_samples, args.seq_len
    )
    dir_path = os.path.join(args.output_path, f"{args.model_name}")
    os.makedirs(os.path.dirname(dir_path), exist_ok=True)
    torch.save(act_scales, os.path.join(dir_path,f"{args.mode}.pth"))


if __name__ == "__main__":
    main()
