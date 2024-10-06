import pandas as pd
from datasets import Dataset

def _return_calibration_dataset(dataset_pairs, num_samples):
    _dataset = []
    for dataset_ in dataset_pairs:
            ds = pd.read_parquet(dataset_)
            dataset = Dataset.from_pandas(ds)
            dataset = dataset.shuffle(seed=42)
            

            source, _ = dataset.column_names[0].split('-') # e.g. source: cs target: en    
            lines_source = []
            # lines_target = []
            
            for idx in range(num_samples):
                sample = dataset[idx][dataset.column_names[0]]
                source_sentence = sample[source] 
                # target_sentence = sample[target]

                lines_source.append(source_sentence)
                # lines_target.append(target_sentence)

            _dataset.extend(lines_source)
    return _dataset

def load_WMT22Testdataset(mode, upper_bound_num_samples=512):
   
    dataset_pairs = ["hf://datasets/haoranxu/WMT22-Test/cs-en/test-00000-of-00001-1a83a591805d9178.parquet", 
    "hf://datasets/haoranxu/WMT22-Test/de-en/test-00000-of-00001-c03dcec47c23d6ca.parquet", 
    "hf://datasets/haoranxu/WMT22-Test/en-cs/test-00000-of-00001-b92f389a2a10e4b5.parquet", 
    "hf://datasets/haoranxu/WMT22-Test/en-de/test-00000-of-00001-c470e1e53ed73302.parquet", 
    "hf://datasets/haoranxu/WMT22-Test/en-is/test-00000-of-00001-872ab78ba9548351.parquet", 
    "hf://datasets/haoranxu/WMT22-Test/en-ru/test-00000-of-00001-889b8af39e8c83c4.parquet", 
    "hf://datasets/haoranxu/WMT22-Test/en-zh/test-00000-of-00001-6b3b7f42ead58b33.parquet", 
    "hf://datasets/haoranxu/WMT22-Test/is-en/test-00000-of-00001-bb3b8280f4b7ff31.parquet", 
    "hf://datasets/haoranxu/WMT22-Test/ru-en/test-00000-of-00001-4455a1b04d42177e.parquet", 
    "hf://datasets/haoranxu/WMT22-Test/zh-en/test-00000-of-00001-a8c846c3e121c2f6.parquet"]
    
    if "mode_1" in mode:
        num_samples = upper_bound_num_samples // len(dataset_pairs) + 1
        _dataset = _return_calibration_dataset(dataset_pairs, num_samples)
        
    elif "mode_2" in mode:
        num_samples = upper_bound_num_samples
        _dataset = _return_calibration_dataset(dataset_pairs, num_samples)

    elif "mode_3" in mode:
        num_samples = upper_bound_num_samples//2
        _dataset = _return_calibration_dataset(dataset_pairs, num_samples)
    elif "mode_4" in mode:
        num_samples = upper_bound_num_samples//4
        _dataset = _return_calibration_dataset(dataset_pairs, num_samples)
    elif "mode_5" in mode:
        num_samples = upper_bound_num_samples//8
        _dataset = _return_calibration_dataset(dataset_pairs, num_samples)
    else:
        raise ValueError(f"Mode: {mode} not found")
    return _dataset
if __name__ == "__main__":
    datset = load_WMT22Testdataset("mode_3", 512)