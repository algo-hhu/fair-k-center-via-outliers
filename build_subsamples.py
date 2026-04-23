import pandas as pd
import os
import numpy as np

from subsampling import subsampling_uniform

datasets = {
    "bank": {"file": "bank_processed", "subsample": 1000},
    "census": {"file": "census_processed", "subsample": 600},
    "diabetes": {"file": "diabetes_processed", "subsample": 1000},
    "income": {"file": "acs_income_2018_1-Year_person_CA_features_processed", "subsample": 1000}
}

sampling_methods = {"uniform": subsampling_uniform}

INITIAL_SEED    = 4711
NORMALIZED_DATA = True

def subsample_dataframe(df: pd.DataFrame, num_samples: int, seed: int, sampling_method: callable) -> np.ndarray:
    samples = []
    for _ in range(num_samples):
        X, _ = sampling_method(df.to_numpy(), df.to_numpy(), num_samples=num_samples, seed=seed)  
        seed=seed+1
        samples.append(X)
    
    return samples


def subsample_dataset(ds_name: str, sampling_type: str, seed: int, normalized_data: bool):
    sampling_method = sampling_methods.get(sampling_type)
    if not sampling_method:
        raise ValueError(f"Unknown sampling method: {sampling_type}")
    
    suffix = "-normalized" if normalized_data else ""
    ds = datasets[ds_name]
    df = pd.read_csv(f"instances/{ds['file']}{suffix}.csv")
    n_samples = len(df) // ds['subsample']
    
    samples = subsample_dataframe(df, n_samples, seed, sampling_method)
    
    subdir = "normalized" if normalized_data else "raw"
    os.makedirs(f"instances/{subdir}/{ds_name}/subsamples/{sampling_type}", exist_ok=True)
    for idx, X in enumerate(samples):
        outfile_path = f"instances/{subdir}/{ds_name}/subsamples/{sampling_type}/{ds['file']}_{idx:05}.csv"
        header = ",".join(df.columns.tolist())
        np.savetxt(outfile_path, X, delimiter=",", header=header, comments="")

    return n_samples


def cut_dataframe_into_subsamples(df: pd.DataFrame, subsample_size: int, seed: int) -> list[pd.DataFrame]:
    n_subsamples = len(df) // subsample_size
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    
    return [df.iloc[i * subsample_size:(i + 1) * subsample_size] for i in range(n_subsamples)]


def cut_dataset_into_subsamples(ds_name: str, seed: int, normalized_data: bool):
    ds = datasets[ds_name]
    suffix = "-normalized" if normalized_data else ""
    df = pd.read_csv(f"instances/{ds['file']}{suffix}.csv")
    
    samples = cut_dataframe_into_subsamples(df, ds['subsample'], seed)

    subdir = "normalized" if normalized_data else "raw"
    os.makedirs(f"instances/{subdir}/{ds_name}/subsamples/cut", exist_ok=True)
    for idx, X in enumerate(samples):
        X.to_csv(f"instances/{subdir}/{ds_name}/subsamples/cut/{ds['file']}_{idx:05}.csv", index=False)


def main():
    seed = INITIAL_SEED
    for ds_name in datasets.keys():
        for sampling_type in sampling_methods.keys():
            print(f"Buildings subsamples for dataset {ds_name} with {sampling_type} sampling...")
            n_samples = subsample_dataset(ds_name, sampling_type, seed, NORMALIZED_DATA)
            seed += n_samples

        print(f"Buildings subsamples for dataset {ds_name} with cut sampling...")
        cut_dataset_into_subsamples(ds_name, seed, NORMALIZED_DATA)
        seed += 1

if __name__ == "__main__":    
    main()


