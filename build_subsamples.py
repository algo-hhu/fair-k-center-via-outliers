import pandas as pd

from subsampling import subsampling_uniform, subsampling_fair

datasets = {
    "bank": {"file": "bank_processed", "subsample": 1000},
    "census": {"file": "census_processed", "subsample": 600},
    "diabetes": {"file": "diabetes_processed", "subsample": 1000},
    "income": {"file": "acs_income_2018_1-Year", "subsample": 1000}
}

sampling_methods = {"fair": subsampling_fair, "uniform": subsampling_uniform}

INITIAL_SEED    = 4711
SAMPLING_METHOD = "uniform" # 'uniform' or 'fair'

def subsample_dataframe(df: pd.DataFrame, num_samples: int, seed: int, sampling_method: callable) -> pd.DataFrame:
    samples = []
    for _ in range(num_samples):
        X, _ = sampling_method(df, pd.DataFrame(), num_samples=num_samples, seed=seed)  
        seed=seed+1
        samples.append(X)
    
    return samples


def subsample_dataset(ds_name: str, sampling_type: str, seed: int):
    sampling_method = sampling_methods.get(sampling_type)
    if not sampling_method:
        raise ValueError(f"Unknown sampling method: {sampling_type}")
    
    ds = datasets[ds_name]
    df = pd.read_csv(f"instances/{ds['file']}_processed.csv")
    n_samples = len(df) // ds['subsample']
    
    samples = subsample_dataframe(df, n_samples, seed, sampling_method)
    
    for idx, X in enumerate(samples):
        X.write_csv(f"instances/subsamples/{ds_name}/{sampling_type}/{ds['file']}_processed_{idx:05}.csv", index=False)

    return n_samples


def cut_dataframe_into_subsamples(df: pd.DataFrame, subsample_size: int, seed: int) -> list[pd.DataFrame]:
    n_subsamples = len(df) // subsample_size
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    
    return [df.iloc[i * subsample_size:(i + 1) * subsample_size] for i in range(n_subsamples)]


def cut_dataset_into_subsamples(ds_name: str, seed: int):
    ds = datasets[ds_name]
    df = pd.read_csv(f"instances/{ds['file']}_processed.csv")
    
    samples = cut_dataframe_into_subsamples(df, ds['subsample'], seed)

    for idx, X in enumerate(samples):
        X.write_csv(f"instances/subsamples/{ds_name}/cut/{ds['file']}_processed_{idx:05}.csv", index=False)


def main():
    seed = INITIAL_SEED
    for ds_name in datasets.keys():
        for sampling_type in sampling_methods.keys():
            n_samples = subsample_dataset(ds_name, sampling_type, seed)
            seed += n_samples

        cut_dataset_into_subsamples(ds_name, seed)
        seed += 1

if __name__ == "__main__":    
    main()


