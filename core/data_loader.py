import os

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
DEFAULT_TAXI_PATH = os.path.join(DATA_DIR, "taxi_with_health_info.csv")
DEFAULT_SERVER_PATH = os.path.join(DATA_DIR, "edge_server_locations.csv")


CORE_COLUMNS = ['taxi_id', 'date_time', 'latitude', 'longitude']


def load_data(file_path=None, sample_fraction=1.0, chunk_size=None, start_index=None, end_index=None):
    """
    Load taxi trajectory CSV, keeping only physical-movement columns.
    
    Parameters
    ----------
    file_path : str, optional
        Path to CSV file. Defaults to DEFAULT_TAXI_PATH.
    sample_fraction : float, optional
        Fraction of taxis to sample (0.0-1.0). Default 1.0 (all taxis).
    chunk_size : int, optional
        Truncate to first N rows.
    start_index : int, optional
        起始索引（包含）
    end_index : int, optional
        结束索引（不包含）
    
    如果同时指定 start_index 和 end_index，则截取 df[start_index:end_index]
    """
    if file_path is None:
        file_path = DEFAULT_TAXI_PATH
    print(f"Loading data from {file_path} ...")
    df = pd.read_csv(file_path)

    # Strip all non-physical columns (health / context features)
    keep = [c for c in CORE_COLUMNS if c in df.columns]
    dropped = [c for c in df.columns if c not in keep]
    if dropped:
        print(f"  Dropped {len(dropped)} non-physical columns: {dropped}")
    df = df[keep]

    if sample_fraction < 1.0:
        unique_taxis = df['taxi_id'].unique()
        sampled_taxis = np.random.choice(
            unique_taxis,
            size=int(len(unique_taxis) * sample_fraction),
            replace=False,
        )
        df = df[df['taxi_id'].isin(sampled_taxis)]
        print(f"  Sampled {len(sampled_taxis)} taxis.")

    df['date_time'] = pd.to_datetime(df['date_time'])
    df = df.sort_values(['taxi_id', 'date_time']).reset_index(drop=True)
    df = df.dropna(subset=['longitude', 'latitude'])

    # 新增：按索引范围截取（在 chunk_size 逻辑之前）
    if start_index is not None and end_index is not None:
        df = df.iloc[start_index:end_index].reset_index(drop=True)
        print(f"  Sliced data: [{start_index}:{end_index}], {len(df):,} records")

    if chunk_size is not None and chunk_size > 0:
        df = df.head(chunk_size)
        print(f"  Truncated to {chunk_size} rows.")

    print(f"  Final: {len(df):,} records, {df['taxi_id'].nunique():,} unique taxis.")
    return df
