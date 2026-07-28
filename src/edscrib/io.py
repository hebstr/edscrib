"""Reading and writing the parquet files an annotation app works on."""

from collections.abc import Sequence
from pathlib import Path

import pandas as pd

EMPTY = ""


def data_path(
    work_dir: str | Path,
    proj: str,
    split: str,
    stem: str,
    suffix: str = "",
    ext: str = "parquet",
) -> Path:
    filename = f"{split}_{proj}_annot_data_{stem}{suffix}.{ext}"
    return Path(work_dir) / "data" / split / filename


def read_data(path: str | Path) -> pd.DataFrame:
    """Read an annotation parquet.

    The caller caches this one under `st.cache_data`, so the path has to stay an
    argument: closing over it would key every app's cache on the same qualified name.
    """
    return pd.read_parquet(path)


def has_note_values(data: pd.DataFrame, prefix: str, values: Sequence[str]) -> bool:
    cols = [col for col in data.columns if col.startswith(prefix)]
    return bool(data[cols].isin(values).any(axis=None))


def build_output(
    df_input: pd.DataFrame,
    path: str | Path,
    text_field: str,
    note_fields: Sequence[str],
    estimate_prefix: str,
    estimate_values: Sequence[str],
) -> pd.DataFrame:
    """Resume an annotation output, or start one from the input.

    An output holding no estimate value at all is treated as unstarted and rebuilt, so
    that an input reshaped between two runs does not freeze the app on a stale frame.
    """
    data = read_data(path) if Path(path).exists() else None

    if data is None or not has_note_values(data, estimate_prefix, estimate_values):
        data = df_input.drop([text_field], axis=1)

    for name in note_fields:
        if name not in data.columns:
            data.loc[:, name] = EMPTY

    return data
