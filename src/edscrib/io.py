"""Reading and writing the parquet files an annotation app works on."""

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


def build_output(
    df_input: pd.DataFrame,
    path: str | Path,
    text_field: str,
    note_fields: tuple[str, ...],
) -> pd.DataFrame:
    """Resume an annotation output, or start one from the input.

    An output is resumed as soon as it exists. A comment carries as much work as a
    radio answer, so a frame holding one and not the other is under way, and starting
    over on it would discard it.

    Resuming freezes every column outside the annotation on what the input carried at
    the first build, in ids, in column names and in values alike. An input regenerated
    between two runs therefore reaches the caller's guards, which stop the app on each
    of the three rather than opening a fresh output over the accumulated one.
    """
    data = read_data(path) if Path(path).exists() else None

    if data is None:
        data = df_input.drop([text_field], axis=1)

    for name in note_fields:
        if name not in data.columns:
            data.loc[:, name] = EMPTY

    return data
