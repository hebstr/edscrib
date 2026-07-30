"""The body an annotation app reduces to, driven by its `AnnotConfig`.

This module holds the data path: resolving the two parquet paths, reading the input,
building or resuming the output, and the two guards that stand between a reshaped
input and a mislabelled gold standard. Rendering lives beside it.
"""

from collections.abc import Sequence
from pathlib import Path

import pandas as pd
import streamlit as st
from filelock import FileLock

from edscrib.config import AnnotConfig
from edscrib.io import build_output, data_path, has_note_values, read_data

MESSAGE_MISMATCH = (
    "The output was built on a different input: its document ids no longer match, "
    "in value or in order. Rebuild the input before annotating."
)

_read_input = st.cache_data(read_data)


def aligned(df_input: pd.DataFrame, df_output: pd.DataFrame, id_field: str) -> bool:
    """Whether the two frames designate the same documents, row for row.

    `Series.equals` compares the index alongside the values, so this also rejects the
    two frames drifting apart into non-contiguous labels together, which is what would
    put the cursor on a row label that no longer exists.
    """
    return df_input[id_field].equals(df_output[id_field])


def needs_write(
    on_disk: pd.DataFrame | None,
    note_fields: Sequence[str],
    estimate_prefix: str,
    estimate_values: Sequence[str],
) -> bool:
    """Whether the freshly built output has to reach the disk before annotating.

    The predicate reads the persisted frame rather than the one in memory: the columns
    `build_output` just added exist only in memory, and leaving them there would have
    the first save create them row by row and leave every other row null.

    The column check is what covers a second annotator arriving on an output another
    one has already started, since the prefix scan below sees every annotator's work.
    """
    if on_disk is None:
        return True

    if not set(note_fields).issubset(on_disk.columns):
        return True

    return not has_note_values(on_disk, estimate_prefix, estimate_values)


def write_output(data: pd.DataFrame, path: str | Path) -> None:
    """Write the output under the same lock `save_notes` takes, to serialise with it."""
    with FileLock(f"{path}.lock"):
        data.to_parquet(path)


def load_frames(config: AnnotConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read the input, build or resume the output, and gate on the two guards.

    Returns the input frame, which carries the document text, and the output frame the
    app annotates. Both are positionally aligned once the first guard has passed, which
    is what lets the rest of the app address a document by position.
    """
    paths = (config.work_dir, config.proj, config.split)
    input_path = data_path(*paths, config.input_suffix, config.data_suffix)
    output_path = data_path(*paths, config.output_suffix, config.data_suffix)

    df_input = _read_input(input_path)
    df_output = build_output(
        df_input,
        output_path,
        config.text_field,
        config.persisted,
        config.estimate_prefix,
        config.estimate_values,
    )

    if not aligned(df_input, df_output, config.id_field):
        st.error(MESSAGE_MISMATCH)
        st.stop()

    on_disk = read_data(output_path) if Path(output_path).exists() else None

    if needs_write(
        on_disk,
        config.persisted,
        config.estimate_prefix,
        config.estimate_values,
    ):
        write_output(df_output, output_path)

    return df_input, df_output
