"""The body an annotation app reduces to, driven by its `AnnotConfig`.

This module holds the data path: resolving the two parquet paths, reading the input,
building or resuming the output, and the guards that stand between a reshaped input
and a mislabelled gold standard. Rendering lives beside it.
"""

import logging
from collections.abc import Sequence
from pathlib import Path

import pandas as pd
import streamlit as st
from filelock import FileLock

from edscrib.config import AnnotConfig
from edscrib.io import build_output, data_path, read_data

MESSAGE_MISMATCH = (
    "The output was built on a different input: its document ids no longer match, "
    "in value or in order. Rebuild the input before annotating."
)

MESSAGE_LABELS = (
    "The documents are not labelled from zero without a gap, so a position in the run "
    "no longer designates the row it shows. Rebuild the data before annotating."
)

MESSAGE_COLUMNS = (
    "The data no longer carries every column the annotation declares. The detail is in "
    "the server log, which the deployment operator reads."
)

MESSAGE_UNAVAILABLE = (
    "The annotation data could not be opened. The detail is in the server log, which "
    "the deployment operator reads."
)

_logger = logging.getLogger(__name__)


def _stamp(path: str | Path) -> tuple[int, int]:
    """What the input file looks like from the outside, without opening it."""
    info = Path(path).stat()
    return info.st_mtime_ns, info.st_size


@st.cache_data(max_entries=1)
def _read_input(path: str | Path, stamp: tuple[int, int]) -> pd.DataFrame:
    """Read the input, keyed on the state of the file as well as on its path.

    `stamp` is never read. It is there to make a regenerated input a different entry:
    without it the first frame is served for the life of the server process, so the
    rebuild the guards below ask for takes effect only on a restart, while the app goes
    on rendering the message that asked for it. It carries no leading underscore, which
    would be the one spelling `st.cache_data` reads as "keep this out of the key".

    The path stays an argument rather than a closure, since the cache is global to the
    process and keyed on the qualified name: one app would otherwise serve another one
    its frame. One entry is kept, so a regenerated input evicts rather than doubles it.
    """
    return read_data(path)


def aligned(df_input: pd.DataFrame, df_output: pd.DataFrame, id_field: str) -> bool:
    """Whether the two frames designate the same documents, row for row.

    `Series.equals` compares the index alongside the values, so this also rejects the
    two frames drifting apart into non-contiguous labels together, which is what would
    put the cursor on a row label that no longer exists.
    """
    return df_input[id_field].equals(df_output[id_field])


def labelled_by_position(data: pd.DataFrame) -> bool:
    """Whether the rows are labelled from zero without a gap, as the cursor assumes.

    The cursor is a position and `save_notes` addresses a label, so the two designate
    the same document on this labelling alone. Under any other one a save lands on
    another document, or raises, or renormalizes the file and has the alignment guard
    stop the app on the next rerun, naming the input for what the app itself did.
    """
    return data.index.equals(pd.RangeIndex(len(data)))


def missing_fields(data: pd.DataFrame, fields: Sequence[str]) -> list[str]:
    """The declared columns the frame does not carry, in a stable order.

    The identifiers are what the alignment guard compares, and a reshape preserving
    them in value and in order passes it while dropping any other declared column. That
    one surfaces on the document the annotator happens to reach, at render or at export,
    where it is a raise inside a callback rather than a guard on the way in.
    """
    return sorted(set(fields) - set(data.columns))


def needs_write(on_disk: pd.DataFrame | None, note_fields: Sequence[str]) -> bool:
    """Whether the freshly built output has to reach the disk before annotating.

    The predicate reads the persisted frame rather than the one in memory: the columns
    `build_output` just added exist only in memory, and leaving them there would have
    the first save create them row by row and leave every other row null.

    The column check is what covers a second annotator arriving on an output another
    one has already started: their own columns are what is missing from it.
    """
    if on_disk is None:
        return True

    return not set(note_fields).issubset(on_disk.columns)


def write_output(data: pd.DataFrame, path: str | Path) -> None:
    """Write the output under the same lock `save_notes` takes, to serialise with it.

    The frame lands on a sibling path and is moved onto the target, as `save_notes`
    does: writing onto the target would leave a truncated parquet behind an interrupted
    write, at a path that exists, and every later start reads it and fails on it with
    no command of the app able to clear it.

    The lock is a singleton, so a caller already holding it for the same path re-enters
    it instead of deadlocking against itself: two distinct `FileLock` objects over one
    path block each other even inside a single process.
    """
    with FileLock(f"{path}.lock", is_singleton=True):
        staged = Path(f"{path}.tmp")
        data.to_parquet(staged)
        staged.replace(path)


def load_frames(config: AnnotConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read the input, build or resume the output, and gate on the guards.

    Returns the input frame, which carries the document text, and the output frame the
    app annotates. Both are labelled by position and aligned row for row once the
    guards have passed, which is what lets the rest of the app address a document by
    position.

    Reading the output, deciding whether it has to be written and writing it happen
    under one lock, the one `save_notes` takes. Deciding outside it would have a save
    landing between the read and the write, and the write would put the frame read
    before that save back on the disk.

    This is the boundary the reads are caught at. A missing file, an unreadable one, a
    truncated parquet and a missing column each raise naming the deployment path or the
    column, and `client.showErrorDetails` paints an uncaught raise into the browser of
    whoever is looking. The detail goes to the module log, one message goes to the page.
    The guards below stop the script through `ScriptControlException`, which descends
    from `BaseException` and so passes through the clause untouched.
    """
    try:
        paths = (config.work_dir, config.proj, config.split)
        input_path = data_path(*paths, config.input_suffix, config.data_suffix)
        output_path = data_path(*paths, config.output_suffix, config.data_suffix)

        df_input = _read_input(input_path, _stamp(input_path))

        with FileLock(f"{output_path}.lock", is_singleton=True):
            missing = missing_fields(df_input, (config.id_field, config.text_field))

            if missing:
                _logger.error("The input does not carry %s", missing)
                st.error(MESSAGE_COLUMNS)
                st.stop()

            df_output = build_output(
                df_input,
                output_path,
                config.text_field,
                config.persisted,
            )

            if not labelled_by_position(df_input) or not labelled_by_position(df_output):
                st.error(MESSAGE_LABELS)
                st.stop()

            if not aligned(df_input, df_output, config.id_field):
                st.error(MESSAGE_MISMATCH)
                st.stop()

            declared = (config.id_field, *config.export_fields, *config.table_fields)
            missing = missing_fields(df_output, declared)

            if missing:
                _logger.error("The output does not carry %s", missing)
                st.error(MESSAGE_COLUMNS)
                st.stop()

            on_disk = read_data(output_path) if Path(output_path).exists() else None

            if needs_write(on_disk, config.persisted):
                write_output(df_output, output_path)
    except Exception:
        _logger.exception("Annotation data unusable, holding the app back")
        st.error(MESSAGE_UNAVAILABLE)
        st.stop()

    return df_input, df_output
