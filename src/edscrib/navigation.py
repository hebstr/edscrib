"""Moving through the documents of an annotation run, and saving what was noted."""

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st
from filelock import FileLock


def previous_index(index: int) -> int:
    return max(index - 1, 0)


def next_index(index: int, nrow: int) -> int:
    return min(index + 1, max(nrow - 1, 0))


def save_notes(path: str | Path, index: int, notes: Mapping[str, Any]) -> None:
    """Write one document's notes into the output parquet, under a file lock.

    The frame is re-read inside the lock instead of taken from the caller, so an
    annotation written by another process between two reruns survives. The row is
    addressed by index label, and the index itself is not persisted.

    An unknown label or an unknown field raises, since assigning one would enlarge
    the frame rather than fail, and the enlarged frame would reach the parquet.

    The frame lands on a sibling path and is moved onto the target, so an interrupted
    write leaves the accumulated run readable instead of truncating it, and a reader
    taking no lock sees either the old file or the new one.
    """
    with FileLock(f"{path}.lock"):
        fresh = pd.read_parquet(path)

        if index not in fresh.index:
            raise KeyError(f"{index!r} is not a row label of {path}")

        unknown = sorted(set(notes) - set(fresh.columns))

        if unknown:
            raise KeyError(f"{unknown} are not fields of {path}")

        for name, value in notes.items():
            fresh.loc[index, name] = value

        staged = Path(f"{path}.tmp")
        fresh.to_parquet(staged, index=False)
        staged.replace(path)


def navigation(
    path: str | Path,
    nrow: int,
    note_fields: Sequence[str],
    *,
    columns: Sequence[float],
    can_save: bool,
    save_label: str,
    key_button_back: str = "button-backward",
    key_button_save: str = "button-save",
    key_button_forward: str = "button-forward",
) -> None:
    """Render the backward / save / forward row and wire it to the session cursor.

    The caller owns the cursor: `doc_index` and `save_count` are read and written in
    `st.session_state` and must already be there. Their absence is raised on render
    rather than left to surface from inside a button callback, mid-annotation.
    """
    state = st.session_state
    missing = [key for key in ("doc_index", "save_count") if key not in state]

    if missing:
        raise KeyError(f"{missing} must be in st.session_state before navigation()")

    def go_back() -> None:
        state.doc_index = previous_index(state.doc_index)

    def go_forward() -> None:
        state.doc_index = next_index(state.doc_index, nrow)

    def save() -> None:
        if not can_save:
            return

        save_notes(path, state.doc_index, {name: state[name] for name in note_fields})
        state.doc_index = next_index(state.doc_index, nrow)
        state.save_count += 1

    col_backward, col_save, col_forward = st.columns(columns)

    with col_backward:
        st.button(
            label="◀",
            width="stretch",
            on_click=go_back,
            key=key_button_back,
        )

    with col_save:
        st.button(
            label=save_label,
            width="stretch",
            on_click=save,
            icon=":material/save:",
            disabled=not can_save,
            key=key_button_save,
            type="primary" if can_save else "secondary",
        )

    with col_forward:
        st.button(
            label="▶",
            width="stretch",
            on_click=go_forward,
            key=key_button_forward,
        )
