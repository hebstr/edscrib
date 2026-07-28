"""Moving through the documents of an annotation run, and saving what was noted."""

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st
from filelock import FileLock


def previous_index(index: int) -> int:
    return index - 1 if index > 0 else index


def next_index(index: int, nrow: int) -> int:
    return index + 1 if index < nrow - 1 else index


def save_notes(path: str | Path, index: int, notes: Mapping[str, Any]) -> None:
    """Write one document's notes into the output parquet, under a file lock.

    The frame is re-read inside the lock instead of taken from the caller, so an
    annotation written by another process between two reruns survives. The row is
    addressed by index label, and the index itself is not persisted.
    """
    with FileLock(f"{path}.lock"):
        fresh = pd.read_parquet(path)

        for name, value in notes.items():
            fresh.loc[index, name] = value

        fresh.to_parquet(path, index=False)


def navigation(
    path: str | Path,
    nrow: int,
    note_fields: Sequence[str],
    *,
    columns: Sequence[float] | int,
    can_save: bool,
    save_label: str,
    key_button_back: str = "button-backward",
    key_button_save: str = "button-save",
    key_button_forward: str = "button-forward",
) -> None:
    """Render the backward / save / forward row and wire it to the session cursor."""
    state = st.session_state

    def go_back() -> None:
        state.doc_index = previous_index(state.doc_index)

    def go_forward() -> None:
        state.doc_index = next_index(state.doc_index, nrow)

    def save() -> None:
        save_notes(path, state.doc_index, {name: state[name] for name in note_fields})
        state.doc_index = next_index(state.doc_index, nrow)
        state.save_count += 1

    col_backward, col_save, col_forward = st.columns(columns)

    with col_backward:
        st.button(
            label="◀",
            use_container_width=True,
            on_click=go_back,
            key=key_button_back,
        )

    with col_save:
        st.button(
            label=save_label,
            use_container_width=True,
            on_click=save,
            icon=":material/save:",
            disabled=not can_save,
            key=key_button_save,
            type="primary" if can_save else "secondary",
        )

    with col_forward:
        st.button(
            label="▶",
            use_container_width=True,
            on_click=go_forward,
            key=key_button_forward,
        )
