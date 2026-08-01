"""Moving through the documents of an annotation run, and saving what was noted."""

import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import streamlit as st
from filelock import FileLock

from edscrib.io import read_data

MESSAGE_CURSOR = (
    "The annotation is not set up to move through its documents. The detail is in the "
    "server log, which the deployment operator reads."
)

MESSAGE_SAVE = (
    "The annotation was not saved, and the document it was written on is unchanged. "
    "Reload the page before annotating further. The detail is in the server log, which "
    "the deployment operator reads."
)

_logger = logging.getLogger(__name__)


def previous_index(index: int) -> int:
    return max(index - 1, 0)


def next_index(index: int, nrow: int) -> int:
    return min(index + 1, max(nrow - 1, 0))


def save_notes(
    path: str | Path,
    index: int,
    notes: Mapping[str, Any],
    *,
    id_field: str,
    expected_id: Any,
) -> None:
    """Write one document's notes into the output parquet, under a file lock.

    The frame is re-read inside the lock instead of taken from the caller, so an
    annotation written by another process between two reruns survives. The row is
    addressed by index label, and the index itself is not persisted.

    That re-read is why the lock is not a singleton, where `write_output` takes the same
    one as one because `load_frames` calls it from inside its own hold. Nothing holds it
    above this call, and a caller that did would already carry the frame this goes back
    to the disk for: `filelock` refuses two distinct objects over one path in one thread
    rather than blocking on them, the display boundary above turns that refusal into one
    message, and nothing is written. Re-entering instead would land the note under a
    frame the outer hold is about to put back, losing it with no trace.

    It goes through the package's own reader, which is what keeps the identifier this
    compares below on the same dtype as the one the caller read it from. Two readers
    drifting apart would not merely disagree: a coercion to float collapses two integer
    identifiers past the mantissa onto one value, so the comparison would match on two
    different documents and the guard would wave through exactly what it stands for.
    That reader is never the cached one, here as at every other freshness-sensitive
    call site: the caller wraps its own cache around it, and this read has to reach the
    disk to see what another process just wrote.

    An unknown label or an unknown field raises, since assigning one would enlarge
    the frame rather than fail, and the enlarged frame would reach the parquet.

    A label carried more than once raises too, whether or not it is the one addressed.
    On the addressed one, an assignment reaches every row holding it, so one document's
    notes land on several. Anywhere else, the write succeeds and the index is not
    persisted, so the labels come back contiguous on the next read: the frame the app's
    own labelling guard would have refused becomes one it accepts, and nothing records
    that it was ever inconsistent.

    The row is also checked to still hold the document the caller was shown, which is
    what `expected_id` names. Every other guard on the identity of a row runs when the
    app reads the frames, and a callback fires ahead of the run that would re-run them,
    so the file the annotator was looking at and the file this writes to are only ever
    the same one by assumption. A run extended with a document inserted ahead of the
    cursor keeps its labels contiguous and its two frames aligned, so it passes each of
    those guards on every later run while the position designates another document: the
    annotation lands on the wrong patient and no later read can tell. Re-reading the
    identifier under the lock is the only place the assumption is observable.

    That check reads an identifier as an identity, so one carried by two documents is
    refused ahead of it, here and not only where the app reads the frames. Against the
    very extension above, an inserted document sharing its identifier with the one it
    displaces leaves the comparison holding a value against itself, and it accepts the
    write onto either document. The load-time guard cannot stand in for this one: it
    observed the frames a rerun boundary earlier, which is the whole reason `expected_id`
    exists, and the labelling check beside it is re-run here for the same reason.

    The frame lands on a sibling path and is moved onto the target, so an interrupted
    write leaves the accumulated run readable instead of truncating it, and a reader
    taking no lock sees either the old file or the new one.

    The staging path is cleared whichever way the write ends, so a failed one leaves no
    partial parquet beside the run it failed to update. The next successful save would
    move its own staging file over it, but nothing guarantees a next one, and a
    half-written sibling of the gold standard is what gets copied by mistake.
    """
    with FileLock(f"{path}.lock"):
        fresh = read_data(path)

        if fresh.index.has_duplicates:
            raise ValueError(f"{path} carries a row label more than once")

        if index not in fresh.index:
            raise KeyError(f"{index!r} is not a row label of {path}")

        unknown = sorted((set(notes) | {id_field}) - set(fresh.columns))

        if unknown:
            raise KeyError(f"{unknown} are not fields of {path}")

        if fresh[id_field].duplicated().any():
            raise ValueError(f"{path} carries a document identifier more than once")

        found = fresh.loc[index, id_field]

        if found != expected_id:
            raise ValueError(
                f"row {index} holds document {found!r}, not {expected_id!r}",
            )

        for name, value in notes.items():
            fresh.loc[index, name] = value

        staged = Path(f"{path}.tmp")

        try:
            fresh.to_parquet(staged, index=False)
            staged.replace(path)
        finally:
            staged.unlink(missing_ok=True)


def navigation(
    path: str | Path,
    nrow: int,
    note_fields: tuple[str, ...],
    *,
    columns: tuple[float, float, float],
    can_save: bool,
    save_label: str,
    id_field: str,
    doc_id: Any,
    back_label: str = "◀",
    forward_label: str = "▶",
    key_button_back: str = "button-backward",
    key_button_save: str = "button-save",
    key_button_forward: str = "button-forward",
) -> None:
    """Render the backward / save / forward row and wire it to the session cursor.

    The caller owns the cursor: `doc_index` and `save_count` are read and written in
    `st.session_state` and must already be there. Their absence stops the app on render
    rather than surfacing from inside a button callback, mid-annotation. It is the
    consumer's own setup that is wrong, so it goes to the log the operator reads and
    not to the page: a raise here is painted into the browser with its traceback, which
    names the path the package is installed under, to whoever is looking.

    The two carry different signals and neither substitutes for the other. `doc_index`
    is which document is shown, and it moves on every button of the row, so it is what
    a per-document widget key folds in. `save_count` counts writes that reached the
    disk: it stands still on the arrows and advances only behind a save, which is what
    makes it the signal for anything displaying the output rather than one document.
    Keying a note widget on it would hold one document's answer across a move to the
    next, and offer it for saving there.

    `columns` carries the three relative widths the row is laid out on, one per button.
    The annotation says three, which is what a consumer running a type checker hears
    before deploying; the same count is checked here for one that computes the widths or
    runs none, since `st.columns` returns as many containers as it is given and the row
    would otherwise fail on unpacking them, painting the traceback into the browser.

    The three button labels are the caller's, since a label is copy and the language it
    is written in belongs to the deployment. The arrows default to the glyphs alone,
    which is what a sighted annotator reads and what a screen reader announces by their
    Unicode name, as "black left-pointing triangle". Streamlit offers no seam for an
    accessible name of its own: `help` renders a tooltip carrying `aria-describedby`,
    which describes the button beside its name rather than replacing it, so a deployment
    whose annotators use one passes text here instead.

    `doc_id` is the identifier of the document this render shows, which the save carries
    down to be checked against the file it is about to write. It is read here rather
    than inside the callback, so what is verified is the document the annotator was
    actually looking at and not whatever the file holds at that position by then.

    The save is the boundary its own failures are caught at. Every way it refuses names
    the deployment path or a document, and `client.showErrorDetails` paints an uncaught
    raise from a callback into the browser of whoever is looking, so the detail goes to
    the module log and one message goes to the page. The clause is broad because a
    narrow one only holds what it was written against, where the point is that nothing
    at all escapes: the cursor stays where it is and the annotator is told to reload,
    which is what re-runs the guards that name what actually changed.
    """
    state = st.session_state
    missing = [key for key in ("doc_index", "save_count") if key not in state]

    if missing:
        _logger.error("The app did not seed %s in st.session_state", missing)
        st.error(MESSAGE_CURSOR)
        st.stop()

    if len(columns) != 3:
        _logger.error("The row is laid out on three widths, not %d", len(columns))
        st.error(MESSAGE_CURSOR)
        st.stop()

    def go_back() -> None:
        state.doc_index = previous_index(state.doc_index)

    def go_forward() -> None:
        state.doc_index = next_index(state.doc_index, nrow)

    def save() -> None:
        if not can_save:
            return

        try:
            save_notes(
                path,
                state.doc_index,
                {name: state[name] for name in note_fields},
                id_field=id_field,
                expected_id=doc_id,
            )
        except Exception:
            _logger.exception("The annotation could not be saved")
            st.error(MESSAGE_SAVE)
            return

        state.doc_index = next_index(state.doc_index, nrow)
        state.save_count += 1

    col_backward, col_save, col_forward = st.columns(columns)

    with col_backward:
        st.button(
            label=back_label,
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
            label=forward_label,
            width="stretch",
            on_click=go_forward,
            key=key_button_forward,
        )
