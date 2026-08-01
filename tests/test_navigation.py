"""Cursor arithmetic and the save path, on synthetic frames and paths."""

import threading
from pathlib import Path

import pandas as pd
import pytest
from filelock import FileLock
from streamlit.testing.v1 import AppTest

from edscrib.messages import MESSAGE_CURSOR, MESSAGE_SAVE
from edscrib.navigation import (
    next_index,
    previous_index,
    save_notes,
)

ESTIMATE = "note_estimate_a"
COMMENT = "note_comment_a"
FIELDS = (ESTIMATE, COMMENT)
ROWS = 3


def make_output(rows=ROWS):
    return pd.DataFrame(
        {
            "n": list(range(1, rows + 1)),
            ESTIMATE: [""] * rows,
            COMMENT: [""] * rows,
        }
    )


@pytest.fixture
def output(tmp_path):
    path = tmp_path / "output.parquet"
    make_output().to_parquet(path, index=False)
    return path


def test_previous_index_stops_at_the_first_document():
    assert previous_index(2) == 1
    assert previous_index(0) == 0


def test_next_index_stops_at_the_last_document():
    assert next_index(0, ROWS) == 1
    assert next_index(ROWS - 1, ROWS) == ROWS - 1


def test_next_index_stays_put_on_an_empty_run():
    assert next_index(0, 0) == 0


def test_next_index_pulls_a_cursor_past_the_end_back_in():
    assert next_index(ROWS + 2, ROWS) == ROWS - 1


def test_previous_index_pulls_a_cursor_below_the_start_back_in():
    assert previous_index(-3) == 0


def test_save_notes_writes_the_named_row_only(output):
    save_notes(output, 1, {ESTIMATE: "oui", COMMENT: "c"}, id_field="n", expected_id=2)

    data = pd.read_parquet(output)

    assert data[ESTIMATE].tolist() == ["", "oui", ""]
    assert data[COMMENT].tolist() == ["", "c", ""]


def test_save_notes_leaves_the_fields_it_was_not_given_alone(output):
    save_notes(output, 0, {ESTIMATE: "oui"}, id_field="n", expected_id=1)

    data = pd.read_parquet(output)

    assert data[COMMENT].tolist() == [""] * ROWS
    assert data["n"].tolist() == [1, 2, 3]


def test_save_notes_does_not_clobber_a_previous_save(output):
    save_notes(output, 0, {ESTIMATE: "oui"}, id_field="n", expected_id=1)
    save_notes(output, 2, {ESTIMATE: "non"}, id_field="n", expected_id=3)

    assert pd.read_parquet(output)[ESTIMATE].tolist() == ["oui", "", "non"]


def test_save_notes_addresses_the_row_by_index_label(tmp_path):
    path = tmp_path / "shifted.parquet"
    make_output().set_index(pd.Index([10, 11, 12])).to_parquet(path)

    save_notes(path, 11, {ESTIMATE: "oui"}, id_field="n", expected_id=2)

    assert pd.read_parquet(path)[ESTIMATE].tolist() == ["", "oui", ""]


def test_save_notes_does_not_persist_the_index(tmp_path):
    path = tmp_path / "shifted.parquet"
    make_output().set_index(pd.Index([10, 11, 12])).to_parquet(path)

    save_notes(path, 11, {ESTIMATE: "oui"}, id_field="n", expected_id=2)

    assert pd.read_parquet(path).index.tolist() == list(range(ROWS))


def test_save_notes_refuses_a_row_that_does_not_exist(output):
    with pytest.raises(KeyError):
        save_notes(output, ROWS + 4, {ESTIMATE: "oui"}, id_field="n", expected_id=1)

    assert len(pd.read_parquet(output)) == ROWS


def test_save_notes_refuses_a_frame_whose_labels_are_not_unique(tmp_path):
    path = tmp_path / "duplicated.parquet"
    make_output().set_index(pd.Index([0, 1, 1])).to_parquet(path)

    with pytest.raises(ValueError, match="more than once"):
        save_notes(path, 1, {ESTIMATE: "oui"}, id_field="n", expected_id=2)

    assert pd.read_parquet(path)[ESTIMATE].tolist() == [""] * ROWS


def test_save_notes_refuses_a_duplicate_away_from_the_row_it_writes(tmp_path):
    path = tmp_path / "duplicated.parquet"
    make_output().set_index(pd.Index([0, 5, 5])).to_parquet(path)

    with pytest.raises(ValueError, match="more than once"):
        save_notes(path, 0, {ESTIMATE: "oui"}, id_field="n", expected_id=1)

    assert pd.read_parquet(path).index.tolist() == [0, 5, 5]


def test_save_notes_refuses_an_identifier_carried_by_two_documents(tmp_path):
    """Otherwise the check below compares a value against itself and accepts either row.

    The app's own guard on this observed the frames a rerun boundary earlier, which is
    the reason `expected_id` exists at all, so it is re-asked here as the labelling one
    already is.
    """
    path = tmp_path / "output.parquet"
    pd.DataFrame({"n": [0, 1, 1], ESTIMATE: [""] * 3, COMMENT: [""] * 3}).to_parquet(
        path, index=False
    )

    with pytest.raises(ValueError, match="identifier more than once"):
        save_notes(path, 1, {ESTIMATE: "oui"}, id_field="n", expected_id=1)

    assert pd.read_parquet(path)[ESTIMATE].tolist() == [""] * 3


def test_save_notes_refuses_a_row_that_no_longer_holds_the_document(tmp_path):
    path = tmp_path / "output.parquet"
    pd.DataFrame({"n": [0, 1, 2, 3], ESTIMATE: [""] * 4, COMMENT: [""] * 4}).to_parquet(
        path, index=False
    )

    with pytest.raises(ValueError):
        save_notes(path, 1, {ESTIMATE: "oui"}, id_field="n", expected_id=2)

    assert pd.read_parquet(path)[ESTIMATE].tolist() == [""] * 4


def test_save_notes_refuses_a_field_that_is_not_a_column(output):
    with pytest.raises(KeyError):
        save_notes(output, 0, {"note_absent_a": "oui"}, id_field="n", expected_id=1)

    assert pd.read_parquet(output).columns.tolist() == ["n", ESTIMATE, COMMENT]


def test_save_notes_waits_for_a_held_lock(output):
    holder = FileLock(f"{output}.lock")
    holder.acquire()

    written = threading.Event()

    def write():
        save_notes(output, 0, {ESTIMATE: "oui"}, id_field="n", expected_id=1)
        written.set()

    writer = threading.Thread(target=write)
    writer.start()

    assert not written.wait(0.3)

    holder.release()

    assert written.wait(5)

    writer.join()

    assert pd.read_parquet(output)[ESTIMATE].tolist() == ["oui", "", ""]


def test_save_notes_refuses_a_lock_the_calling_thread_already_holds(output):
    with FileLock(f"{output}.lock", is_singleton=True), pytest.raises(RuntimeError):
        save_notes(output, 0, {ESTIMATE: "oui"}, id_field="n", expected_id=1)

    assert pd.read_parquet(output)[ESTIMATE].tolist() == [""] * ROWS


def test_save_notes_refuses_an_output_that_does_not_exist(tmp_path):
    absent = tmp_path / "absent.parquet"

    with pytest.raises(FileNotFoundError):
        save_notes(absent, 0, {ESTIMATE: "oui"}, id_field="n", expected_id=1)

    assert not absent.exists()


def test_save_notes_leaves_the_output_intact_when_the_write_fails(output, monkeypatch):
    original = pd.read_parquet(output)

    def torn_write(self, path, *args, **kwargs):
        Path(path).write_bytes(b"PAR1-truncated")
        raise OSError("no space left on device")

    monkeypatch.setattr(pd.DataFrame, "to_parquet", torn_write)

    with pytest.raises(OSError):
        save_notes(output, 0, {ESTIMATE: "oui"}, id_field="n", expected_id=1)

    monkeypatch.undo()

    pd.testing.assert_frame_equal(pd.read_parquet(output), original)


def test_save_notes_clears_the_staging_file_when_the_write_fails(output, monkeypatch):
    def torn_write(self, path, *args, **kwargs):
        Path(path).write_bytes(b"PAR1-truncated")
        raise OSError("no space left on device")

    monkeypatch.setattr(pd.DataFrame, "to_parquet", torn_write)

    with pytest.raises(OSError):
        save_notes(output, 0, {ESTIMATE: "oui"}, id_field="n", expected_id=1)

    monkeypatch.undo()

    assert list(output.parent.glob("*.tmp")) == []


def test_save_notes_leaves_no_temporary_file_behind(output):
    save_notes(output, 0, {ESTIMATE: "oui"}, id_field="n", expected_id=1)

    assert list(output.parent.glob("*.tmp")) == []


def test_save_notes_refuses_the_label_a_first_save_has_dropped(tmp_path):
    path = tmp_path / "shifted.parquet"
    make_output().set_index(pd.Index([10, 11, 12])).to_parquet(path)

    save_notes(path, 12, {ESTIMATE: "oui"}, id_field="n", expected_id=3)

    with pytest.raises(KeyError):
        save_notes(path, 12, {ESTIMATE: "non"}, id_field="n", expected_id=3)

    assert len(pd.read_parquet(path)) == ROWS


def _render(
    path,
    nrow,
    note_fields,
    values,
    columns,
    can_save,
    save_label,
    id_field,
    doc_id,
    keys,
    cursor,
):
    import streamlit as st

    from edscrib.navigation import navigation

    state = st.session_state

    for name, value in cursor.items():
        state.setdefault(name, value)

    for name in note_fields:
        state.setdefault(name, values[name])

    navigation(
        path,
        nrow,
        note_fields,
        columns=columns,
        can_save=can_save,
        save_label=save_label,
        id_field=id_field,
        doc_id=doc_id,
        **keys,
    )


def run_navigation(output, **kwargs):
    app = AppTest.from_function(
        _render,
        kwargs={
            "path": str(output),
            "nrow": ROWS,
            "note_fields": FIELDS,
            "values": {ESTIMATE: "oui", COMMENT: "c"},
            "columns": (1, 1, 1),
            "can_save": True,
            "save_label": "save",
            "id_field": "n",
            "doc_id": 1,
            "keys": {},
            "cursor": {"doc_index": 0, "save_count": 0},
        }
        | kwargs,
    )
    return app.run()


def test_navigation_renders_three_buttons(output):
    app = run_navigation(output)

    assert not app.exception
    assert len(app.button) == 3


def test_navigation_takes_the_arrow_labels_from_the_caller(output):
    default = run_navigation(output)
    passed = run_navigation(
        output, keys={"back_label": "Précédent", "forward_label": "Suivant"}
    )

    assert [button.label for button in default.button] == ["◀", "save", "▶"]
    assert [button.label for button in passed.button] == ["Précédent", "save", "Suivant"]


def test_navigation_save_writes_the_session_values_and_moves_on(output):
    app = run_navigation(output)

    app.button(key="button-save").click().run()

    assert app.session_state.doc_index == 1
    assert app.session_state.save_count == 1

    data = pd.read_parquet(output)

    assert data[ESTIMATE].tolist() == ["oui", "", ""]
    assert data[COMMENT].tolist() == ["c", "", ""]


def test_navigation_names_the_session_key_it_is_missing(output, caplog):
    app = run_navigation(output, cursor={"save_count": 0})

    assert not app.exception
    assert [error.value for error in app.error] == [MESSAGE_CURSOR]
    assert "doc_index" in caplog.text
    assert "doc_index" not in MESSAGE_CURSOR


def test_navigation_renders_nothing_when_the_cursor_is_missing(output):
    app = run_navigation(output, cursor={"save_count": 0})

    assert app.button.values == []


def test_navigation_names_the_width_count_it_was_given(output, caplog):
    app = run_navigation(output, columns=(1, 1))

    assert not app.exception
    assert [error.value for error in app.error] == [MESSAGE_CURSOR]
    assert app.button.values == []
    assert "not 2" in caplog.text


def test_navigation_does_not_write_when_saving_is_not_allowed(output):
    app = run_navigation(output, can_save=False)

    app.button(key="button-save").click().run()

    assert app.session_state.doc_index == 0
    assert app.session_state.save_count == 0
    assert pd.read_parquet(output)[ESTIMATE].tolist() == [""] * ROWS


def test_navigation_refuses_to_save_past_the_frame_on_disk(output):
    app = run_navigation(output, nrow=ROWS + 2)

    for _ in range(ROWS + 2):
        app.button(key="button-forward").click().run()

    app.button(key="button-save").click().run()

    assert not app.exception
    assert [error.value for error in app.error] == [MESSAGE_SAVE]
    assert app.session_state.save_count == 0
    assert pd.read_parquet(output)[ESTIMATE].tolist() == [""] * ROWS


def test_navigation_keeps_a_failed_save_out_of_the_browser(output, caplog):
    app = run_navigation(output, nrow=ROWS + 2)

    for _ in range(ROWS + 2):
        app.button(key="button-forward").click().run()

    app.button(key="button-save").click().run()

    assert not app.exception
    assert [error.value for error in app.error] == [MESSAGE_SAVE]
    assert all(str(output) not in error.value for error in app.error)
    assert str(output) in caplog.text


def test_navigation_does_not_move_on_from_a_failed_save(output):
    app = run_navigation(output, doc_id=99)

    app.button(key="button-save").click().run()

    assert app.session_state.doc_index == 0
    assert app.session_state.save_count == 0
    assert pd.read_parquet(output)[ESTIMATE].tolist() == [""] * ROWS


def test_navigation_honours_the_button_key_overrides(output):
    app = run_navigation(
        output,
        keys={
            "key_button_back": "back-alt",
            "key_button_save": "save-alt",
            "key_button_forward": "forward-alt",
        },
    )

    assert [button.key for button in app.button] == [
        "back-alt",
        "save-alt",
        "forward-alt",
    ]


def test_navigation_arrows_move_the_cursor_without_writing(output):
    app = run_navigation(output)

    app.button(key="button-forward").click().run()
    app.button(key="button-forward").click().run()
    app.button(key="button-forward").click().run()

    assert app.session_state.doc_index == ROWS - 1
    assert app.session_state.save_count == 0
    assert pd.read_parquet(output)[ESTIMATE].tolist() == [""] * ROWS

    app.button(key="button-backward").click().run()

    assert app.session_state.doc_index == ROWS - 2
