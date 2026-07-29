"""Cursor arithmetic and the save path, on synthetic frames and paths."""

import threading
from pathlib import Path

import pandas as pd
import pytest
from filelock import FileLock
from streamlit.testing.v1 import AppTest

from edscrib.navigation import next_index, previous_index, save_notes

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
    save_notes(output, 1, {ESTIMATE: "oui", COMMENT: "c"})

    data = pd.read_parquet(output)

    assert data[ESTIMATE].tolist() == ["", "oui", ""]
    assert data[COMMENT].tolist() == ["", "c", ""]


def test_save_notes_leaves_the_fields_it_was_not_given_alone(output):
    save_notes(output, 0, {ESTIMATE: "oui"})

    data = pd.read_parquet(output)

    assert data[COMMENT].tolist() == [""] * ROWS
    assert data["n"].tolist() == [1, 2, 3]


def test_save_notes_does_not_clobber_a_previous_save(output):
    save_notes(output, 0, {ESTIMATE: "oui"})
    save_notes(output, 2, {ESTIMATE: "non"})

    assert pd.read_parquet(output)[ESTIMATE].tolist() == ["oui", "", "non"]


def test_save_notes_addresses_the_row_by_index_label(tmp_path):
    path = tmp_path / "shifted.parquet"
    make_output().set_index(pd.Index([10, 11, 12])).to_parquet(path)

    save_notes(path, 11, {ESTIMATE: "oui"})

    assert pd.read_parquet(path)[ESTIMATE].tolist() == ["", "oui", ""]


def test_save_notes_does_not_persist_the_index(tmp_path):
    path = tmp_path / "shifted.parquet"
    make_output().set_index(pd.Index([10, 11, 12])).to_parquet(path)

    save_notes(path, 11, {ESTIMATE: "oui"})

    assert pd.read_parquet(path).index.tolist() == list(range(ROWS))


def test_save_notes_refuses_a_row_that_does_not_exist(output):
    with pytest.raises(KeyError):
        save_notes(output, ROWS + 4, {ESTIMATE: "oui"})

    assert len(pd.read_parquet(output)) == ROWS


def test_save_notes_refuses_a_field_that_is_not_a_column(output):
    with pytest.raises(KeyError):
        save_notes(output, 0, {"note_absent_a": "oui"})

    assert pd.read_parquet(output).columns.tolist() == ["n", ESTIMATE, COMMENT]


def test_save_notes_waits_for_a_held_lock(output):
    holder = FileLock(f"{output}.lock")
    holder.acquire()

    written = threading.Event()

    def write():
        save_notes(output, 0, {ESTIMATE: "oui"})
        written.set()

    writer = threading.Thread(target=write)
    writer.start()

    assert not written.wait(0.3)

    holder.release()

    assert written.wait(5)

    writer.join()

    assert pd.read_parquet(output)[ESTIMATE].tolist() == ["oui", "", ""]


def test_save_notes_refuses_an_output_that_does_not_exist(tmp_path):
    absent = tmp_path / "absent.parquet"

    with pytest.raises(FileNotFoundError):
        save_notes(absent, 0, {ESTIMATE: "oui"})

    assert not absent.exists()


def test_save_notes_leaves_the_output_intact_when_the_write_fails(output, monkeypatch):
    original = pd.read_parquet(output)

    def torn_write(self, path, *args, **kwargs):
        Path(path).write_bytes(b"PAR1-truncated")
        raise OSError("no space left on device")

    monkeypatch.setattr(pd.DataFrame, "to_parquet", torn_write)

    with pytest.raises(OSError):
        save_notes(output, 0, {ESTIMATE: "oui"})

    monkeypatch.undo()

    pd.testing.assert_frame_equal(pd.read_parquet(output), original)


def test_save_notes_leaves_no_temporary_file_behind(output):
    save_notes(output, 0, {ESTIMATE: "oui"})

    assert list(output.parent.glob("*.tmp")) == []


def test_save_notes_refuses_the_label_a_first_save_has_dropped(tmp_path):
    path = tmp_path / "shifted.parquet"
    make_output().set_index(pd.Index([10, 11, 12])).to_parquet(path)

    save_notes(path, 12, {ESTIMATE: "oui"})

    with pytest.raises(KeyError):
        save_notes(path, 12, {ESTIMATE: "non"})

    assert len(pd.read_parquet(path)) == ROWS


def _render(path, nrow, note_fields, values, columns, can_save, save_label, keys, cursor):
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
            "columns": [1, 1, 1],
            "can_save": True,
            "save_label": "save",
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


def test_navigation_save_writes_the_session_values_and_moves_on(output):
    app = run_navigation(output)

    app.button(key="button-save").click().run()

    assert app.session_state.doc_index == 1
    assert app.session_state.save_count == 1

    data = pd.read_parquet(output)

    assert data[ESTIMATE].tolist() == ["oui", "", ""]
    assert data[COMMENT].tolist() == ["c", "", ""]


def test_navigation_names_the_session_key_it_is_missing(output):
    app = run_navigation(output, cursor={"save_count": 0})

    assert app.exception
    assert "doc_index" in app.exception[0].value


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

    assert app.exception
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
