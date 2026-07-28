"""Cursor arithmetic and the save path, on synthetic frames and paths."""

import pandas as pd
import pytest
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


def _render(path, nrow, note_fields, values):
    import streamlit as st

    from edscrib.navigation import navigation

    state = st.session_state
    state.setdefault("doc_index", 0)
    state.setdefault("save_count", 0)

    for name in note_fields:
        state.setdefault(name, values[name])

    navigation(
        path,
        nrow,
        note_fields,
        columns=[1, 1, 1],
        can_save=True,
        save_label="save",
    )


def run_navigation(output, **kwargs):
    app = AppTest.from_function(
        _render,
        kwargs={
            "path": str(output),
            "nrow": ROWS,
            "note_fields": FIELDS,
            "values": {ESTIMATE: "oui", COMMENT: "c"},
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
    assert pd.read_parquet(output)[ESTIMATE].tolist() == ["oui", "", ""]


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
