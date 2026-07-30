"""The data path an app runs before rendering anything: reads, guards, first write."""

import threading
from pathlib import Path

import pandas as pd
import pytest
from filelock import FileLock
from streamlit.testing.v1 import AppTest

from edscrib.app import aligned, needs_write, write_output
from edscrib.config import AnnotConfig, FieldGroup, NoteField

ID = "id_doc"
TEXT = "doc_text"
ESTIMATE = "note_estimate_a"
COMMENT = "note_comment_a"
PREFIX = "note_estimate"
VALUES = ("oui", "non")
FIELDS = (ESTIMATE, COMMENT)


def make_input(rows=3, **extra):
    frame = {
        "n": list(range(1, rows + 1)),
        ID: [f"d{i}" for i in range(rows)],
        "pat_age": [70] * rows,
        TEXT: ["<p>t</p>"] * rows,
    }
    return pd.DataFrame(frame | {name: [value] * rows for name, value in extra.items()})


def make_config(work_dir):
    return AnnotConfig(
        work_dir=work_dir,
        proj="avc",
        split="2023-01",
        input_suffix="input-review",
        output_suffix="output-review",
        groups=(
            FieldGroup(
                fields=(
                    NoteField(ESTIMATE, "AVC", "radio", VALUES),
                    NoteField(COMMENT, "COMMENTAIRE", "text"),
                ),
            ),
        ),
        table_fields=("n", ESTIMATE),
        estimate_prefix=PREFIX,
        index_field="n",
        id_field=ID,
        text_field=TEXT,
        meta_fields={"pat_age": "ÂGE"},
    )


@pytest.fixture
def project(tmp_path):
    """A work_dir holding the input where data_path expects to find it."""
    config = make_config(tmp_path)
    folder = tmp_path / "data" / "2023-01"
    folder.mkdir(parents=True)
    make_input().to_parquet(folder / "2023-01_avc_annot_data_input-review.parquet")
    return config, folder / "2023-01_avc_annot_data_output-review.parquet"


### ALIGNMENT GUARD ------------------------------------------------------------


def test_aligned_accepts_two_frames_carrying_the_same_ids_in_order():
    data = make_input()

    assert aligned(data, data.drop(columns=[TEXT]), ID)


def test_aligned_rejects_a_permutation():
    data = make_input()
    shuffled = data.iloc[::-1].reset_index(drop=True)

    assert not aligned(data, shuffled, ID)


def test_aligned_rejects_a_different_length():
    assert not aligned(make_input(3), make_input(2), ID)


def test_aligned_rejects_a_non_contiguous_index():
    data = make_input()
    gapped = data.drop(index=1)

    assert not aligned(data, gapped, ID)


### WRITE CONDITION ------------------------------------------------------------


def test_needs_write_when_no_output_exists():
    assert needs_write(None, FIELDS, PREFIX, VALUES)


def test_needs_write_when_the_output_lacks_a_note_column():
    started = make_input(**{ESTIMATE: "oui"}).drop(columns=[TEXT])

    assert needs_write(started, FIELDS, PREFIX, VALUES)


def test_needs_write_when_the_output_holds_no_estimate():
    empty = make_input(**{ESTIMATE: "", COMMENT: ""}).drop(columns=[TEXT])

    assert needs_write(empty, FIELDS, PREFIX, VALUES)


def test_no_write_when_the_output_is_already_under_way():
    started = make_input(**{COMMENT: ""}).assign(**{ESTIMATE: ["oui", "", ""]})

    assert not needs_write(started.drop(columns=[TEXT]), FIELDS, PREFIX, VALUES)


def test_a_second_annotator_column_does_not_count_as_this_ones_progress():
    """The prefix scan sees every annotator, the column check sees only this one."""
    other = make_input(**{"note_estimate_b": "oui"}).drop(columns=[TEXT])

    assert needs_write(other, FIELDS, PREFIX, VALUES)


### WRITE ----------------------------------------------------------------------


def test_write_output_lands_the_frame_and_leaves_no_lock_behind(tmp_path):
    target = tmp_path / "output.parquet"

    write_output(make_input().drop(columns=[TEXT]), target)

    assert pd.read_parquet(target).shape == (3, 3)
    assert not list(tmp_path.glob("*.tmp"))


def test_write_output_waits_for_a_held_lock(tmp_path):
    """Without this one the suite stays green when the lock is deleted."""
    target = tmp_path / "output.parquet"
    holder = FileLock(f"{target}.lock")
    holder.acquire()

    written = threading.Event()

    def write():
        write_output(make_input().drop(columns=[TEXT]), target)
        written.set()

    writer = threading.Thread(target=write)
    writer.start()

    assert not written.wait(0.3)

    holder.release()

    assert written.wait(5)

    writer.join()

    assert pd.read_parquet(target).shape == (3, 3)


### SHELL ----------------------------------------------------------------------


def run_shell(config):
    def script(config):
        """Runs as a standalone script, so it imports what it uses itself."""
        import streamlit as st

        from edscrib.app import load_frames

        df_input, df_output = load_frames(config)

        st.text(f"{len(df_input)}/{len(df_output)}")

    return AppTest.from_function(script, kwargs={"config": config}).run()


def test_load_frames_creates_the_output_on_a_first_run(project):
    config, output = project

    app = run_shell(config)

    assert not app.exception
    assert output.exists()
    assert app.text[0].value == "3/3"


def test_load_frames_leaves_an_output_under_way_untouched(project):
    config, output = project
    started = make_input().drop(columns=[TEXT]).assign(**{ESTIMATE: ["oui", "", ""]})
    started = started.assign(**{COMMENT: ""})
    started.to_parquet(output)
    before = output.read_bytes()

    app = run_shell(config)

    assert not app.exception
    assert output.read_bytes() == before


def test_load_frames_stops_on_an_input_the_output_was_not_built_on(project):
    config, output = project
    stale = make_input(2).drop(columns=[TEXT]).assign(**{ESTIMATE: ["oui", "oui"]})
    stale.assign(**{COMMENT: ""}).to_parquet(output)

    app = run_shell(config)

    assert not app.exception
    assert app.error
    assert Path(output).exists()
