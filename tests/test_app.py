"""The data path an app runs before rendering anything: reads, guards, first write."""

import threading
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest
from filelock import FileLock
from streamlit.testing.v1 import AppTest

from edscrib.app import (
    MESSAGE_COLUMNS,
    MESSAGE_LABELS,
    MESSAGE_UNAVAILABLE,
    aligned,
    labelled_by_position,
    load_frames,
    missing_fields,
    needs_write,
    write_output,
)
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


### LABEL GUARD ---------------------------------------------------------------


def test_labelled_by_position_accepts_a_frame_numbered_from_zero():
    assert labelled_by_position(make_input())


@pytest.mark.parametrize("labels", [[0, 1, 3], [1, 2, 4], [2, 0, 1]])
def test_labelled_by_position_rejects_a_gap_an_offset_and_a_permutation(labels):
    """The three labellings a parquet carries when its producer never reset the index."""
    data = make_input().set_axis(labels, axis=0)

    assert not labelled_by_position(data)


### DECLARED COLUMNS ----------------------------------------------------------


def test_missing_fields_names_what_the_frame_does_not_carry():
    data = make_input().drop(columns=["pat_age"])

    assert missing_fields(data, ("n", "pat_age", ID)) == ["pat_age"]
    assert missing_fields(data, ("n", ID)) == []


@pytest.mark.parametrize("dropped", ["pat_age", ID])
def test_load_frames_stops_on_a_reshape_the_alignment_guard_lets_through(
    project, dropped, caplog
):
    """A dropped meta column keeps the ids in value and in order, so `aligned` passes."""
    config, output = project
    folder = Path(config.work_dir) / "data" / "2023-01"
    reshaped = make_input().drop(columns=[dropped])
    reshaped.to_parquet(folder / "2023-01_avc_annot_data_input-review.parquet")

    app = run_shell(config)

    assert not app.exception
    assert app.error[0].value == MESSAGE_COLUMNS
    assert dropped not in app.error[0].value
    assert dropped in caplog.text
    assert not output.exists()


### WRITE CONDITION ------------------------------------------------------------


def test_needs_write_when_no_output_exists():
    assert needs_write(None, FIELDS)


def test_needs_write_when_the_output_lacks_a_note_column():
    started = make_input(**{ESTIMATE: "oui"}).drop(columns=[TEXT])

    assert needs_write(started, FIELDS)


def test_no_write_when_the_output_carries_the_columns_but_no_answer_yet():
    """The frame is on the disk already; writing over it would land a rebuild on it."""
    empty = make_input(**{ESTIMATE: "", COMMENT: ""}).drop(columns=[TEXT])

    assert not needs_write(empty, FIELDS)


def test_no_write_when_the_output_is_already_under_way():
    started = make_input(**{COMMENT: ""}).assign(**{ESTIMATE: ["oui", "", ""]})

    assert not needs_write(started.drop(columns=[TEXT]), FIELDS)


def test_a_second_annotator_column_does_not_count_as_this_ones_progress():
    """The column check sees this annotator alone: another one's column is not theirs."""
    other = make_input(**{"note_estimate_b": "oui"}).drop(columns=[TEXT])

    assert needs_write(other, FIELDS)


### WRITE ----------------------------------------------------------------------


def test_write_output_lands_the_frame_and_leaves_no_lock_behind(tmp_path):
    target = tmp_path / "output.parquet"

    write_output(make_input().drop(columns=[TEXT]), target)

    assert pd.read_parquet(target).shape == (3, 3)
    assert not list(tmp_path.glob("*.tmp"))


def test_write_output_leaves_the_accumulated_run_readable_when_it_fails(
    tmp_path, monkeypatch
):
    """An interrupted write on the target would be read by every later start."""
    target = tmp_path / "output.parquet"
    make_input().drop(columns=[TEXT]).to_parquet(target)
    before = target.read_bytes()

    def interrupted(self, path, *args, **kwargs):
        Path(path).write_bytes(b"PAR1truncated")
        raise OSError("no space left on device")

    monkeypatch.setattr(pd.DataFrame, "to_parquet", interrupted)

    with pytest.raises(OSError):
        write_output(make_input(2).drop(columns=[TEXT]), target)

    assert target.read_bytes() == before


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


def test_load_frames_does_not_write_over_a_save_it_waited_for(project):
    """Without this one the suite stays green when the read moves out of the lock."""
    config, output = project
    started = make_input(**{COMMENT: ""}).drop(columns=[TEXT])
    started.assign(**{ESTIMATE: ["oui", "", ""]}).to_parquet(output, index=False)

    other = make_config(config.work_dir)
    other = replace(
        other,
        groups=(
            FieldGroup(
                fields=(
                    NoteField("note_estimate_b", "AVC", "radio", VALUES),
                    NoteField("note_comment_b", "COMMENTAIRE", "text"),
                ),
            ),
        ),
    )

    holder = FileLock(f"{output}.lock")
    holder.acquire()

    loaded = threading.Event()

    def load():
        load_frames(other)
        loaded.set()

    reader = threading.Thread(target=load)
    reader.start()

    assert not loaded.wait(0.3)

    committed = pd.read_parquet(output)
    committed.loc[1, ESTIMATE] = "non"
    committed.to_parquet(output, index=False)
    holder.release()

    assert loaded.wait(5)

    reader.join()

    assert list(pd.read_parquet(output)[ESTIMATE]) == ["oui", "non", ""]


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


def test_load_frames_keeps_a_comment_saved_with_no_answer_to_the_radio(project):
    """Without this one the suite stays green when the resume predicate reads radios.

    The frame is asserted alongside the file: a predicate rebuilding in memory without
    writing leaves the comment on the disk and out of what the app renders.
    """
    config, output = project
    started = make_input().drop(columns=[TEXT]).assign(**{ESTIMATE: ""})
    started.assign(**{COMMENT: ["", "a revoir", ""]}).to_parquet(output, index=False)

    def script(config, comment):
        """Runs as a standalone script, so it imports what it uses itself."""
        import streamlit as st

        from edscrib.app import load_frames

        _, df_output = load_frames(config)

        st.text("|".join(df_output[comment]))

    app = AppTest.from_function(
        script, kwargs={"config": config, "comment": COMMENT}
    ).run()

    assert not app.exception
    assert app.text[0].value == "|a revoir|"
    assert list(pd.read_parquet(output)[COMMENT]) == ["", "a revoir", ""]


def test_load_frames_stops_on_an_input_whose_rows_are_not_labelled_by_position(project):
    """The input the guard rejects is the one a save would record on another row."""
    config, output = project
    folder = Path(config.work_dir) / "data" / "2023-01"
    permuted = make_input().set_axis([2, 0, 1], axis=0)
    permuted.to_parquet(folder / "2023-01_avc_annot_data_input-review.parquet")

    app = run_shell(config)

    assert not app.exception
    assert app.error[0].value == MESSAGE_LABELS
    assert not output.exists()


@pytest.mark.parametrize("broken", ["absent", "truncated"])
def test_load_frames_keeps_a_failing_read_out_of_the_browser(project, broken, caplog):
    """Two shapes of an unreadable input, neither of which may name the path.

    A readable input missing a declared column is the column guard's, not this one's.
    """
    config, output = project
    folder = Path(config.work_dir) / "data" / "2023-01"
    source = folder / "2023-01_avc_annot_data_input-review.parquet"

    if broken == "absent":
        source.unlink()
    else:
        source.write_bytes(source.read_bytes()[:64])

    app = run_shell(config)

    assert not app.exception
    assert app.error[0].value == MESSAGE_UNAVAILABLE
    assert str(folder) not in app.error[0].value
    assert not output.exists()
    assert "Annotation data unusable" in caplog.text


def test_load_frames_sees_an_input_regenerated_under_it(project):
    """Without this one the cache serves the first frame and the repair has no effect."""
    config, output = project
    folder = Path(config.work_dir) / "data" / "2023-01"
    source = folder / "2023-01_avc_annot_data_input-review.parquet"

    make_input().drop(columns=["pat_age"]).to_parquet(source)
    flagged = run_shell(config)

    assert flagged.error[0].value == MESSAGE_COLUMNS

    make_input().to_parquet(source)
    repaired = run_shell(config)

    assert not repaired.error
    assert output.exists()


def test_load_frames_stops_on_an_input_the_output_was_not_built_on(project):
    config, output = project
    stale = make_input(2).drop(columns=[TEXT]).assign(**{ESTIMATE: ["oui", "oui"]})
    stale.assign(**{COMMENT: ""}).to_parquet(output)

    app = run_shell(config)

    assert not app.exception
    assert app.error
    assert Path(output).exists()
