"""The data path an app runs before rendering anything: reads, guards, first write."""

import os
import threading
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest
from filelock import FileLock
from streamlit.testing.v1 import AppTest

from edscrib.app import (
    _stamp,
    aligned,
    frozen_fields,
    labelled_by_position,
    load_frames,
    missing_fields,
    needs_write,
    repeated_ids,
    retyped_fields,
    same_file,
    unresolved_fields,
    write_output,
)
from edscrib.config import USER, AnnotConfig, FieldGroup, NoteField, Styles, Tuto
from edscrib.messages import (
    MESSAGE_COLUMNS,
    MESSAGE_DUPLICATES,
    MESSAGE_LABELS,
    MESSAGE_MISMATCH,
    MESSAGE_TYPES,
    MESSAGE_UNAVAILABLE,
    MESSAGE_VALUES,
)

ID = "id_doc"
TEXT = "doc_text"
ESTIMATE = "note_estimate_a"
COMMENT = "note_comment_a"
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
        input_stem="input-review",
        output_stem="output-review",
        secrets=Path(work_dir) / "secrets.toml",
        styles=Styles(
            app=Path(work_dir) / "style_app.css",
            text=Path(work_dir) / "style_text.css",
        ),
        groups=(
            FieldGroup(
                fields=(
                    NoteField(ESTIMATE, "AVC", "radio", VALUES),
                    NoteField(COMMENT, "COMMENTAIRE", "text"),
                ),
            ),
        ),
        table_fields=("n", ESTIMATE),
        index_field="n",
        id_field=ID,
        text_field=TEXT,
        estimate_field=ESTIMATE,
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


def test_aligned_accepts_ids_whose_type_moved_without_their_values():
    """The cohort is what this answers on, and a backend is not a cohort.

    `Series.equals` reads the dtype beside the values, so the two frames designating the
    same documents came back as different ones the day a parquet round trip landed the
    identifier on the other spelling of text.
    """
    data = make_input()
    retyped = data.drop(columns=[TEXT]).astype({ID: "string"})

    assert aligned(data, retyped, ID)
    assert not data[ID].equals(retyped[ID])


### INPUT STAMP ---------------------------------------------------------------


def test_stamp_separates_two_writes_a_coarse_mtime_would_collapse(tmp_path):
    """Without this one the cache serves the frame the guards asked to regenerate.

    A mount reporting seconds collapses the two modification times, and a regenerator
    writing the same rows lands the same size. The rename onto a fresh inode is what
    the stamp still has to tell apart.
    """
    path = tmp_path / "in.parquet"
    make_input().to_parquet(path)
    before = path.stat()
    stamp = _stamp(path)

    staged = tmp_path / "in.parquet.tmp"
    make_input().to_parquet(staged)
    staged.replace(path)
    os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))

    assert path.stat().st_size == before.st_size
    assert path.stat().st_mtime_ns == before.st_mtime_ns
    assert _stamp(path) != stamp


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


@pytest.mark.parametrize(
    ("dropped", "guarded"),
    [("pat_age", "output"), (ID, "input")],
    ids=["meta-column", "identifier"],
)
def test_load_frames_stops_on_a_column_the_reshaped_input_no_longer_carries(
    project, dropped, guarded, caplog
):
    """Two reshapes of one input, each stopped by a different guard.

    A dropped meta column passes the guard on the way in, which reads the identifier
    and the text alone, and is found missing from the output built on it. A dropped
    identifier never gets that far. Both render the one message, so the log is what
    tells them apart, and asserting the frame it names is what keeps each parameter
    on the guard it is written for.
    """
    config, output = project
    folder = Path(config.work_dir) / "data" / "2023-01"
    reshaped = make_input().drop(columns=[dropped])
    reshaped.to_parquet(folder / "2023-01_avc_annot_data_input-review.parquet")

    app = run_shell(config)

    assert not app.exception
    assert app.error[0].value == MESSAGE_COLUMNS
    assert dropped not in app.error[0].value
    assert f"The {guarded} does not carry ['{dropped}']" in caplog.text
    assert not output.exists()


def test_load_frames_names_the_identifier_the_output_does_not_carry(project, caplog):
    """The guard runs ahead of `aligned`, which would raise on the same frame.

    Caught at the boundary, that raise reaches the browser as an unreadable file rather
    than as the missing column it is.
    """
    config, output = project
    make_input().drop(columns=[TEXT, ID]).to_parquet(output)

    app = run_shell(config)

    assert not app.exception
    assert app.error[0].value == MESSAGE_COLUMNS
    assert ID not in app.error[0].value
    assert f"The output does not carry ['{ID}']" in caplog.text


### PLACEHOLDER ---------------------------------------------------------------


def test_unresolved_fields_is_empty_on_a_configuration_carrying_no_placeholder(tmp_path):
    assert unresolved_fields(make_config(tmp_path)) == []


@pytest.mark.parametrize(
    ("field", "value", "named"),
    [
        ("work_dir", f"wk/{USER}", f"wk/{USER}"),
        ("data_suffix", f"_{USER}", f"_{USER}"),
        ("id_field", f"id_{USER}", f"id_{USER}"),
        ("meta_fields", {f"pat_{USER}": f"ÂGE {USER}"}, f"pat_{USER}"),
        ("table_fields", ("n", f"note_{USER}"), f"note_{USER}"),
        ("tuto", Tuto(text=Path(f"{USER}.md"), media=Path("t.webm")), f"{USER}.md"),
    ],
    ids=["work-dir", "suffix", "column", "meta-key", "table-field", "asset"],
)
def test_unresolved_fields_walks_every_field_the_binding_never_touches(
    tmp_path, field, value, named
):
    """`resolve` binds three of the twenty-three, so the guard covers the twenty.

    The walk is generic, so a field the shape grows later is covered without anyone
    remembering to add it here, which is the failure this replaces.
    """
    config = replace(make_config(tmp_path), **{field: value})

    assert named in unresolved_fields(config)


def test_unresolved_fields_reaches_a_label_and_a_group_that_is_never_written(tmp_path):
    """The two the old scan was blind to, and the label has no feedback path at all.

    It reaches the annotator beside the clinical question, and reading the names off
    `persisted_fields` leaves a reference group's own out of the scan entirely.
    """
    config = replace(
        make_config(tmp_path),
        groups=(
            FieldGroup(
                fields=(
                    NoteField(f"note_estimate_{USER}", "AVC", "radio", VALUES),
                    NoteField("note_comment_b", f"COMMENTAIRE {USER}", "text"),
                ),
                persisted=False,
            ),
            FieldGroup(
                fields=(
                    NoteField(ESTIMATE, "AVC", "radio", VALUES),
                    NoteField(COMMENT, "COMMENTAIRE", "text"),
                ),
            ),
        ),
    )

    assert unresolved_fields(config) == [
        f"COMMENTAIRE {USER}",
        f"note_estimate_{USER}",
    ]


def test_load_frames_stops_on_a_configuration_that_was_never_resolved(project, caplog):
    """Without this one the placeholder becomes a column of the gold standard.

    Nothing later catches it: the output carries the literal name, and every guard
    passes against it on every run that follows.
    """
    config, output = project
    unresolved = replace(
        config,
        groups=(
            FieldGroup(
                fields=(
                    NoteField(f"note_estimate_{USER}", "AVC", "radio", VALUES),
                    NoteField(f"note_comment_{USER}", "COMMENTAIRE", "text"),
                ),
            ),
        ),
        table_fields=("n", f"note_estimate_{USER}"),
        estimate_field=f"note_estimate_{USER}",
    )

    app = run_shell(unresolved)

    assert not app.exception
    assert app.error[0].value == MESSAGE_UNAVAILABLE
    assert USER not in app.error[0].value
    assert f"note_estimate_{USER}" in caplog.text
    assert not output.exists()


@pytest.mark.parametrize("component", ["work_dir", "split", "input_stem"])
def test_load_frames_names_a_placeholder_left_in_a_path_component(
    project, component, caplog
):
    """The guard walks the fields the paths are built from, so it runs ahead of them.

    Behind the read, each of these names a file nothing ever wrote: the operator is
    handed a `FileNotFoundError` and sent to the data generation for a configuration
    that was never resolved. The assertion is on the class of the failure, which is all
    the ordering decides, nothing being written between the two positions either way.
    """
    config, output = project
    value = str(Path(config.work_dir) / USER) if component == "work_dir" else USER
    app = run_shell(replace(config, **{component: value}))

    assert not app.exception
    assert app.error[0].value == MESSAGE_UNAVAILABLE
    assert "The configuration was never resolved" in caplog.text
    assert "FileNotFoundError" not in caplog.text
    assert not output.exists()


def test_load_frames_names_a_reference_column_the_output_does_not_carry(project, caplog):
    """A group shown for reference owns columns `build_output` never creates.

    They reach the output through the input's first copy alone, so an input that drops
    or renames one leaves the render reading a column nothing put there. Reading the
    declared names off `persisted_fields` would cover them only when the consumer repeats
    among the table fields, which is a summary table's list.
    """
    config, output = project
    reference = replace(
        config,
        groups=(
            FieldGroup(
                fields=(
                    NoteField("note_estimate_b", "AVC_B", "radio", VALUES, False),
                    NoteField("note_comment_b", "COMMENTAIRE_B", "text", (), False),
                ),
                persisted=False,
            ),
            *config.groups,
        ),
    )

    app = run_shell(reference)

    assert not app.exception
    assert app.error[0].value == MESSAGE_COLUMNS
    assert "note_estimate_b" not in app.error[0].value
    assert (
        "The output does not carry ['note_comment_b', 'note_estimate_b']" in caplog.text
    )
    assert not output.exists()


def test_load_frames_stops_on_two_fields_the_binding_put_on_one_column(project, caplog):
    """Reference and reconciliation collapsing onto one column of the gold standard.

    A group naming an annotator literally and a persisted group naming the same one
    through the placeholder resolve together, and the save writes the second answer
    over what the first was there to show. `persisted_fields` holds the second name alone,
    the placeholder guard is blind to it, and `frozen_fields` excludes the column for
    being one the annotation writes.
    """
    config, output = project
    reference = NoteField(ESTIMATE, "AVC_A", "radio", VALUES, editable=False)
    collided = replace(
        config,
        groups=(
            FieldGroup(
                fields=(reference, NoteField(COMMENT, "COMMENTAIRE_A", "text")),
                persisted=False,
            ),
            *config.groups,
        ),
    )

    app = run_shell(collided)

    assert not app.exception
    assert app.error[0].value == MESSAGE_UNAVAILABLE
    assert ESTIMATE not in app.error[0].value
    assert f"Two fields are declared on ['{COMMENT}', '{ESTIMATE}']" in caplog.text
    assert not output.exists()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("table_fields", ("n", TEXT)),
        ("meta_fields", {TEXT: "COMPTE-RENDU"}),
        ("index_field", TEXT),
    ],
    ids=["table", "meta", "index"],
)
def test_load_frames_stops_on_a_declaration_naming_the_document_text(
    project, field, value, caplog
):
    """The output is what every declared column is read from, and it carries no text.

    `build_output` drops it unconditionally, so the summary table is sourced from the
    output and a consumer asking it for the clinical text is asking for a column that
    cannot exist. Without this the run stopped on the declared-column guard instead,
    which names a column the operator then goes looking for in a data generation that
    is not wrong.

    Every declaration is covered and not the table alone: the same name reaches the
    output through the metadata or the counter just as well, and the failure and the
    remedy are the same wherever it was written.
    """
    config, output = project

    app = run_shell(replace(config, **{field: value}))

    assert not app.exception
    assert app.error[0].value == MESSAGE_UNAVAILABLE
    assert TEXT not in app.error[0].value
    assert f"The document text is declared as a column: '{TEXT}'" in caplog.text
    assert not output.exists()


def test_load_frames_stops_on_a_field_annotating_a_column_the_input_carries(
    project, caplog
):
    """The annotation lands on the metadata and nothing downstream can see it.

    `build_output` creates a column only where the frame has none, so the field inherits
    the input's values in place of the sentinel, the save writes the answer over them,
    and `frozen_fields` drops the column from its comparison for being one the annotation
    writes. The export then ships the mixture under the metadata's own header.
    """
    config, output = project
    collided = replace(
        config,
        groups=(
            FieldGroup(
                fields=(
                    NoteField("pat_age", "ÂGE ?", "radio", VALUES),
                    NoteField(COMMENT, "COMMENTAIRE", "text"),
                ),
            ),
        ),
        table_fields=("n", "pat_age"),
        estimate_field="pat_age",
    )

    app = run_shell(collided)

    assert not app.exception
    assert app.error[0].value == MESSAGE_UNAVAILABLE
    assert "pat_age" not in app.error[0].value
    assert "A field annotates the input column ['pat_age']" in caplog.text
    assert not output.exists()


def test_load_frames_accepts_a_reference_field_reading_a_column_of_the_input(project):
    """A reconciliation app shows a prior annotator's answer, which the input carries.

    The guard above reads `persisted_fields` and not `rendered` for this: nothing writes
    a non-persisted field, `build_output` never touches it and `frozen_fields` does cover
    it, so widening the guard would stop the very app a reference group exists for.
    """
    config, output = project
    folder = Path(config.work_dir) / "data" / "2023-01"
    reviewed = make_input(note_estimate_b="oui", note_comment_b="")
    reviewed.to_parquet(folder / "2023-01_avc_annot_data_input-review.parquet")
    merged = replace(
        config,
        groups=(
            FieldGroup(
                fields=(
                    NoteField("note_estimate_b", "AVC_B", "radio", VALUES),
                    NoteField("note_comment_b", "COMMENTAIRE_B", "text"),
                ),
                persisted=False,
            ),
            *config.groups,
        ),
    )

    app = run_shell(merged)

    assert not app.exception
    assert not app.error
    assert output.exists()


def test_repeated_ids_names_the_positions_and_never_the_values():
    """The identifier is patient-linked, and the log is the deployment's to read."""
    data = make_input().assign(**{ID: ["d0", "d1", "d1"]})

    assert repeated_ids(data, ID) == [1, 2]
    assert repeated_ids(make_input(), ID) == []


def test_load_frames_stops_on_an_identifier_designating_two_documents(project, caplog):
    """What rests on the answer is the save's re-check, at the click.

    A run extended with a document that shares its identifier with the one it displaces
    leaves that check comparing a value against itself, so it accepts the write onto
    either document. Nothing else asks: the alignment guard compares the two frames
    against each other and the labelling guard constrains another axis.
    """
    config, output = project
    folder = Path(config.work_dir) / "data" / "2023-01"
    repeated = make_input().assign(**{ID: ["d0", "d1", "d1"]})
    repeated.to_parquet(folder / "2023-01_avc_annot_data_input-review.parquet")

    app = run_shell(config)

    assert not app.exception
    assert app.error[0].value == MESSAGE_DUPLICATES
    assert "d1" not in app.error[0].value
    assert "d1" not in caplog.text
    assert "One identifier designates the documents at [1, 2]" in caplog.text
    assert not output.exists()


def test_same_file_asks_the_kernel_where_the_two_names_differ(tmp_path):
    """A lexical comparison answers none of the ways one file takes two names."""
    target = tmp_path / "target.parquet"
    target.write_bytes(b"x")
    linked = tmp_path / "linked.parquet"
    linked.symlink_to(target)
    hard = tmp_path / "hard.parquet"
    os.link(target, hard)

    assert same_file(target, target)
    assert same_file(target, linked)
    assert same_file(target, hard)
    assert not same_file(target, tmp_path / "elsewhere.parquet")

    dangling = tmp_path / "dangling.parquet"
    dangling.symlink_to(tmp_path / "absent.parquet")

    assert not same_file(target, dangling)


def test_load_frames_stops_on_a_second_name_for_the_input(project, caplog):
    """Nothing is destroyed this way, and the deliverable keeps the clinical text.

    `Path.replace` takes the link's own entry rather than its target, so the input
    survives where a folded case would lose it. What lands instead is an output built by
    resuming on the input, text column and all, which is what the export reads.
    """
    config, output = project
    folder = Path(config.work_dir) / "data" / "2023-01"
    source = folder / "2023-01_avc_annot_data_input-review.parquet"
    output.symlink_to(source)

    app = run_shell(config)

    assert not app.exception
    assert app.error[0].value == MESSAGE_UNAVAILABLE
    assert str(source) not in app.error[0].value
    assert str(source) in caplog.text
    assert output.is_symlink()
    assert ESTIMATE not in pd.read_parquet(output).columns


def test_load_frames_stops_on_two_stems_resolving_to_one_file(project, caplog):
    """Without this one the app annotates the input and no second copy is left.

    Every guard that follows compares the file against itself and passes, the build
    resumes on the input and keeps its text, and the write puts the merged frame back
    over it. Nothing surfaces, and the rebuild the other guards prescribe would erase
    the annotation accumulated since.
    """
    config, _ = project
    collided = replace(config, output_stem=config.input_stem)
    folder = Path(config.work_dir) / "data" / "2023-01"
    path = folder / "2023-01_avc_annot_data_input-review.parquet"
    before = pd.read_parquet(path)

    app = run_shell(collided)

    assert not app.exception
    assert app.error[0].value == MESSAGE_UNAVAILABLE
    assert str(path) not in app.error[0].value
    assert str(path) in caplog.text
    assert pd.read_parquet(path).equals(before)


### FROZEN VALUES -------------------------------------------------------------


def test_frozen_fields_names_the_columns_the_output_records_differently():
    df_input = make_input()
    df_output = df_input.drop(columns=[TEXT]).assign(pat_age=[81, 82, 83])

    assert frozen_fields(df_input, df_output, ("n", "pat_age", ID)) == ["pat_age"]
    assert frozen_fields(df_input, df_output, ("n", ID)) == []


def test_frozen_fields_raises_on_a_name_a_frame_does_not_carry():
    """Skipping it would exempt the column from the comparison written to catch it.

    A regenerated input that dropped a declared column is what reaches this, and an
    intersection would answer that nothing changed. The caller establishes presence on
    both frames, as it does for the identifier `aligned` compares.
    """
    df_input = make_input()
    df_output = df_input.drop(columns=[TEXT]).assign(**{ESTIMATE: "oui"})

    with pytest.raises(KeyError, match=TEXT):
        frozen_fields(df_input, df_output, (TEXT, "pat_age"))

    with pytest.raises(KeyError, match=ESTIMATE):
        frozen_fields(df_input, df_output, (ESTIMATE, "pat_age"))


def test_frozen_fields_names_a_value_that_changed_under_a_reconciled_type():
    """The reconciliation covers how a type is spelled and never what a column holds.

    Both moves land on one column here, which is the case a comparison run on the
    converted pair could swallow whole: the type is stepped over, the corrected value
    is not.
    """
    df_output = make_input()
    df_input = make_input().astype({"pat_age": "Int64"})
    df_input.loc[1, "pat_age"] = 71

    assert not retyped_fields(df_input, df_output, ("pat_age",))
    assert frozen_fields(df_input, df_output, ("pat_age",)) == ["pat_age"]


def test_frozen_fields_names_a_missing_value_the_wider_type_allowed():
    """A value the narrower type could not hold at all, which is still a value.

    The nullable integer is a family the reconciliation accepts, and the pair is only
    accepted as far as the type: an answer that became missing between the two frames
    is a correction the annotator has to be stopped on, not a spelling.
    """
    df_output = make_input()
    df_input = make_input().astype({"pat_age": "Int64"})
    df_input.loc[1, "pat_age"] = pd.NA

    assert not retyped_fields(df_input, df_output, ("pat_age",))
    assert frozen_fields(df_input, df_output, ("pat_age",)) == ["pat_age"]


def test_load_frames_stops_on_an_input_corrected_after_the_output_was_built(
    project, caplog
):
    """Without this one the suite stays green when the correction never lands.

    The regenerated input keeps its ids and its column names, so the alignment and
    column guards both pass and the output goes on carrying the value it froze.
    """
    config, output = project
    folder = Path(config.work_dir) / "data" / "2023-01"
    source = folder / "2023-01_avc_annot_data_input-review.parquet"

    assert not run_shell(config).error
    assert list(pd.read_parquet(output)["pat_age"]) == [70, 70, 70]

    make_input().assign(pat_age=[81, 82, 83]).to_parquet(source)

    app = run_shell(config)

    assert not app.exception
    assert app.error[0].value == MESSAGE_VALUES
    assert "pat_age" not in app.error[0].value
    assert "pat_age" in caplog.text
    assert list(pd.read_parquet(output)["pat_age"]) == [70, 70, 70]


def test_load_frames_stops_on_a_declared_column_the_input_no_longer_carries(
    project, caplog
):
    """The output froze its copy, so the guard reading it alone cannot see one leave.

    A column removed upstream, deliberately or not, was served and exported from that
    frozen copy on every later run with nothing rendered and nothing logged, the value
    comparison having dropped the name for want of something to compare it against.
    """
    config, output = project
    folder = Path(config.work_dir) / "data" / "2023-01"
    source = folder / "2023-01_avc_annot_data_input-review.parquet"

    assert not run_shell(config).error
    assert list(pd.read_parquet(output)["pat_age"]) == [70, 70, 70]

    make_input().drop(columns=["pat_age"]).to_parquet(source)

    app = run_shell(config)

    assert not app.exception
    assert app.error[0].value == MESSAGE_COLUMNS
    assert "pat_age" not in app.error[0].value
    assert "The output carries ['pat_age'] where the input does not" in caplog.text


def test_load_frames_refuses_an_output_that_carries_the_document_text(project, caplog):
    """Declaring the text nowhere says nothing about what the output holds.

    `build_output` drops it on the first build alone and hands back a resumed one as it
    sits on the disk, and no declared name reaches the text, so all three column guards
    pass over it. It would stay for the life of the file, at 190 times the size and 16
    times the save.
    """
    config, output = project
    folder = Path(config.work_dir) / "data" / "2023-01"
    source = folder / "2023-01_avc_annot_data_input-review.parquet"
    read = pd.read_parquet(source)
    read.assign(**{ESTIMATE: "", COMMENT: ""}).to_parquet(output, index=False)

    app = run_shell(config)

    assert not app.exception
    assert app.error[0].value == MESSAGE_UNAVAILABLE
    assert TEXT not in app.error[0].value
    assert f"The output carries the document text: '{TEXT}'" in caplog.text


def test_load_frames_refuses_a_reference_the_output_alone_carries(project, caplog):
    """A reference column comes from the input, and the guard says so without inventing.

    It is the one thing the output can carry where the input never did: another
    annotator's app declares the same column persisted, and `build_output` creates it
    there against that configuration alone. So the message cannot say the input stopped
    carrying it, and names the asymmetry it actually observed.

    Refusing is the point rather than the cost. Sourcing the reference from the output
    would leave it compared against nothing, and a prior annotator revising an answer
    would stop nothing and reach the reconciliation nowhere.
    """
    config, _ = project
    first = replace(
        config,
        groups=(
            FieldGroup(
                fields=(
                    NoteField("note_estimate_a", "AVC", "radio", VALUES),
                    NoteField("note_comment_a", "COMMENTAIRE", "text"),
                ),
            ),
        ),
        table_fields=("n",),
    )

    assert not run_shell(first).error

    reconciler = replace(
        first,
        groups=(
            FieldGroup(fields=first.groups[0].fields, persisted=False),
            FieldGroup(
                fields=(
                    NoteField("note_estimate_m", "AVC", "radio", VALUES),
                    NoteField("note_comment_m", "COMMENTAIRE", "text"),
                ),
            ),
        ),
        estimate_field="note_estimate_m",
    )

    app = run_shell(reconciler)

    assert not app.exception
    assert app.error[0].value == MESSAGE_COLUMNS
    assert "note_estimate_a" not in app.error[0].value
    assert "where the input does not" in caplog.text
    assert "no longer" not in caplog.text


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
        table_fields=("n",),
        estimate_field="note_estimate_b",
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


@pytest.mark.parametrize(
    ("column", "dtype"),
    [("pat_age", "float64"), ("pat_age", "str"), (ID, "category")],
)
def test_retyped_fields_names_a_column_whose_type_moved_under_equal_values(column, dtype):
    """`Series.equals` answers on the values and the dtype at once, this one on neither.

    The values are held equal across the pair on purpose: what the guard has to see is
    the type alone, since that is the case the two comparisons downstream misreport.

    Every pair here crosses two families, which is what keeps it reported. The
    reconciliation knows two families and nothing outside them is guessed at: a
    conversion accepted by mistake is a value that reads as equal without being one.
    """
    df_input = make_input().astype({column: dtype})
    df_output = make_input()

    assert retyped_fields(df_input, df_output, (column,)) == [column]
    assert not retyped_fields(df_output, df_output, (column,))
    assert not df_input[column].equals(df_output[column])


def test_retyped_fields_raises_on_a_name_a_frame_does_not_carry():
    """The same contract its sibling carries, and the same reason to keep it.

    Both frames are established to hold every name before either is called. Softening
    the lookup into a skip would exempt from the type comparison exactly the column a
    reshaped input dropped, which is the one it exists for, and nothing would be
    rendered or logged.
    """
    df_output = make_input()
    df_input = df_output.drop(columns=["pat_age"])

    with pytest.raises(KeyError):
        retyped_fields(df_input, df_output, ("pat_age",))


@pytest.mark.parametrize(
    ("column", "dtype"),
    [("pat_age", "Int64"), ("pat_age", "int32"), (ID, "string"), (ID, "object")],
)
def test_retyped_fields_reconciles_a_spelling_that_holds_the_same_values(column, dtype):
    """Two spellings of one type are what a version bump lands on, not a drift.

    The pair is asked of the value guard in the same breath, which is where the report
    was doing its damage: reading the type alone would leave `frozen_fields` answering
    through `Series.equals` on the dtype, and the app stopped on a value nobody touched.

    The last assertion is what keeps this from passing vacuously: pandas itself still
    calls the two columns different, and the reconciliation is what steps over it.
    """
    df_input = make_input().astype({column: dtype})
    df_output = make_input()

    assert not retyped_fields(df_input, df_output, (column,))
    assert not frozen_fields(df_input, df_output, (column,))
    assert not df_input[column].equals(df_output[column])


def test_retyped_fields_reports_a_conversion_that_would_not_hold_every_value():
    """The reconciliation is closed on failure, on an integer past the signed range.

    `astype` refuses that cast rather than wrapping it, so the pair comes back untouched
    and the column is reported: a type that cannot hold what the other carries is not
    the same type spelled twice.
    """
    df_input = pd.DataFrame({"count": pd.Series([2**63], dtype="uint64")})
    df_output = pd.DataFrame({"count": pd.Series([1], dtype="int64")})

    assert retyped_fields(df_input, df_output, ("count",)) == ["count"]


def test_retyped_fields_reports_an_object_column_holding_more_than_strings():
    """An object column is a family only where every value it holds is a string.

    The dtype alone answers nothing here, `is_string_dtype` holding for any object
    column whatever sits in it, so the contents are what decides. A mixed one converts
    to text and compares equal against the digits someone wrote as digits.
    """
    df_output = make_input()
    df_input = make_input()
    df_input[ID] = pd.Series(["d0", "d1", 2], dtype=object)

    assert retyped_fields(df_input, df_output, (ID,)) == [ID]


@pytest.mark.parametrize("column", ["pat_age", ID])
def test_load_frames_names_a_type_that_moved_rather_than_a_value_or_a_cohort(
    project, column, caplog
):
    """One drift reported by two guards as two things that did not happen.

    A metadata column came back as values that changed and the identifier as a cohort
    that did, and the rebuild both prescribe reproduces the type it is regenerated by,
    so the app stayed stopped with the annotation intact and out of reach. Asserting the
    two old messages are absent is what carries the fix; asserting the run still stops
    is what keeps it from being a loosening.
    """
    config, output = project
    folder = Path(config.work_dir) / "data" / "2023-01"
    started = make_input().drop(columns=[TEXT]).assign(**{ESTIMATE: "", COMMENT: ""})
    started.to_parquet(output, index=False)
    dtype = "category" if column == ID else "float64"
    make_input().astype({column: dtype}).to_parquet(
        folder / "2023-01_avc_annot_data_input-review.parquet"
    )

    app = run_shell(config)

    assert not app.exception
    assert app.error[0].value == MESSAGE_TYPES
    assert app.error[0].value not in (MESSAGE_VALUES, MESSAGE_MISMATCH)
    assert f"'{column}': '{dtype} against " in caplog.text


def test_load_frames_resumes_on_an_input_whose_types_were_spelled_otherwise(project):
    """What the reconciliation buys, on the path the app actually takes.

    The output is written once and read back a release later, so the two frames meet
    under two spellings of one type on a library that moved underneath them. The run
    stopped there with the accumulated annotation intact and out of reach, and the
    rebuild the message prescribed reproduced the very type it regenerated by.

    The output is asserted byte for byte: a run that resumes by rewriting the gold
    standard under the new types is not the same outcome as one that reads it.
    """
    config, output = project
    folder = Path(config.work_dir) / "data" / "2023-01"
    started = make_input().drop(columns=[TEXT]).assign(**{ESTIMATE: ["oui", "", ""]})
    started.assign(**{COMMENT: ""}).to_parquet(output, index=False)
    before = output.read_bytes()
    make_input().astype({ID: "string", "pat_age": "Int64"}).to_parquet(
        folder / "2023-01_avc_annot_data_input-review.parquet"
    )

    app = run_shell(config)

    assert not app.exception
    assert not app.error
    assert app.text[0].value == "3/3"
    assert output.read_bytes() == before


def test_load_frames_refuses_a_reshaped_input_without_waiting_on_the_output_lock(
    project, caplog
):
    """The lock spans the build and the write, and the input guards stand ahead of it.

    Held inside it, a guard that opens nothing and reads a frame already in memory makes
    one annotator wait on another's save before being told their input is unusable. The
    wait is what this asserts: the run completes while the lock is held by someone else.

    It runs on its own thread against a timeout rather than in line, so a regression that
    puts the guard back under the lock fails here instead of hanging the suite.
    """
    config, output = project
    folder = Path(config.work_dir) / "data" / "2023-01"
    reshaped = make_input().drop(columns=[ID])
    reshaped.to_parquet(folder / "2023-01_avc_annot_data_input-review.parquet")

    holder = FileLock(f"{output}.lock")
    holder.acquire()

    stopped = threading.Event()
    seen = {}

    def load():
        seen["app"] = run_shell(config)
        stopped.set()

    reader = threading.Thread(target=load)
    reader.start()

    try:
        assert stopped.wait(5)
    finally:
        holder.release()
        reader.join(5)

    assert seen["app"].error[0].value == MESSAGE_COLUMNS
    assert f"The input does not carry ['{ID}']" in caplog.text
    assert not output.exists()


### SHELL ----------------------------------------------------------------------


def run_shell(config):
    def script(config):
        """Runs as a standalone script, so it imports what it uses itself."""
        import streamlit as st

        from edscrib.app import load_frames

        df_input, df_output, output_path = load_frames(config)

        st.text(f"{len(df_input)}/{len(df_output)}")
        st.text(str(output_path))

    return AppTest.from_function(script, kwargs={"config": config}).run()


### THE HALT OFF THE SCRIPT THREAD -------------------------------------------


def test_load_frames_stops_a_caller_that_is_not_a_script_run(project, caplog):
    """`load_frames` is public, so its guards cannot rest on who is calling it.

    `st.stop` requests a stop of the ambient run context and does nothing whatever
    without one, under a `NoReturn` annotation no checker sees through. Each guard being
    one statement followed by another, a consumer's own test calling this directly fell
    through all of them down to the write, and on two stems resolving to one file that
    replaced the raw input with the merged frame: the outcome the first guard exists to
    prevent, reached by the shortest path a consumer takes to the package.
    """
    config, _ = project
    collided = replace(config, output_stem=config.input_stem)
    path = (
        Path(config.work_dir)
        / "data"
        / "2023-01"
        / ("2023-01_avc_annot_data_input-review.parquet")
    )
    before = pd.read_parquet(path)

    with pytest.raises(BaseException) as raised:
        load_frames(collided)

    assert not isinstance(raised.value, Exception)
    assert pd.read_parquet(path).equals(before)
    assert str(path) in caplog.text


def test_the_halt_carries_no_path_to_the_page_off_the_script_thread(project, caplog):
    """The stop is structural, and what it renders is still the one opaque message.

    A guard that raised where it used to fall through would be a disclosure if the raise
    carried the detail: `client.showErrorDetails` paints an uncaught one into the browser
    whole. The detail stays in the log and the exception stays empty.
    """
    config, _ = project
    unresolvable = replace(config, proj=f"avc{USER}")

    with pytest.raises(BaseException) as raised:
        load_frames(unresolvable)

    assert not str(raised.value)
    assert str(config.work_dir) not in str(raised.value)
    assert USER in caplog.text


def test_load_frames_hands_back_the_output_path_it_validated(project):
    """`save_notes` takes a path, and this is where the only checked one exists.

    Derived from five fields of the configuration and put to the kernel against the
    input's, it would otherwise be derived again by whoever wires the save, off the same
    five fields and with none of that behind it.
    """
    config, output = project

    app = run_shell(config)

    assert not app.exception
    assert app.text[1].value == str(output)
    assert Path(app.text[1].value).exists()


def test_the_returned_path_follows_the_suffix_a_second_derivation_could_drop(project):
    """The divergence this closes is a field the caller forgets, so vary that field."""
    config, _ = project
    suffixed = replace(config, data_suffix="_v2")
    folder = Path(config.work_dir) / "data" / "2023-01"
    make_input().to_parquet(folder / "2023-01_avc_annot_data_input-review_v2.parquet")

    app = run_shell(suffixed)

    assert not app.exception
    assert app.text[1].value.endswith("2023-01_avc_annot_data_output-review_v2.parquet")


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

        _, df_output, _ = load_frames(config)

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


@pytest.mark.parametrize(
    ("scenario", "named"),
    [
        ("upstream", "['input', 'output']"),
        ("resumed", "['input']"),
        ("drifted", "['output']"),
    ],
)
def test_load_frames_names_which_frame_is_mislabelled(project, scenario, named, caplog):
    """One message covers two frames whose remedies differ, so the log forks it.

    Three states are reachable and each prescribes something else. A mislabelled input
    with nothing built yet carries its labels into the output `build_output` derives
    from it, and both are named: the remedy is upstream. Against an output already on
    the disk the same input is named alone, that output being resumed rather than
    derived. And an output whose labels drifted under a sound input is named alone in
    turn, where the remedy is to discard it and rebuild it from that input.

    Asserting the frame that is absent is what carries the fork: a line naming both
    every time would read the same and prescribe nothing.
    """
    config, output = project
    folder = Path(config.work_dir) / "data" / "2023-01"
    source = folder / "2023-01_avc_annot_data_input-review.parquet"
    started = make_input().drop(columns=[TEXT]).assign(**{ESTIMATE: "", COMMENT: ""})

    if scenario == "upstream":
        make_input().set_axis([2, 0, 1], axis=0).to_parquet(source)
    elif scenario == "resumed":
        make_input().set_axis([2, 0, 1], axis=0).to_parquet(source)
        started.to_parquet(output, index=False)
    else:
        started.set_axis([0, 1, 5], axis=0).to_parquet(output)

    app = run_shell(config)

    assert not app.exception
    assert app.error[0].value == MESSAGE_LABELS
    assert f"The rows of the {named} are not labelled by position" in caplog.text


def test_load_frames_stops_on_two_frames_designating_different_documents(project, caplog):
    """The alignment guard, whose stop had no coverage at the app's own boundary.

    The two row counts are what the page cannot carry and what forks the remedy: a
    cohort that changed size is not one that was reordered. The identifiers themselves
    are patient-linked and stay out of the log, as they do at every other guard.
    """
    config, output = project
    started = (
        make_input(rows=2).drop(columns=[TEXT]).assign(**{ESTIMATE: "", COMMENT: ""})
    )
    started.to_parquet(output, index=False)

    app = run_shell(config)

    assert not app.exception
    assert app.error[0].value == MESSAGE_MISMATCH
    assert "3 rows against 2" in caplog.text
    assert "d0" not in caplog.text


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
