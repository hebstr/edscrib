"""The declarative shape two annotation apps reduce to, and what it derives."""

from dataclasses import fields, replace
from pathlib import Path

import pytest

from edscrib.config import AnnotConfig, FieldGroup, NoteField, Styles, Tuto

VALUES = ("oui", "peut-être", "non")


def pair(name, label):
    return (
        NoteField(f"note_estimate_{name}", f"AVC{label}", "radio", VALUES),
        NoteField(f"note_comment_{name}", f"COMMENTAIRE{label}", "text"),
    )


def make_config(groups, table_fields=("n",), **kwargs):
    return AnnotConfig(
        work_dir="annot",
        proj="avc",
        split="2023-01",
        input_stem="input-review",
        output_stem="output-review",
        secrets="annot/secrets.toml",
        styles=Styles(app=Path("style_app.css"), text=Path("style_text.css")),
        groups=groups,
        table_fields=table_fields,
        index_field="n",
        id_field="id_doc",
        text_field="doc_text",
        meta_fields={"pat_age": "ÂGE"},
        **kwargs,
    )


@pytest.fixture
def review():
    """One group whose names carry the authenticated user."""
    return make_config(
        groups=(FieldGroup(fields=pair("{user}", "")),),
        table_fields=("n", "note_estimate_{user}", "note_comment_{user}"),
        estimate_field="note_estimate_{user}",
    )


@pytest.fixture
def merge():
    """Two groups, the first displayed for reference and never written."""
    return make_config(
        groups=(
            FieldGroup(
                fields=(
                    *pair("fanny", "_FANNY"),
                    *pair("roberto", "_ROBERTO"),
                ),
                persisted=False,
            ),
            FieldGroup(fields=pair("merge", "_FINAL")),
        ),
        table_fields=("n", "note_estimate_fanny", "note_estimate_merge"),
        estimate_field="note_estimate_merge",
    )


### RESOLUTION -----------------------------------------------------------------


def test_resolve_substitutes_the_user_in_field_names(review):
    resolved = review.resolve("fanny")

    assert [f.name for f in resolved.rendered] == [
        "note_estimate_fanny",
        "note_comment_fanny",
    ]


def test_resolve_substitutes_the_user_in_table_fields(review):
    resolved = review.resolve("fanny")

    assert resolved.table_fields == ("n", "note_estimate_fanny", "note_comment_fanny")


def test_resolve_leaves_a_literal_name_untouched(merge):
    resolved = merge.resolve("someone")

    assert [f.name for f in resolved.rendered] == [
        "note_estimate_fanny",
        "note_comment_fanny",
        "note_estimate_roberto",
        "note_comment_roberto",
        "note_estimate_merge",
        "note_comment_merge",
    ]


def test_resolve_returns_a_config_so_derivations_need_no_user(review):
    resolved = review.resolve("fanny")

    assert isinstance(resolved, AnnotConfig)
    assert resolved.resolve("other").table_fields == resolved.table_fields


### DERIVATIONS ----------------------------------------------------------------


def test_rendered_follows_declaration_order_across_groups(merge):
    labels = [f.label for f in merge.rendered]

    assert labels == [
        "AVC_FANNY",
        "COMMENTAIRE_FANNY",
        "AVC_ROBERTO",
        "COMMENTAIRE_ROBERTO",
        "AVC_FINAL",
        "COMMENTAIRE_FINAL",
    ]


def test_persisted_fields_holds_only_the_fields_of_persisted_groups(merge):
    assert merge.persisted_fields == ("note_estimate_merge", "note_comment_merge")


def test_persisted_fields_covers_every_field_when_no_group_opts_out(review):
    resolved = review.resolve("fanny")

    assert resolved.persisted_fields == ("note_estimate_fanny", "note_comment_fanny")


def test_rows_pair_each_radio_with_its_comment(merge):
    assert [tuple(f.name for f in row) for row in merge.rows] == [
        ("note_estimate_fanny", "note_comment_fanny"),
        ("note_estimate_roberto", "note_comment_roberto"),
        ("note_estimate_merge", "note_comment_merge"),
    ]


def test_export_fields_are_the_index_the_metadata_and_the_persisted(merge):
    assert merge.export_fields == (
        "n",
        "pat_age",
        "note_estimate_merge",
        "note_comment_merge",
    )


def test_export_fields_names_a_column_declared_twice_only_once(merge):
    """A caller selects on this, and a repeated label selects the column twice.

    The two guards reading a column list fold theirs into a set, so neither ever saw
    it; the workbook takes the frame as it comes.
    """
    doubled = replace(merge, meta_fields={merge.index_field: "N°", "pat_age": "ÂGE"})

    assert doubled.export_fields == (
        "n",
        "pat_age",
        "note_estimate_merge",
        "note_comment_merge",
    )


def test_the_row_counter_and_the_document_id_are_distinct_columns(merge):
    """The table and the export key on the counter, the alignment guard on the id."""
    assert merge.index_field in merge.export_fields
    assert merge.id_field not in merge.export_fields


### CONSTRUCTION ---------------------------------------------------------------


def test_the_configuration_refuses_a_positional_construction(review):
    """Four of the twenty-three fields are consecutive column names of one type.

    Transposing the counter and the identifier is silent all the way down: the export
    carries the identifier the counter is there to keep out of it, and both identity
    guards compare a row counter against itself and pass on any two frames of equal
    length. A consumer meets this in its own type checker; the raise is what pins it.
    """
    declared = [getattr(review, field.name) for field in fields(review)]

    with pytest.raises(TypeError, match="positional"):
        AnnotConfig(*declared)  # pyrefly: ignore[missing-argument]


def test_the_tutorial_names_its_two_files(review):
    """The pair is named rather than positional, which is the whole of the type.

    A bare two-tuple says which file comes first and nothing about which is which, so a
    transposition hands a markdown file to a video player and the bytes of a recording
    to a markdown renderer, at the click and inside a dialog.
    """
    config = replace(review, tuto=Tuto(text=Path("tuto.md"), media=Path("tuto.webm")))

    assert config.tuto is not None
    assert config.tuto.text.suffix == ".md"
    assert config.tuto.media.suffix == ".webm"


def test_the_tutorial_refuses_a_positional_pair():
    """Two consecutive fields of one type, the same reason the configuration is kw-only.

    It costs nothing here, where no consumer constructs one yet, and it is what makes the
    two roles unmistakable at the only place they are written.
    """
    files = (Path("tuto.md"), Path("tuto.webm"))

    with pytest.raises(TypeError, match="positional"):
        Tuto(*files)  # pyrefly: ignore[missing-argument, unexpected-positional-argument]


def test_a_configuration_declares_no_tutorial_by_default(review):
    assert review.tuto is None


def test_the_stylesheets_refuse_a_positional_pair():
    """Two consecutive fields of one type, the third pair to be keyword-only for it.

    A transposition renders: the page chrome lands inside the frame holding a clinical
    note, and the note's own rules reach every widget of the app. Nothing raises, and
    what a deployment sees is a page that looks wrong for no stated reason.
    """
    sheets = (Path("style_app.css"), Path("style_text.css"))

    with pytest.raises(TypeError, match="positional"):
        Styles(*sheets)  # pyrefly: ignore[missing-argument, unexpected-positional-argument]


### THE ESTIMATE ---------------------------------------------------------------


def test_the_estimate_offers_the_answers_its_own_field_declares(merge):
    """Five derivations read this field, and one of them asks what it offers."""
    assert merge.estimate_options == VALUES


@pytest.mark.parametrize(
    "name",
    ["note_comment_merge", "note_estimate_fanny", "note_estimate_absent"],
    ids=["a comment", "a reference radio", "no field at all"],
)
def test_the_configuration_refuses_an_estimate_that_is_not_a_persisted_radio(merge, name):
    """The one field that is a column name and a role at once, so it is checked.

    Five derivations read it: where the cursor opens, whether the save button is live,
    what the gauge shows, whether the export reads as complete, and which rows the
    export carries. None of them can tell a wrong name from a right one, and two answer
    silently: a comment field makes every document read as answered, and a reference
    radio measures the run by an annotator who is not the one at the keyboard.

    Declared rather than taken as the first persisted field, which is why the check
    exists at all: a derivation off a position needs no check and is wrong for free.
    """
    with pytest.raises(ValueError, match="radio of a persisted group"):
        replace(merge, estimate_field=name)


### GUARD ----------------------------------------------------------------------


def test_a_group_accepts_one_row():
    assert len(FieldGroup(fields=pair("a", "")).fields) == 2


def test_a_group_accepts_several_rows():
    fields = (*pair("a", ""), *pair("b", ""))

    assert len(FieldGroup(fields=fields).fields) == 4


@pytest.mark.parametrize(
    ("kind", "options", "message"),
    [("radio", (), "got none"), ("text", VALUES, "got: oui")],
    ids=["radio-without-options", "text-with-options"],
)
def test_a_field_whose_options_do_not_match_its_kind_is_rejected(kind, options, message):
    """The default empty tuple is right for one kind and never for the other.

    A radio declared without options renders and holds `None`, a third value beside
    the empty sentinel and a real answer, which is what a save would write.
    """
    with pytest.raises(ValueError, match=message):
        NoteField("f", "L", kind, options)


def test_an_empty_group_is_rejected():
    """The pairing holds vacuously on no field, both slices being empty."""
    with pytest.raises(ValueError, match="none"):
        FieldGroup(fields=())


@pytest.mark.parametrize(
    ("label", "kinds"),
    [
        ("odd count", ("radio", "text", "radio")),
        ("two radios in a row", ("radio", "radio")),
        ("comment before its radio", ("text", "radio")),
    ],
    ids=["odd-count", "two-radios", "inverted"],
)
def test_a_group_that_is_not_a_sequence_of_rows_is_rejected(label, kinds):
    fields = tuple(
        NoteField(f"f{i}", "L", kind, VALUES if kind == "radio" else ())
        for i, kind in enumerate(kinds)
    )

    with pytest.raises(ValueError, match="row"):
        FieldGroup(fields=fields)
