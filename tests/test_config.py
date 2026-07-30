"""The declarative shape two annotation apps reduce to, and what it derives."""

import pytest

from edscrib.config import AnnotConfig, FieldGroup, NoteField

VALUES = ("oui", "peut-être", "non")


def pair(name, label, *, editable=True):
    return (
        NoteField(f"note_estimate_{name}", f"AVC{label}", "radio", VALUES, editable),
        NoteField(f"note_comment_{name}", f"COMMENTAIRE{label}", "text", (), editable),
    )


def make_config(groups, table_fields=("n",), **kwargs):
    return AnnotConfig(
        work_dir="annot",
        proj="avc",
        split="2023-01",
        input_suffix="input-review",
        output_suffix="output-review",
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
    )


@pytest.fixture
def merge():
    """Two groups, the first displayed for reference and never written."""
    return make_config(
        groups=(
            FieldGroup(
                fields=(
                    *pair("fanny", "_FANNY", editable=False),
                    *pair("roberto", "_ROBERTO", editable=False),
                ),
                persisted=False,
            ),
            FieldGroup(fields=pair("merge", "_FINAL")),
        ),
        table_fields=("n", "note_estimate_fanny", "note_estimate_merge"),
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


def test_persisted_holds_only_the_fields_of_persisted_groups(merge):
    assert merge.persisted == ("note_estimate_merge", "note_comment_merge")


def test_persisted_covers_every_field_when_no_group_opts_out(review):
    resolved = review.resolve("fanny")

    assert resolved.persisted == ("note_estimate_fanny", "note_comment_fanny")


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


def test_the_row_counter_and_the_document_id_are_distinct_columns(merge):
    """The table and the export key on the counter, the alignment guard on the id."""
    assert merge.index_field in merge.export_fields
    assert merge.id_field not in merge.export_fields


def test_a_non_persisted_group_can_be_declared_read_only(merge):
    reference = [f for f in merge.rendered if f.name.endswith(("fanny", "roberto"))]

    assert not any(f.editable for f in reference)


### GUARD ----------------------------------------------------------------------


def test_a_group_accepts_one_row():
    assert len(FieldGroup(fields=pair("a", "")).fields) == 2


def test_a_group_accepts_several_rows():
    fields = (*pair("a", ""), *pair("b", ""))

    assert len(FieldGroup(fields=fields).fields) == 4


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
    fields = tuple(NoteField(f"f{i}", "L", kind) for i, kind in enumerate(kinds))

    with pytest.raises(ValueError, match="row"):
        FieldGroup(fields=fields)
