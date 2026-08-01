"""The render half of an annotation app: the page `run_app` draws from a configuration.

The shell tests go through `AppTest.from_function`, which runs the function body as a
script, so the script defined here imports what it uses and takes its configuration
through `kwargs`. What `st.dialog` wraps is not run by `AppTest` (streamlit#9786), so the
tutorial's own body is exercised as the plain function it is.
"""

import re
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from edscrib.app import (
    complete,
    cursor_seed,
    exportable,
    moved_to,
    stylesheet,
    tuto_form,
)
from edscrib.config import AnnotConfig, FieldGroup, NoteField, Styles, Tuto
from edscrib.messages import (
    LABEL_PASSWORD,
    LABEL_TUTO,
    LABEL_USERNAME,
    MESSAGE_UNAVAILABLE,
)

ID = "id_doc"
TEXT = "doc_text"
ESTIMATE = "note_estimate_{user}"
COMMENT = "note_comment_{user}"
ANSWERED = "note_estimate_admin"
VALUES = ("oui", "non")
REFERENCE = ("note_estimate_ref", "note_comment_ref")


def make_input(rows=3, **extra):
    frame = {
        "n": list(range(1, rows + 1)),
        ID: [f"d{i}" for i in range(rows)],
        "pat_age": [70] * rows,
        "extract": ["un extrait"] * rows,
        TEXT: ["<p>compte rendu</p>"] * rows,
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
        footer_fields={"EXTRACTION": "extract"},
        column_widths={"n": 40, "pat_age": 300},
    )


@pytest.fixture
def project(tmp_path):
    """A deployment as the render meets it: an input, two stylesheets, a secrets file."""
    folder = tmp_path / "data" / "2023-01"
    folder.mkdir(parents=True)
    make_input(**dict.fromkeys(REFERENCE, "oui")).to_parquet(
        folder / "2023-01_avc_annot_data_input-review.parquet"
    )
    (tmp_path / "style_app.css").write_text(".page {}", encoding="utf-8")
    (tmp_path / "style_text.css").write_text("p {}", encoding="utf-8")
    (tmp_path / "secrets.toml").write_text(
        '[users]\nfanny = "mot de passe"\n', encoding="utf-8"
    )
    output = folder / "2023-01_avc_annot_data_output-review.parquet"
    return make_config(tmp_path), output


def started(output, answers):
    """Put an output on the disk carrying the answers a prior run accumulated."""
    frame = make_input(**dict.fromkeys(REFERENCE, "oui")).drop(columns=[TEXT])
    frame = frame.assign(**{ANSWERED: answers, "note_comment_admin": [""] * len(answers)})
    frame.to_parquet(output, index=False)


def script(config):
    """Runs as a standalone script, so it imports what it uses itself."""
    from edscrib.app import run_app

    run_app(config)


def run(config, **session):
    app = AppTest.from_function(script, kwargs={"config": config})

    for key, value in session.items():
        app.session_state[key] = value

    return app.run()


def sidebar_kinds(app):
    return [getattr(item, "type", "") for item in app.sidebar.children.values()]


def framed_document(app):
    """What the frame holding the clinical note carries, stylesheet and all."""
    for block in app.main.children.values():
        for column in getattr(block, "children", {}).values():
            for element in getattr(column, "children", {}).values():
                if getattr(element, "type", "") == "iframe":
                    return element.proto.srcdoc

    raise AssertionError("no frame holds the document")


def container_keys(app):
    """The container names Streamlit turns into the classes a stylesheet hooks on."""
    return [
        getattr(getattr(item, "proto", None), "id", "").split("-", 2)[-1]
        for item in app.sidebar.children.values()
    ]


### STYLESHEET -----------------------------------------------------------------


def test_the_stylesheet_is_wrapped_in_the_tag_that_carries_it(tmp_path):
    sheet = tmp_path / "style.css"
    sheet.write_text(".page { color: red; }", encoding="utf-8")

    assert stylesheet(sheet) == "<style>.page { color: red; }</style>"


### CURSOR ---------------------------------------------------------------------


def test_the_cursor_opens_on_the_last_document_answered():
    """On the last answer rather than past it, so the annotator sees what it was."""
    data = make_input(4).assign(**{ANSWERED: ["oui", "non", "", ""]})

    assert cursor_seed(data, ANSWERED) == 1


def test_the_cursor_opens_on_the_first_document_when_nothing_is_answered():
    data = make_input(4).assign(**{ANSWERED: [""] * 4})

    assert cursor_seed(data, ANSWERED) == 0


### EXPORT ---------------------------------------------------------------------


def make_exportable(tmp_path, answers, excluded=()):
    config = replace(
        make_config(tmp_path),
        estimate_field=ANSWERED,
        groups=(
            FieldGroup(
                fields=(
                    NoteField(ANSWERED, "AVC", "radio", VALUES),
                    NoteField("note_comment_admin", "COMMENTAIRE", "text"),
                ),
            ),
        ),
        export_excluded=excluded,
    )
    data = make_input(len(answers)).assign(
        **{ANSWERED: answers, "note_comment_admin": [""] * len(answers)}
    )
    return exportable(config, data)


def test_the_export_drops_a_document_with_no_answer(tmp_path):
    """The sentinel is the absence of an answer, and an export is a gold standard."""
    kept = make_exportable(tmp_path, ["oui", "", "non"])

    assert list(kept["n"]) == [1, 3]


def test_the_export_drops_an_answer_the_deployment_excludes(tmp_path):
    kept = make_exportable(tmp_path, ["oui", "non", "oui"], excluded=("non",))

    assert list(kept["n"]) == [1, 3]


def test_the_export_needs_no_declaration_to_drop_the_unanswered(tmp_path):
    """A deployment excluding one answer does not have to remember the sentinel."""
    kept = make_exportable(tmp_path, ["", "non", "oui"], excluded=("non",))

    assert list(kept["n"]) == [3]


def test_the_export_carries_the_declared_columns_and_no_other(tmp_path):
    """The counter, the metadata and the persisted fields, and never the identifier."""
    kept = make_exportable(tmp_path, ["oui"])

    assert list(kept.columns) == ["n", "pat_age", ANSWERED, "note_comment_admin"]
    assert ID not in kept.columns


def test_an_excluded_answer_carrying_an_apostrophe_still_excludes(tmp_path):
    """A membership test where a `query` expression would fail at its own parse.

    The excluded values go into a string that is then parsed, and a French deployment
    writes its classification levels by hand: an apostrophe closes the literal early and
    the export fails inside the render, on a label nobody typed wrong.
    """
    kept = make_exportable(tmp_path, ["pas d'AVC", "oui"], excluded=("pas d'AVC",))

    assert list(kept["n"]) == [2]


### COMPLETION -----------------------------------------------------------------


def make_complete(tmp_path, answers):
    config = replace(
        make_config(tmp_path),
        estimate_field=ANSWERED,
        groups=(
            FieldGroup(
                fields=(
                    NoteField(ANSWERED, "AVC", "radio", VALUES),
                    NoteField("note_comment_admin", "COMMENTAIRE", "text"),
                ),
            ),
        ),
    )
    return complete(config, make_input(len(answers)).assign(**{ANSWERED: answers}))


def test_a_run_reads_as_complete_when_every_document_carries_an_offered_answer(tmp_path):
    assert make_complete(tmp_path, ["oui", "non", "oui"])


def test_a_run_reads_as_incomplete_while_one_document_is_unanswered(tmp_path):
    assert not make_complete(tmp_path, ["oui", "", "oui"])


### TUTORIAL BODY --------------------------------------------------------------


def test_the_tutorial_renders_its_text_and_its_media(tmp_path):
    """The body of the dialog, which `AppTest` does not run, tested as a function."""
    tuto = Tuto(text=tmp_path / "tuto.md", media=tmp_path / "tuto.webm")
    tuto.text.write_text("# Comment annoter", encoding="utf-8")
    tuto.media.write_bytes(b"\x00\x01")

    def show(tuto):
        from edscrib.app import tuto_form

        tuto_form(tuto)

    app = AppTest.from_function(show, kwargs={"tuto": tuto}).run()

    assert not app.exception
    assert app.markdown[0].value == "# Comment annoter"


def test_the_tutorial_reads_its_text_as_utf8(tmp_path):
    """A tutorial is written for a French-speaking annotator, accents and all."""
    tuto = Tuto(text=tmp_path / "tuto.md", media=tmp_path / "tuto.webm")
    tuto.text.write_text("Répétez l'opération", encoding="utf-8")
    tuto.media.write_bytes(b"\x00")

    def show(tuto):
        from edscrib.app import tuto_form

        tuto_form(tuto)

    app = AppTest.from_function(show, kwargs={"tuto": tuto}).run()

    assert app.markdown[0].value == "Répétez l'opération"
    assert tuto_form is not None


### THE PAGE -------------------------------------------------------------------


def test_the_page_renders_without_reaching_the_browser(project):
    config, _ = project

    app = run(config)

    assert not app.exception
    assert not app.error


def test_the_session_carries_the_cursor_and_the_count_of_saves(project):
    """`navigation` stops the app without them, and it is this render that seeds them."""
    config, _ = project

    app = run(config)

    assert app.session_state["doc_index"] == 0
    assert app.session_state["save_count"] == 0


def test_the_run_opens_on_the_document_last_answered(project):
    config, output = project
    started(output, ["oui", "non", ""])

    app = run(config)

    assert app.session_state["doc_index"] == 1


def test_the_cursor_is_seeded_once_and_never_reset_under_the_annotator(project):
    """A seed and not an assignment: it answers where to open, not where to stay.

    Reseeding on every rerun puts the cursor back on the last answered document at the
    click after each arrow, so the run cannot be moved through at all, and the two
    positions agree often enough that a fixture opening on the seed sees nothing.
    """
    config, output = project
    started(output, ["oui", "non", ""])

    app = run(config, doc_index=2, save_count=0)

    assert app.session_state["doc_index"] == 2
    assert app.slider[0].value == 3


def test_the_page_is_drawn_on_the_document_the_cursor_points_at(project):
    """The cursor is a position, and everything the page shows is read at that position.

    The counter is displayed one-based because that is what a person counts from, and it
    is never read back out of the frame to make a position: the slider offers positions
    and the table hands back the row it was clicked on.
    """
    config, output = project
    started(output, ["oui", "non", ""])

    app = run(config, doc_index=1, save_count=0)

    assert app.radio[0].value == "non"
    assert app.slider[0].value == 2


def test_the_document_comes_from_the_input_and_the_metadata_from_the_output(project):
    """The text is the one column the output does not carry, and the only one."""
    config, _ = project

    app = run(config)

    assert list(app.dataframe[0].value.columns) == ["ÂGE"]
    assert list(app.dataframe[0].value["ÂGE"]) == [70]


def test_the_document_is_framed_with_the_stylesheet_written_for_it(project):
    """A clinical note is markup no page-level sheet reaches, and it is not the app's.

    Rendering it into the page would let a note style the widgets around it, so it goes
    into a frame of its own carrying the sheet written for a document.
    """
    config, _ = project

    assert framed_document(run(config)) == "<style>p {}</style><p>compte rendu</p>"


def test_the_summary_table_carries_the_columns_the_configuration_declares(project):
    config, _ = project

    app = run(config)

    assert list(app.dataframe[1].value.columns) == ["n", ANSWERED]


### WIDGET KEYS ----------------------------------------------------------------


def test_what_displays_the_whole_run_is_redrawn_on_every_save(project):
    """A save rewrites the column both of these display, across every document.

    Keyed on the cursor alone they would hand back the state of a widget drawn before
    the write, so the gauge and the table would answer for the run as it stood one
    annotation ago.
    """
    config, _ = project

    before = run(config, doc_index=0, save_count=0)
    after = run(config, doc_index=0, save_count=1)

    assert before.slider[1].key != after.slider[1].key
    assert before.dataframe[1].proto.id != after.dataframe[1].proto.id


def test_a_note_widget_is_redrawn_on_every_document(project):
    """The other half of the same distinction, and the one that reaches the file.

    A field keyed on the number of saves would hold one document's answer across a move
    to the next, and offer it for saving there.
    """
    config, _ = project

    here = run(config, doc_index=0, save_count=0)
    there = run(config, doc_index=1, save_count=0)

    assert here.radio[0].key != there.radio[0].key
    assert here.text_input[0].key != there.text_input[0].key


### ROW SELECTION --------------------------------------------------------------


def test_selecting_a_row_moves_the_cursor_onto_that_position():
    """The selection is a position, and the cursor takes it as it is."""
    assert moved_to([2], 0) == 2


def test_selecting_no_row_moves_the_cursor_nowhere():
    assert moved_to([], 1) is None


def test_selecting_the_document_already_shown_moves_the_cursor_nowhere():
    """What follows a move is a rerun, which would draw the same page on every pass."""
    assert moved_to([1], 1) is None


### FIELDS ---------------------------------------------------------------------


def two_groups(config):
    """A reconciliation shape: a reference group, then the group that is written."""
    return replace(
        config,
        groups=(
            FieldGroup(
                fields=(
                    NoteField(REFERENCE[0], "AVC_REF", "radio", VALUES, editable=False),
                    NoteField(REFERENCE[1], "COMM_REF", "text", editable=False),
                ),
                persisted=False,
            ),
            config.groups[0],
        ),
    )


def test_a_reference_group_renders_read_only(project):
    """The one interface change the descent takes on, and the reason the field exists.

    Those widgets were editable and never saved, and the session loop rewrote them from
    the frame on every rerun, so an answer typed into one was lost at the next click
    with nothing said.
    """
    config, _ = project

    app = run(two_groups(config))

    assert [item.proto.disabled for item in app.radio] == [True, False]
    assert [item.proto.disabled for item in app.text_input] == [True, False]


def test_the_fields_render_in_the_order_the_groups_declare_them(project):
    config, _ = project

    app = run(two_groups(config))

    assert [item.label for item in app.radio] == ["AVC_REF", "AVC"]
    assert [item.label for item in app.text_input] == ["COMM_REF", "COMMENTAIRE"]


def test_a_divider_separates_two_groups_and_never_splits_one(project):
    """The layout reads the groups, which is the second thing the type carries."""
    config, _ = project

    assert sidebar_kinds(run(two_groups(config))).count("divider") == 1
    assert sidebar_kinds(run(config)).count("divider") == 0


def test_a_radio_selects_nothing_on_an_answer_its_options_no_longer_offer(project):
    """A document annotated before a level was renamed, which the file still holds.

    The answer is displayed as no selection rather than raising, and it is still in the
    file: this is a display of what is there and not a correction of it.
    """
    config, output = project
    started(output, ["ancien", "", ""])

    app = run(config)

    assert app.radio[0].value is None
    assert pd.read_parquet(output)[ANSWERED][0] == "ancien"


def test_the_save_button_is_held_back_until_the_document_is_answered(project):
    config, output = project

    assert run(config).button[1].proto.disabled

    started(output, ["oui", "", ""])

    assert not run(config).button[1].proto.disabled


### GAUGE, EXPORT AND FOOTER ---------------------------------------------------


def test_the_gauge_counts_the_documents_answered(project):
    config, output = project
    started(output, ["oui", "non", ""])

    app = run(config)

    assert app.slider[1].value == 2


def test_the_download_container_names_the_visibility_declared(project):
    config, _ = project

    assert "button-download-hidden" in container_keys(run(config))
    assert "button-download-visible" in container_keys(
        run(replace(config, download_visible=True))
    )


def test_the_footer_renders_one_block_per_declared_field(project):
    config, _ = project

    assert sidebar_kinds(run(config)).count("html") == 1
    assert sidebar_kinds(run(replace(config, footer_fields={}))).count("html") == 0


def test_a_footer_column_the_data_stopped_carrying_is_named_on_the_way_in(
    project, caplog
):
    """The render reads it by name, so it is a declared column like the table's.

    Without it the lookup raises mid-page and arrives as the generic message, on a
    deployment that merely stopped producing one column.
    """
    config, _ = project

    app = run(replace(config, footer_fields={"EXTRACTION": "parti"}))

    assert not app.exception
    assert "The output does not carry ['parti']" in caplog.text


### THE TUTORIAL BUTTON --------------------------------------------------------


def with_tuto(config, work_dir, *, media=True):
    tuto = Tuto(text=work_dir / "tuto.md", media=work_dir / "tuto.webm")
    tuto.text.write_text("# tuto", encoding="utf-8")

    if media:
        tuto.media.write_bytes(b"\x00")

    return replace(config, tuto=tuto)


def test_no_tutorial_button_is_rendered_without_a_tutorial(project):
    config, _ = project

    assert LABEL_TUTO not in [item.label for item in run(config).button]


def test_the_tutorial_button_is_held_back_while_an_asset_is_missing(project):
    config, _ = project
    work_dir = Path(config.work_dir)

    app = run(with_tuto(config, work_dir, media=False))

    assert app.button[0].label == LABEL_TUTO
    assert app.button[0].proto.disabled


def test_the_tutorial_button_is_live_once_both_assets_are_in_place(project):
    config, _ = project

    app = run(with_tuto(config, Path(config.work_dir)))

    assert app.button[0].label == LABEL_TUTO
    assert not app.button[0].proto.disabled


### THE LOGIN GATE -------------------------------------------------------------


def test_the_login_gate_withholds_every_annotation_widget(project):
    """The gate runs first because the column names are bound to what it answers.

    What it renders instead is asserted beside what it withholds. A run that fell
    through to the boundary would withhold the same widgets and write the same nothing,
    and tell the annotator the service is out where the form was all they needed.
    """
    config, output = project

    app = run(replace(config, auth=True))

    assert not app.exception
    assert not app.error
    assert [item.label for item in app.text_input] == [LABEL_USERNAME, LABEL_PASSWORD]
    assert not app.radio
    assert not app.dataframe
    assert not output.exists()


def test_an_authenticated_session_annotates_under_its_own_columns(project):
    config, output = project

    app = run(replace(config, auth=True), password_correct=True, user="fanny")

    assert not app.exception
    assert not app.error
    assert "note_estimate_fanny" in pd.read_parquet(output).columns


def test_a_run_with_no_gate_annotates_under_the_anonymous_name(project):
    """The anonymous name is not cosmetic: it names the columns of the gold standard."""
    config, output = project

    app = run(config)

    assert not app.error
    assert ANSWERED in pd.read_parquet(output).columns


### THE DISPLAY BOUNDARY -------------------------------------------------------


def test_the_render_keeps_its_failure_off_the_page(project, caplog):
    """`client.showErrorDetails` defaults to `full`, so an uncaught raise is painted.

    The data path has its own boundary and this one covers everything after it, which
    is where a deployment path is read for the first time.
    """
    config, _ = project
    missing = replace(
        config, styles=Styles(app=Path(config.work_dir) / "absent.css", text=Path("t"))
    )

    app = run(missing)

    assert not app.exception
    assert app.error[0].value == MESSAGE_UNAVAILABLE
    assert "could not be rendered" in caplog.text


def test_the_rendered_message_names_nothing_about_the_failure(project, caplog):
    """The path is the operator's and goes to the log.

    What the page says is that the service is out, which is all a visitor is told.
    """
    config, _ = project
    missing = replace(
        config, styles=Styles(app=Path(config.work_dir) / "absent.css", text=Path("t"))
    )

    app = run(missing)

    assert "absent.css" not in app.error[0].value
    assert str(config.work_dir) not in app.error[0].value
    assert "absent.css" in caplog.text


def test_the_boundary_lets_the_framework_move_the_cursor():
    """`st.rerun` and `st.stop` raise from `BaseException`, which the clause must miss.

    A boundary catching them would swallow every cursor move the render itself asks for,
    and the guards of the data path leave through a stop of the same kind.
    """
    from streamlit.runtime.scriptrunner_utils.exceptions import (
        RerunException,
        StopException,
    )

    assert not issubclass(RerunException, Exception)
    assert not issubclass(StopException, Exception)


### THE PLACEHOLDER ------------------------------------------------------------


def test_the_estimate_is_bound_like_every_other_column_name(project):
    """It names a column, so leaving it unbound stops the app on the resolution guard."""
    config, _ = project

    resolved = config.resolve("fanny")

    assert resolved.estimate_field == "note_estimate_fanny"
    assert not re.search(r"\{user\}", resolved.estimate_field)
