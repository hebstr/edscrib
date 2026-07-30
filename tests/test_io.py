"""Output creation on synthetic frames: the layer where a silent divergence would land."""

from pathlib import Path

import pandas as pd
import pytest

from edscrib.io import build_output, data_path

TEXT = "doc_text"
ESTIMATE = "note_estimate_a"
COMMENT = "note_comment_a"
FIELDS = (ESTIMATE, COMMENT)


def make_input(rows=3, **extra):
    frame = {
        "n": list(range(1, rows + 1)),
        "meta": ["m"] * rows,
        TEXT: ["<p>t</p>"] * rows,
    }
    return pd.DataFrame(frame | {name: [value] * rows for name, value in extra.items()})


@pytest.fixture
def output(tmp_path):
    return tmp_path / "output.parquet"


def test_data_path_follows_the_naming_convention():
    path = data_path("annot", "avc", "2023-01", "input-review")

    assert path == Path("annot/data/2023-01/2023-01_avc_annot_data_input-review.parquet")


def test_data_path_takes_a_suffix_and_an_extension():
    path = data_path(
        "annot", "avc", "2023-01", "output-review", suffix="-bonus", ext="xlsx"
    )

    assert path.name == "2023-01_avc_annot_data_output-review-bonus.xlsx"


def test_build_output_drops_the_text_column_and_appends_the_note_fields(output):
    data = build_output(make_input(), output, TEXT, FIELDS)

    assert list(data.columns) == ["n", "meta", ESTIMATE, COMMENT]
    assert (data[ESTIMATE] == "").all()


def test_build_output_keeps_note_columns_already_carried_by_the_input(output):
    df_input = make_input(**{ESTIMATE: "oui"})

    data = build_output(df_input, output, TEXT, FIELDS)

    assert list(data.columns) == ["n", "meta", ESTIMATE, COMMENT]
    assert (data[ESTIMATE] == "oui").all()


def test_build_output_resumes_an_output_holding_an_estimate(output):
    started = make_input().drop(columns=[TEXT]).assign(**{ESTIMATE: ["oui", "", ""]})
    started.to_parquet(output)

    data = build_output(make_input(), output, TEXT, FIELDS)

    assert data[ESTIMATE].tolist() == ["oui", "", ""]
    assert COMMENT in data.columns


def test_build_output_resumes_an_output_holding_only_a_comment(output):
    """A comment is committed work: an empty estimate leaves the output under way."""
    started = make_input().drop(columns=[TEXT]).assign(**{ESTIMATE: ""})
    started.assign(**{COMMENT: ["", "a revoir", ""]}).to_parquet(output)

    data = build_output(make_input(), output, TEXT, FIELDS)

    assert data[COMMENT].tolist() == ["", "a revoir", ""]


def test_build_output_leaves_a_reshaped_output_to_the_alignment_guard(output):
    """Starting over on it would open a fresh output over the accumulated one."""
    stale = pd.DataFrame({"gone": ["x"], ESTIMATE: [""]})
    stale.to_parquet(output)

    data = build_output(make_input(), output, TEXT, FIELDS)

    assert "gone" in data.columns
    assert len(data) == 1
