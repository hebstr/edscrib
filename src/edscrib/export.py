"""Shaping an annotation frame into an Excel workbook, and gating its download."""

import logging
from io import BytesIO
from pathlib import Path
from typing import Literal

import pandas as pd
import streamlit as st
from openpyxl.cell.cell import Cell
from openpyxl.styles import Alignment, Border, PatternFill, Side
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.worksheet import Worksheet

from edscrib.auth import (
    LABEL_PASSWORD,
    LABEL_USERNAME,
    MESSAGE_REJECTED,
    MESSAGE_UNAVAILABLE,
    SecretsError,
    verify_credentials,
)

_logger = logging.getLogger(__name__)

_MIME = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"

_MAX_COLUMN_WIDTH = 60

_MAX_SHEET_NAME = 31

_MAX_CELL_LENGTH = 32767


def _finish_sheet(sheet: Worksheet) -> None:
    """Type, centre, border and widen every cell, then filter on the header row.

    A cell whose text opens on an equals sign is written as a formula, which is what
    `openpyxl` makes of any such string, and it is demoted back to text here. An
    annotation comment reading `=> voir le compte-rendu` otherwise reaches the operator
    as `#NAME?` and reads back as nothing at all, silently, on the one field the
    annotation exists to collect. The demotion holds because a text-typed cell is
    written without a formula element, and it is what decides evaluation rather than
    the literal content.

    It happens in this pass rather than its own: the traversal is the expensive half of
    the export, and there is no reason to pay for it twice. What the traversal yields is
    not only cells: a frame carrying a multi-level header merges its top row, and a
    merged position is a placeholder whose type is fixed and cannot be assigned to.

    A width is capped, the columns being widened holding free text no length bounds.
    The schema puts no ceiling on the attribute, but Office requires it between 0 and
    255 (MS-OI29500 18.3.1.13), which a comment of some two hundred and thirty
    characters already exceeds, and nothing here validates what it writes: the file
    then leaves the deployment carrying a width outside what its reader accepts.
    The cap sits well under that, a column as wide as Office allows taking a screen of
    its own and leaving a reviewer nothing to scan; what it clips stays in the cell,
    and a reader widens the column back or reads the value where it is entered.
    """
    align = Alignment(horizontal="center", vertical="center")

    side = Side(style="thin", color="A5A5A5")

    border = Border(left=side, right=side, top=side, bottom=side)

    fill = PatternFill(start_color="E5E5E5", end_color="E5E5E5", fill_type="solid")

    for index, cells in enumerate(sheet.columns, start=1):
        length = max(len(str(cell.value)) for cell in cells)
        width = min((length + 2) * 1.1, _MAX_COLUMN_WIDTH)
        sheet.column_dimensions[get_column_letter(index)].width = width

    for row in sheet.iter_rows(
        min_row=1,
        max_col=sheet.max_column,
        max_row=sheet.max_row,
    ):
        for cell in row:
            if isinstance(cell, Cell) and cell.data_type == "f":
                cell.data_type = "s"

            cell.alignment = align
            cell.border = border

    for cell in sheet[1]:
        cell.fill = fill

    sheet.auto_filter.ref = f"A1:{get_column_letter(sheet.max_column)}{sheet.max_row}"


def build_workbook(data: pd.DataFrame, sheet_name: str) -> BytesIO:
    """Write a frame into an in-memory xlsx, formatted, and rewind the buffer.

    The index is not written: what an annotation output carries in its columns is the
    export, and a row position is not one of them.

    The buffer is rewound because the caller hands it to a reader that starts where it
    is left, and a workbook read from its end is an empty download.

    The sheet is taken by creation order rather than by the name asked for, which is
    not the name it necessarily carries: the writer titles it before renaming it, and
    `openpyxl` renames a title colliding case-insensitively with one already in the
    book, its own default title included. Asking for `sheet` therefore yields `sheet1`
    and a lookup by name that finds nothing. One frame is written into a writer that
    holds no other sheet, so the last is the one just written.

    Two limits are checked here rather than left to `openpyxl`, which passes both of
    them and says so in a `UserWarning` nobody is listening for: a title over
    thirty-one characters it keeps, against a workbook the spreadsheet may then refuse,
    and a value over the cell limit it silently cuts, against a comment that leaves the
    deployment shorter than the one the annotator wrote. Both are refusals rather than
    repairs, the export being a copy of a gold standard and a copy that quietly differs
    being worse than one that does not arrive. Only the columns pandas holds as objects
    are measured, which is where a string lands, `str` included; no number or timestamp
    reaches that length once rendered.
    """
    if len(sheet_name) > _MAX_SHEET_NAME:
        raise ValueError(
            f"A sheet name takes at most {_MAX_SHEET_NAME} characters, "
            f"and this one takes {len(sheet_name)} characters"
        )

    for name, dtype in data.dtypes.items():
        if dtype.kind != "O":
            continue

        longest = data[name].astype("string").str.len().max()

        if pd.notna(longest) and longest > _MAX_CELL_LENGTH:
            raise ValueError(
                f"A cell takes at most {_MAX_CELL_LENGTH} characters, "
                f"and column {name} holds one of {longest} characters"
            )

    buffer = BytesIO()

    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        data.to_excel(writer, index=False, sheet_name=sheet_name)
        _finish_sheet(writer.book.worksheets[-1])

    buffer.seek(0)

    return buffer


def download_form(
    path: str | Path,
    data: pd.DataFrame,
    *,
    filename: str,
    label: str,
    info: str = "",
    icon: str = ":material/download:",
) -> None:
    """Ask for a pair, and serve the export once it matches the deployment secrets.

    This is the body of the dialog `download` opens, and it sits at module level
    because `AppTest` does not run the body of an `st.dialog` (streamlit#9786): held
    inside the decorated shell, none of it would be reachable by a test.

    The pair is checked against the same secrets file as the login gate, so taking a
    gold standard out of the deployment is a deliberate second gate rather than one
    click away in a tab left open. Which is also its cadence: the file is read on every
    rerun for as long as a password is typed, so a file that cannot be read withholds
    the button and never raises. An operator editing it to add a colleague would
    otherwise paint a traceback over the dialog of an annotator mid-session.

    What the gate holds back is the link and not the bytes. Rendering the button puts
    the workbook in Streamlit's media store and serves it at a URL its endpoint answers
    with no session, cookie or credential of any kind: the path is a 128-bit digest of
    the content, so nobody reaches it by guessing, and it stops answering once no
    session renders the button any more. Until then it is whoever holds it, which is
    the browser history of the annotator and any proxy the deployment sits behind. The
    package cannot close this, Streamlit exposing no other way to serve a download, so
    the operator reads it here rather than assuming the gate covers more than it does.

    Neither input carries a `key`, so its value leaves the session when the dialog
    closes and the next opening asks again. `persist_state` is passed rather than
    inherited: the value is a password in clear, so an upstream change of the default
    has to fail a test rather than quietly keep it in server memory.

    The workbook is built once the pair matches and not before: it formats every cell
    of the frame, and each rerun above would otherwise pay for it, the ones where
    nobody has authenticated included. The button does not rerun on its click for the
    same reason: the download takes the file the render already registered, so the
    rerun a click fires by default reaches this line again and pays a second time for
    a workbook nobody collects. It is left eager rather than deferred to the click,
    which would build once but hand the failure to Streamlit's own handler, and with
    it an English message and a log line the operator does not read.

    That build is the second boundary of this shell, and it catches broadly for what
    it holds rather than for what it expects: `openpyxl` refuses a cell carrying a
    control character, which clinical text pasted out of a PDF or an export does carry,
    and it names the whole cell in the message. Uncaught, `client.showErrorDetails`
    paints that message and the server's paths into the browser. The detail goes to the
    module logger, where the operator reads the offending cell, against the one message
    the secrets failure already renders: neither is anything the annotator can act on.
    Streamlit's own control flow descends from `BaseException` and passes through.
    """
    username = st.text_input(label=LABEL_USERNAME, persist_state=None)

    password = st.text_input(label=LABEL_PASSWORD, type="password", persist_state=None)

    if not password:
        return

    try:
        granted = verify_credentials(path, username or "", password)
    except SecretsError:
        _logger.exception("Secrets unusable while checking an export request")
        st.error(MESSAGE_UNAVAILABLE)
        return

    if not granted:
        st.error(MESSAGE_REJECTED)
        return

    if info:
        st.markdown(info)

    st.space("stretch")

    try:
        st.download_button(
            label=label,
            type="primary",
            width="stretch",
            data=build_workbook(data, filename),
            file_name=f"{filename}.xlsx",
            mime=_MIME,
            icon=icon,
            on_click="ignore",
        )
    except Exception:
        _logger.exception("The export could not be built")
        st.error(MESSAGE_UNAVAILABLE)


def download(
    path: str | Path,
    data: pd.DataFrame,
    *,
    filename: str,
    label: str,
    info: str = "",
    icon: str = ":material/download:",
    button_type: Literal["primary", "secondary", "tertiary"] = "secondary",
    dialog_title: str = " ",
    key_button: str = "button-export",
) -> None:
    """Render the button that opens the export dialog.

    `filename` names both the worksheet and the file the browser receives.

    `dialog_title` holds a single space because `st.dialog` takes a title and the form
    needs none: what it asks for is legible from its own two fields.

    The button carries a key so that a page offering the export twice, in a sidebar and
    again in its body, names which of the two to move rather than failing on an
    identifier Streamlit derived from parameters the two calls share. The form the
    dialog opens takes none: only one dialog is ever open, and the two fields it holds
    are keyless on purpose, that being what takes the password out of the session when
    it closes.

    What the export carries is the frame of the run the button was clicked in, not of
    the moment it is downloaded. `st.dialog` is a fragment, and a fragment rerun calls
    the body Streamlit stored rather than this function again, so every rerun the open
    dialog drives re-enters a closure holding that frame. Nothing the annotator does
    can move it, the dialog being modal for them, but the output is shared: a colleague
    saving meanwhile is not in what leaves. Reopening the dialog is what takes it.
    """

    @st.dialog(title=dialog_title, width="medium")
    def export_dialog() -> None:
        download_form(path, data, filename=filename, label=label, info=info, icon=icon)

    if st.button(
        label=label, type=button_type, width="stretch", icon=icon, key=key_button
    ):
        export_dialog()
