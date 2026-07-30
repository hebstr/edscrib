"""Shaping an annotation frame into an Excel workbook, and gating its download."""

import logging
from io import BytesIO
from pathlib import Path
from typing import Literal

import pandas as pd
import streamlit as st
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


def _style_sheet(sheet: Worksheet) -> None:
    """Centre, border and widen every cell, then filter on the header row."""
    align = Alignment(horizontal="center", vertical="center")

    side = Side(style="thin", color="A5A5A5")

    border = Border(left=side, right=side, top=side, bottom=side)

    fill = PatternFill(start_color="E5E5E5", end_color="E5E5E5", fill_type="solid")

    for index, cells in enumerate(sheet.columns, start=1):
        length = max(len(str(cell.value)) for cell in cells)
        sheet.column_dimensions[get_column_letter(index)].width = (length + 2) * 1.1

    for row in sheet.iter_rows(
        min_row=1,
        max_col=sheet.max_column,
        max_row=sheet.max_row,
    ):
        for cell in row:
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
    """
    buffer = BytesIO()

    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        data.to_excel(writer, index=False, sheet_name=sheet_name)
        _style_sheet(writer.book[sheet_name])

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

    Neither input carries a `key`, so its value leaves the session when the dialog
    closes and the next opening asks again. `persist_state` is passed rather than
    inherited: the value is a password in clear, so an upstream change of the default
    has to fail a test rather than quietly keep it in server memory.

    The workbook is built once the pair matches and not before: it formats every cell
    of the frame, and each rerun above would otherwise pay for it, the ones where
    nobody has authenticated included.
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

    st.download_button(
        label=label,
        type="primary",
        width="stretch",
        data=build_workbook(data, filename),
        file_name=f"{filename}.xlsx",
        mime=_MIME,
        icon=icon,
    )


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
) -> None:
    """Render the button that opens the export dialog.

    `filename` names both the worksheet and the file the browser receives.

    `dialog_title` holds a single space because `st.dialog` takes a title and the form
    needs none: what it asks for is legible from its own two fields.
    """

    @st.dialog(title=dialog_title, width="medium")
    def export_dialog() -> None:
        download_form(path, data, filename=filename, label=label, info=info, icon=icon)

    if st.button(label=label, type=button_type, width="stretch", icon=icon):
        export_dialog()
