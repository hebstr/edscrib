"""The body an annotation app reduces to, driven by its `AnnotConfig`.

Two halves. The data path resolves the two parquet paths, reads the input, builds or
resumes the output, and runs the guards that stand between a reshaped input and a
mislabelled gold standard. The render lays the page out from the same configuration and
calls the three modules that hold widgets of their own, so a consuming app is an import,
a configuration literal and one call to `run_app`.
"""

import dataclasses
import logging
from base64 import b64encode
from collections import Counter
from collections.abc import Iterable, Iterator, Mapping
from pathlib import Path
from typing import NoReturn

import pandas as pd
import streamlit as st
from filelock import FileLock

from edscrib.auth import login
from edscrib.config import ANONYMOUS, USER, AnnotConfig, NoteField, Tuto
from edscrib.export import download
from edscrib.io import EMPTY, build_output, data_path, read_data
from edscrib.messages import (
    LABEL_DOWNLOAD,
    LABEL_SAVE,
    LABEL_TUTO,
    MESSAGE_COLUMNS,
    MESSAGE_DUPLICATES,
    MESSAGE_LABELS,
    MESSAGE_MISMATCH,
    MESSAGE_POSITION,
    MESSAGE_TYPES,
    MESSAGE_UNAVAILABLE,
    MESSAGE_VALUES,
)
from edscrib.navigation import navigation

_COLUMNS_BODY = (3, 1)
_HEIGHT_DOCUMENT = 600
_COLUMNS_ROW = (1.1, 2.4)
_COLUMNS_NAVIGATION = (0.5, 3.5, 0.5)

_logger = logging.getLogger(__name__)


class _Stopped(BaseException):
    """The stop a guard leaves through, where `st.stop` only ever requests one.

    Descends from `BaseException` and not from `Exception`, as the runner's own
    `ScriptControlException` does and for the same reason: the data path catches
    `Exception` to keep a deployment path out of the browser, and a stop that clause
    swallowed would land back on the fall-through this exists to close.
    """


def _halt(message: str) -> NoReturn:
    """Render one message to the page and leave, whatever run context there is.

    `st.stop` is a request and not a raise. Its body asks the ambient context to stop
    and forces a yield point at which the runner raises, so on the script thread the
    line below is never reached; with no context at all it does nothing whatever and
    returns, under a `NoReturn` annotation the type checker believes either way. A guard
    is one statement followed by another, so a caller outside a script run fell through
    every one of them down to the write, and two stems resolving to one file then
    replaced the raw input with the merged frame: the outcome the first guard exists to
    prevent, reached by a function that is public and whose consumers write their own
    tests. The raise is what makes the stop a property of this module rather than a
    convention about who is allowed to call it.
    """
    st.error(message)
    st.stop()
    raise _Stopped


def _stamp(path: str | Path) -> tuple[int, int, int]:
    """What the input file looks like from the outside, without opening it.

    The inode is there because a modification time and a size are not an identity: a
    mount serving them at second granularity collapses two writes of equal size into
    one stamp, and the frame regenerated to answer a guard is then served from the
    cache exactly as the stale one was. A generator that stages and renames lands on a
    new inode, which no granularity collapses. One rewritten in place over the same
    inode still rests on the modification time alone.
    """
    info = Path(path).stat()
    return info.st_ino, info.st_mtime_ns, info.st_size


@st.cache_data(max_entries=1)
def _read_input(path: str | Path, stamp: tuple[int, int, int]) -> pd.DataFrame:
    """Read the input, keyed on the state of the file as well as on its path.

    `stamp` is never read. It is there to make a regenerated input a different entry:
    without it the first frame is served for the life of the server process, so the
    rebuild the guards below ask for takes effect only on a restart, while the app goes
    on rendering the message that asked for it. It carries no leading underscore, which
    would be the one spelling `st.cache_data` reads as "keep this out of the key".

    The path stays an argument rather than a closure, since the cache is global to the
    process and keyed on the qualified name: one app would otherwise serve another one
    its frame. One entry is kept, so a regenerated input evicts rather than doubles it.
    """
    return read_data(path)


def _strings(value: object) -> Iterator[str]:
    """Every string the value holds, however deeply the shape nests it."""
    if isinstance(value, str):
        yield value
    elif isinstance(value, Path):
        yield str(value)
    elif isinstance(value, Mapping):
        for key, item in value.items():
            yield from _strings(key)
            yield from _strings(item)
    elif dataclasses.is_dataclass(value) and not isinstance(value, type):
        for field in dataclasses.fields(value):
            yield from _strings(getattr(value, field.name))
    elif isinstance(value, Iterable):
        for item in value:
            yield from _strings(item)


def same_file(input_path: str | Path, output_path: str | Path) -> bool:
    """Whether the two paths designate one file, by their names or by the kernel's.

    The names alone are not the question. Both are built from one directory, project,
    split and suffix and differ in the stem, so two stems differing only in case name
    one file wherever the storage folds it, which a CIFS share mounted `nocase` does and
    a case-insensitive server does whatever the client asked for; and either name can be
    a link onto the other. The kernel's own identity, the device and the inode, answers
    all of those at once where a lexical comparison answers none, and normalizing the
    case would answer none either, `os.path.normcase` being a no-op on POSIX.

    What follows from the collision differs by the way it was reached and is unwanted
    every way. Folded onto one directory entry, the write replaces the raw input with
    the merged frame. Through a link, the replace takes the entry rather than its
    target, so the input survives and the gold standard keeps the clinical text
    `build_output` drops, which is what the export reads its columns out of. Either way
    every guard downstream compares the file against itself and passes.

    The kernel is asked only where both names exist, an absent one designating nothing
    to collide with. That leaves a dangling link answering `False` here, which is right:
    nothing is built yet, and the read behind it fails into the boundary.
    """
    paths = (Path(input_path), Path(output_path))

    if paths[0] == paths[1]:
        return True

    return all(path.exists() for path in paths) and paths[0].samefile(paths[1])


def unresolved_fields(config: AnnotConfig) -> list[str]:
    """Every string the configuration carries that still names the placeholder.

    `resolve` binds it in the field names, in the table fields and in the estimate, and
    nowhere else, which is a decision the shape documents rather than an omission: a
    placeholder in a working directory or in an identifier is an authoring mistake to
    name, never one to make work. So the binding stays narrow and this is what covers
    the rest, a label reaching the annotator with the braces beside the clinical
    question being the one that has no feedback path at all.

    The walk is generic rather than a list of the fields as they stand, since the shape
    is still growing and an enumeration stops covering whatever is added next. That is
    the very failure it is written against: `resolve`'s own short list is what left the
    other twenty fields uncovered.
    """
    return sorted({value for value in _strings(config) if USER in value})


def _reads_as_integer(column: pd.Series) -> bool:
    """Whether the column is an integer, of any width and either nullability."""
    return pd.api.types.is_integer_dtype(column.dtype)


def _reads_as_text(column: pd.Series) -> bool:
    """Whether the column is text, which an object dtype answers on its contents.

    `is_string_dtype` holds for an object column whatever sits in it, so the dtype alone
    would enrol a column of mixed values into the family and then compare it against
    text through a conversion that stringifies whatever it found. The inference is what
    settles that one. An empty column joins: nothing it holds is not a string.
    """
    if column.dtype == object:
        return pd.api.types.infer_dtype(column, skipna=True) in {"string", "empty"}

    return pd.api.types.is_string_dtype(column.dtype)


def _comparable(left: pd.Series, right: pd.Series) -> tuple[pd.Series, pd.Series]:
    """The pair read as one type where a lossless one holds both, untouched otherwise.

    Two families and no more. An integer widens into the nullable one exactly, whatever
    its width, and every spelling of text converts into `string` unchanged: a library
    moving under a parquet written a release earlier is what lands a column on the other
    spelling of one of those, and that is the whole of what this reconciles.

    The rest is handed back as it came, for the caller to report. A float compared across
    widths reads equal on values that are not, a category carries an order beside its
    values, and a datetime carries a zone. Deciding what those lose is an arbitration
    rather than a mechanism, and one made wrong accepts a value that genuinely differs
    without raising anything.

    Two equal dtypes come back untouched as well, which is the ordinary run: nothing is
    converted while the library underneath stays where it was, and what the guards pay
    is one dtype comparison per declared column.

    A conversion that raises hands the pair back untouched too, so the caller reports the
    column and the app stops. `astype` refuses an unsigned integer past the signed range
    rather than wrapping it, and a type that cannot hold what the other side carries is
    not that side's type spelled differently.

    Nothing converted is ever written. The pair exists for the length of a comparison,
    and the frame that reaches the annotator and the export is the one read from disk.
    """
    if left.dtype == right.dtype:
        return left, right

    try:
        if _reads_as_integer(left) and _reads_as_integer(right):
            return left.astype("Int64"), right.astype("Int64")

        if _reads_as_text(left) and _reads_as_text(right):
            return left.astype("string"), right.astype("string")
    except (OverflowError, TypeError, ValueError):
        return left, right

    return left, right


def _retyped(left: pd.Series, right: pd.Series) -> bool:
    """Whether no lossless type holds the two columns as they are spelled."""
    left, right = _comparable(left, right)
    return left.dtype != right.dtype


def _same_values(left: pd.Series, right: pd.Series) -> bool:
    """Whether the two columns hold the same values, under one spelling of their type.

    `Series.equals` answers on the dtype beside the values and the index, which is what
    made a spelling that moved read as a value that changed. Reconciling ahead of it
    leaves the other two answers untouched: an index that drifted is still a false, and
    so is a value the wider type merely allowed, a missing one against a number.
    """
    left, right = _comparable(left, right)
    return left.equals(right)


def aligned(df_input: pd.DataFrame, df_output: pd.DataFrame, id_field: str) -> bool:
    """Whether the two frames designate the same documents, row for row.

    `Series.equals` compares the index alongside the values, so this also rejects the
    two frames drifting apart into non-contiguous labels together, which is what would
    put the cursor on a row label that no longer exists. It compares the dtype as well,
    which a cohort is not, so the pair is reconciled first and `_same_values` carries
    what that covers.

    Both frames are expected to carry the identifier, which is what the two column
    guards establish ahead of every call. A frame without it does not answer the
    question, it invalidates it, so the lookup is left to raise rather than folded into
    a `False` that would read as a mismatch and send the operator to rebuild an input
    that carries nothing wrong.
    """
    return _same_values(df_input[id_field], df_output[id_field])


def labelled_by_position(data: pd.DataFrame) -> bool:
    """Whether the rows are labelled from zero without a gap, as the cursor assumes.

    The cursor is a position and `save_notes` addresses a label, so the two designate
    the same document on this labelling alone. Under any other one a save raises, the
    identifier the render carried down no longer matching the row the label reaches, or
    renormalizes the file and has the alignment guard stop the app on the next rerun,
    naming the input for what the app itself did.
    """
    return data.index.equals(pd.RangeIndex(len(data)))


def missing_fields(data: pd.DataFrame, fields: tuple[str, ...]) -> list[str]:
    """The declared columns the frame does not carry, in a stable order.

    The identifiers are what the alignment guard compares, and a reshape preserving
    them in value and in order passes it while dropping any other declared column. That
    one surfaces on the document the annotator happens to reach, at render or at export,
    where it is a raise inside a callback rather than a guard on the way in.
    """
    return sorted(set(fields) - set(data.columns))


def repeated_ids(data: pd.DataFrame, id_field: str) -> list[int]:
    """The positions whose identifier designates more than one document.

    The positions and never the values: the identifier is patient-linked, and what a
    guard hands the log is read by the deployment's monitoring rather than by the
    annotation. A position is what the operator needs to find the row anyway.

    An identifier repeated is what makes the save's own identity check vacuous. That
    check re-reads the row under the lock and compares the document the annotator was
    shown against the one the position now holds, which is the only place the assumption
    is observable at all; on two documents sharing an identifier it compares a value
    against itself and accepts the write onto either.
    """
    repeated = data[id_field].duplicated(keep=False)
    return [position for position, flag in enumerate(repeated) if flag]


def retyped_fields(
    df_input: pd.DataFrame,
    df_output: pd.DataFrame,
    fields: tuple[str, ...],
) -> list[str]:
    """The declared columns no lossless type holds under both frames' spellings.

    `Series.equals` answers on the values and on the dtype at once, so a column whose
    type moved between the first build and now is reported by the value comparison as a
    value that changed, and by the alignment comparison as ids that stopped matching.
    Neither is what happened, and both prescribe rebuilding an input that is not wrong:
    a regeneration reproduces the type it was regenerated by, so the app stays stopped
    on the same message with the accumulated annotation intact and out of reach.

    That is not a hypothetical drift. The output is written once at the first build and
    read back a release later, and a round trip through parquet is where a pandas
    default or an arrow-backed read lands a column on another type for the same values.

    So a pair one type holds without loss is reconciled rather than reported, and the
    two comparisons downstream read that same reconciled pair. Reporting nothing here
    while they went on answering through `Series.equals` would move the stop from one
    message to another and buy the deployment nothing. `_shared_type` carries which two
    families qualify and why the rest are left alone.

    Asked before the two comparisons rather than beside them, since a pair it refuses is
    one they would answer wrong: they read values, and two types holding different
    things are not a value that changed.

    What stays reported still stops the app. An identifier that moved from an integer to
    a float is refused, which is the collapse past the mantissa that `save_notes` reads
    through the package's own reader to avoid. What changed is that a spelling moving
    under a stored parquet no longer stops anything, and that the operator meeting a
    real one is told the type moved instead of being sent to rebuild against it.

    Naming the column is not enough for that, and the caller logs the two types beside
    it. The remedy is not at the data layer at all: the type a column reads as comes
    from the environment doing the reading, so what unsticks the app is a pinned library
    or a rewritten schema on the output, and neither is reachable from a message that
    says only which column disagreed. A type names no patient, so it is a detail the log
    can carry whole.
    """
    return sorted(name for name in fields if _retyped(df_input[name], df_output[name]))


def frozen_fields(
    df_input: pd.DataFrame,
    df_output: pd.DataFrame,
    fields: tuple[str, ...],
) -> list[str]:
    """The declared columns whose values the output froze at its first build.

    An output is resumed on its own existence, so everything outside the annotation is
    whatever the input carried when it was created. An input regenerated to correct a
    value keeps its ids and its column names, so it passes the alignment and column
    guards while the correction reaches neither the frame the app renders nor the
    export, which reads those columns out of the output.

    Both frames are expected to carry every name passed, which the two column guards
    establish ahead of every call: the annotation's own columns are dropped by the
    caller, and the text is one no declared name survives to reach. A frame without one
    does not answer the question, it invalidates it, so the lookup is left to raise
    rather than folded into an intersection that skips the name. Skipping it would
    exempt a column the regenerated input dropped from the very comparison written to
    catch that column changing, and the app would go on serving and exporting the copy
    frozen at the first build with nothing rendered and nothing logged.

    The comparison runs on the reconciled pair, so a column stored under one spelling of
    a type and read back under another is not a value that changed. What the wider type
    additionally allows still is: a number that became missing between the two frames is
    a correction, and it is reported here rather than passed over as a spelling.
    """
    return sorted(
        name for name in fields if not _same_values(df_input[name], df_output[name])
    )


def needs_write(on_disk: pd.DataFrame | None, note_fields: tuple[str, ...]) -> bool:
    """Whether the freshly built output has to reach the disk before annotating.

    The predicate reads the persisted frame rather than the one in memory: the columns
    `build_output` just added exist only in memory, and leaving them there would have
    the first save create them row by row and leave every other row null.

    The column check is what covers a second annotator arriving on an output another
    one has already started: their own columns are what is missing from it.
    """
    if on_disk is None:
        return True

    return not set(note_fields).issubset(on_disk.columns)


def write_output(data: pd.DataFrame, path: str | Path) -> None:
    """Write the output under the same lock `save_notes` takes, to serialise with it.

    The frame lands on a sibling path and is moved onto the target, as `save_notes`
    does: writing onto the target would leave a truncated parquet behind an interrupted
    write, at a path that exists, and every later start reads it and fails on it with
    no command of the app able to clear it.

    The staging path is cleared whichever way the write ends, so a failed one leaves no
    partial parquet beside the run it failed to update. The next successful write would
    move its own staging file over it, but nothing guarantees a next one, and a
    half-written sibling of the gold standard is what gets copied by mistake.

    The lock is a singleton, so a caller already holding it for the same path re-enters
    it instead of raising against itself: `filelock` refuses two distinct objects over
    one path in one thread rather than blocking on them.
    """
    with FileLock(f"{path}.lock", is_singleton=True):
        staged = Path(f"{path}.tmp")

        try:
            data.to_parquet(staged)
            staged.replace(path)
        finally:
            staged.unlink(missing_ok=True)


def load_frames(config: AnnotConfig) -> tuple[pd.DataFrame, pd.DataFrame, Path]:
    """Read the input, build or resume the output, and gate on the guards.

    Returns the input frame, which carries the document text, the output frame the app
    annotates, and the path that output was read from. Both frames are labelled by
    position and aligned row for row once the guards have passed, which is what lets the
    rest of the app address a document by position, and they agree on every declared
    column outside the annotation, which is what makes the output's own copy of them the
    export.

    The path is returned because `save_notes` takes one and this is where the only
    validated one exists. Derived here from five fields of the configuration and put to
    the kernel against the input's, it would otherwise be derived a second time by
    whoever wires the save, off the same five fields and with none of that behind it: a
    forgotten suffix then names a file this function never looked at, and the save writes
    into whatever parquet is sitting there, under its own lock, having re-read and
    re-checked the identifier against a frame nothing aligned. Handing back the one that
    passed leaves the caller nothing to derive.

    The two stems are what separates the input from the gold standard, and nothing else
    does: both paths are built from the same directory, project, split and suffix, so
    equal stems make one file of the two. Every guard below then compares that file
    against itself and passes, the build resumes on the input and keeps its text, and
    the write replaces the input with the merged frame. The run reads as normal from
    both sides while the raw input is gone and the annotation accumulates in the only
    copy left, which the rebuild the guards prescribe would erase. That one is a
    configuration defect and goes to the log alone.

    Two stems that differ still make one file where the storage folds their case or one
    name links onto the other, so the question is asked of the kernel and not of the two
    strings, `same_file` carrying what each of those costs.

    A name is one column, so two fields carrying the same one after the binding are
    refused. The binding is what makes that reachable: a group naming an annotator
    literally, for reference, and a persisted group naming the same annotator through
    the placeholder both resolve onto one column, and the save then writes the second
    field's answer over what the first was there to show. The column is checked against
    `rendered` rather than against `persisted_fields`, since the collision that loses an
    annotation is exactly the one spanning a group that writes and a group that does
    not, and no guard reading the persisted names alone can see it.

    An identifier designates one document, so an input repeating one is refused before
    anything is built on it. The alignment guard compares the two frames against each
    other and the labelling guard constrains the index, which is another axis, so
    neither asks the question; and a repeated identifier is what a join fanning out
    upstream produces. What rests on the answer is the save's own re-check, which reads
    the identifier as an identity at the one moment the cursor's assumption is
    observable, and which on two documents sharing one compares a value against itself.
    The guard is here so nothing is persisted on it, and again in `save_notes` because
    this one observed the frames a rerun boundary before the click.

    A persisted field names a column the annotation creates, so one the input already
    carries is refused. `build_output` creates a column only where the frame has none, so
    the field inherits the input's values in place of the sentinel, the save writes the
    annotation over them, and the frozen-value guard is blind to it by construction: the
    name sits in `persisted_fields`, which is precisely what that guard drops from its
    comparison. The check reads the input's own columns rather than a list of the names
    the configuration reserves, since what the field lands on is whatever the input
    happens to carry and not what the configuration says about it. A non-persisted field
    is left alone: `build_output` never touches it and the frozen-value guard does cover
    it, which is how a reconciliation app shows a prior annotator's answer.

    The configuration is expected resolved, and a string still carrying the placeholder
    stops the app rather than reaching the disk or the page: `build_output` would create
    the column literally, every guard would pass against it on every later run, and the
    mislabelled schema would be as permanent as the annotation accumulated in it, while
    a label carrying it reaches the annotator beside the clinical question with no
    feedback path at all. The audience of that one is whoever deployed the app, so it
    goes to the log alone.

    It is asked first, ahead of the paths that are built from the very fields it walks.
    A placeholder in the working directory, the project, the split, a stem or the suffix
    names a file nothing ever wrote, so the read fails and the boundary below renders a
    file that could not be opened: the operator is sent to the data generation for a
    configuration that was never resolved, and the traceback names the one component
    that broke the path first where the guard enumerates every string still carrying it.
    Nothing is written between the two positions, so the move buys the class of the
    failure rather than the failure itself. The duplicate-column check goes with it and
    the text-declaration one after them, all three reading the configuration and nothing
    else on the same argument, and none of them takes the lock: what they refuse is true
    of the deployment before any file is opened. The declared set the last one reads is
    built there for the same reason and reused under the lock, being a derivation of the
    configuration rather than of anything on the disk.

    The declared columns are checked against the output, which is what makes
    `table_fields` a description of the output's own columns rather than the input's.
    That is a decision and no longer an accident: the summary table is sourced from the
    output, so an excerpt of the clinical text is a column the input generation creates,
    which reaches the output through the first copy like any other.

    The footer's columns are declared too, on the same argument as the table's: the
    render reads them by name off the output, so one the deployment stopped producing is
    a column to name here rather than a lookup that raises mid-page and arrives as the
    generic message. They are read from the values of that mapping, its keys being the
    headings a page paints and not columns to look for anywhere.

    Which is why the text itself, declared anywhere, is refused by name and up front.
    `build_output` drops it unconditionally, so the column cannot exist in the output
    and the check against it would report a name the operator then goes looking for in
    a data generation that is not wrong. Every declaration is covered rather than the
    table alone, the metadata and the counter reaching the same comparison, and the
    remedy is one wherever it was written.

    The output is then asked for the text by name, under the lock and on its own. That
    the configuration declares it nowhere says what the app reads, and `build_output`
    drops it on the first build alone: an output resumed comes back as it sits on the
    disk, and the text is the one column no declared name reaches, so the column guard
    passes over it, the type guard has nothing to read it from and the value guard drops
    it with the rest. Whatever put it there, a hand-built output, a generation writing
    one directly or a release predating the drop, it stays for the life of the file with
    nothing rendered and nothing logged. Measured over two thousand documents of four
    kilobytes: the gold standard is 190 times its size, and one save round trip goes
    from 5.7 ms to 93 ms, paid under the exclusive lock on the one operation an annotator
    repeats. The message is the generic one and not a rebuild: the annotation in that
    file is intact and worth keeping, so the remedy is to drop the column and not to
    regenerate anything, and it is the log that carries it.

    Every rendered field is declared, not only the persisted ones. A group shown for
    reference and never written owns columns `build_output` does not create: they reach
    the output through the input's first copy alone, so a regenerated input that drops
    or renames one passes every other guard and surfaces on whichever document the
    annotator reaches. Reading them off `persisted_fields` would leave them covered
    only when the consumer happens to repeat them among the table fields, which is a
    summary table's list and not a statement about what the annotation reads. So too
    the frozen-value guard covers them too: a reference answer the input has since
    corrected stops the app rather than being reconciled against in its stale form.

    A reference column has to be in the input, then, and that is a precondition on the
    deployment rather than a property the package can arrange. It is the one thing the
    output can carry without the input ever having done so: another annotator's app
    declares the same column as persisted, and `build_output` creates it there against
    that configuration and no other. A reconciliation arriving on the shared output of
    the annotators it reconciles is refused for that reason, and its input is what has
    to fold their answers in, upstream, as the reconciler's own generation does. Sourcing
    the reference from the output instead is what is refused and not the pattern: nothing
    would compare that column against anything, and a prior annotator revising an answer
    would reach the reconciliation nowhere and stop nothing, which is the silent stale
    reference the guard was widened to cover in the first place.

    Those same columns are asked of the input as well, and not of the output alone. The
    output carries every one of them by construction, having frozen its copy at the
    first build, so the declared-column guard above cannot see one leave the input; and
    a name the input no longer carries is a name the value comparison would have had
    nothing to compare, which is how a column removed upstream went on being served and
    exported from the frozen copy with nothing rendered and nothing logged. The audience
    differs from the value guard's: one says the schema moved, the other says a value
    did, and a removal deliberate enough to be the point of the regeneration is exactly
    the one that must not read as a drifting value.

    Once both frames are known to carry those columns, they are asked to agree on the
    type of them before either comparison reads a value, `retyped_fields` carrying what
    that ordering is worth. Agreeing is not spelling it the same way: a pair one type
    holds without loss is reconciled for the length of the comparisons and refused
    nowhere, so a pandas or arrow release moving under a stored parquet no longer puts
    the accumulated annotation out of reach.

    It is also why the alignment guard follows the input's own column check rather than
    preceding it: the identifier is one of the columns whose type can move, so the check
    that establishes the question has to run before the guard that would otherwise
    answer it wrong.

    Every guard logs what the page withholds, and says on the page that it did. The
    withholding is by disclosure: a column, a path, a row count and a position are each
    read by whoever is looking at the browser, and a guard is not the place to hand them
    over. What follows is that the page alone never resolves the remedy, so a message
    that named no log left the operator with a prescription and no way to apply it. The
    labelling guard is where that costs the most: it reads two frames, the message names
    neither, and the two remedies are opposite, a mislabelled input being rebuilt
    upstream where a mislabelled output is discarded and rebuilt from the input. The
    alignment guard's two row counts separate a cohort that changed size from one that
    was reordered, which is the same fork.

    Reading the output, deciding whether it has to be written and writing it happen
    under one lock, the one `save_notes` takes. Deciding outside it would have a save
    landing between the read and the write, and the write would put the frame read
    before that save back on the disk.

    The lock spans that and no more. Every guard reading the input alone stands ahead of
    it: the frame they read is already in memory, they open nothing, and holding an
    exclusive lock over them makes one annotator's reshaped input wait on another's
    save before being refused. The guards between the build and the write stay inside,
    where they belong, reading a frame the lock is what makes current.

    The output is read twice under it, once by `build_output` and once for the write
    decision, and that stands. The decision reads what the file carries where the build
    returns a frame it has already added the annotation columns to, so the second read
    is not the first one's result: collapsing them moves the resumption decision out of
    `build_output`, whose whole contract it is. Measured on the annotation's own scale,
    the redundant read is 2.9 ms over 500 documents, 5.2 ms over 5000 and 9.5 ms over
    50000. The unit of work above it is a clinician reading a clinical note, and nothing
    here asks for those milliseconds back at the price of that coupling.

    This is the boundary the reads are caught at. A missing file, an unreadable one, a
    truncated parquet and a missing column each raise naming the deployment path or the
    column, and `client.showErrorDetails` paints an uncaught raise into the browser of
    whoever is looking. The detail goes to the module log, one message goes to the page.
    Every guard leaves through `_halt`, whose raise descends from `BaseException` and so
    passes through the clause untouched, as the runner's own `ScriptControlException`
    does, and the clause itself leaves the same way rather than falling past its own stop.

    That the stop is this module's and not `st.stop`'s is what makes the chain hold off
    the script thread, each guard here being one statement followed by another. `st.stop`
    requests a stop of the ambient context and forces a yield point at which the runner
    raises; with no context at all it does nothing whatever, and from the worker thread
    of a `parallel=True` fragment the yield point consults a coordinator whose stop event
    the request never sets, streamlit's own `_check_not_parallel_worker` being wired to
    nothing in 1.60. Either way the annotation reads `NoReturn` and no checker can tell
    the two apart. The data path still belongs on the script thread, a worker pool
    breaking its lock discipline before it reaches its guards, but that is a property of
    the deployment now rather than the only thing standing in front of the write.

    The return sits outside the `try` and stays there, an `else` clause buying nothing.
    What holds the binding is `_halt`'s own `NoReturn` and not `st.stop`'s: measured both
    ways, dropping it reports two uninitialized names here and one missing return under
    the clause, and a boundary reverted to a bare `st.stop` reports nothing under either.
    The two shapes catch the same regression and are blind to the same one, so what is
    left is where the survivor lands: falling past the return raises in this module,
    where returning `None` from a clause raises in the consumer, on a frame naming its
    file and nothing about why.
    """
    try:
        unresolved = unresolved_fields(config)

        if unresolved:
            _logger.error("The configuration was never resolved: %s", unresolved)
            _halt(MESSAGE_UNAVAILABLE)

        declared_twice = sorted(
            name
            for name, count in Counter(field.name for field in config.rendered).items()
            if count > 1
        )

        if declared_twice:
            _logger.error("Two fields are declared on %s", declared_twice)
            _halt(MESSAGE_UNAVAILABLE)

        declared = tuple(
            dict.fromkeys(
                (
                    config.id_field,
                    *config.export_fields,
                    *config.table_fields,
                    *config.footer_fields.values(),
                    *(field.name for field in config.rendered),
                )
            )
        )

        if config.text_field in declared:
            _logger.error(
                "The document text is declared as a column: %r", config.text_field
            )
            _halt(MESSAGE_UNAVAILABLE)

        paths = (config.work_dir, config.proj, config.split)
        input_path = data_path(*paths, config.input_stem, config.data_suffix)
        output_path = data_path(*paths, config.output_stem, config.data_suffix)

        if same_file(input_path, output_path):
            _logger.error("One file answers to %s and to %s", input_path, output_path)
            _halt(MESSAGE_UNAVAILABLE)

        df_input = _read_input(input_path, _stamp(input_path))

        missing = missing_fields(df_input, (config.id_field, config.text_field))

        if missing:
            _logger.error("The input does not carry %s", missing)
            _halt(MESSAGE_COLUMNS)

        repeated = repeated_ids(df_input, config.id_field)

        if repeated:
            _logger.error("One identifier designates the documents at %s", repeated)
            _halt(MESSAGE_DUPLICATES)

        overwritten = sorted(set(config.persisted_fields) & set(df_input.columns))

        if overwritten:
            _logger.error("A field annotates the input column %s", overwritten)
            _halt(MESSAGE_UNAVAILABLE)

        with FileLock(f"{output_path}.lock", is_singleton=True):
            df_output = build_output(
                df_input,
                output_path,
                config.text_field,
                config.persisted_fields,
            )

            if config.text_field in df_output.columns:
                _logger.error(
                    "The output carries the document text: %r", config.text_field
                )
                _halt(MESSAGE_UNAVAILABLE)

            mislabelled = [
                name
                for name, frame in (("input", df_input), ("output", df_output))
                if not labelled_by_position(frame)
            ]

            if mislabelled:
                _logger.error(
                    "The rows of the %s are not labelled by position", mislabelled
                )
                _halt(MESSAGE_LABELS)

            missing = missing_fields(df_output, declared)

            if missing:
                _logger.error("The output does not carry %s", missing)
                _halt(MESSAGE_COLUMNS)

            frozen = tuple(
                name for name in declared if name not in config.persisted_fields
            )
            missing = missing_fields(df_input, frozen)

            if missing:
                _logger.error("The output carries %s where the input does not", missing)
                _halt(MESSAGE_COLUMNS)

            retyped = retyped_fields(df_input, df_output, frozen)

            if retyped:
                _logger.error(
                    "The input and the output read %s",
                    {
                        name: f"{df_input[name].dtype} against {df_output[name].dtype}"
                        for name in retyped
                    },
                )
                _halt(MESSAGE_TYPES)

            if not aligned(df_input, df_output, config.id_field):
                _logger.error(
                    "The two frames designate different documents, %d rows against %d",
                    len(df_input),
                    len(df_output),
                )
                _halt(MESSAGE_MISMATCH)

            drifted = frozen_fields(df_input, df_output, frozen)

            if drifted:
                _logger.error("The output was built on other values for %s", drifted)
                _halt(MESSAGE_VALUES)

            on_disk = read_data(output_path) if Path(output_path).exists() else None

            if needs_write(on_disk, config.persisted_fields):
                write_output(df_output, output_path)
    except Exception:
        _logger.exception("Annotation data unusable, holding the app back")
        _halt(MESSAGE_UNAVAILABLE)

    return df_input, df_output, output_path


def stylesheet(path: str | Path) -> str:
    """A project stylesheet, wrapped in the tag that carries it into a page.

    Read on every run rather than cached. `st.cache_data` is keyed on the qualified
    name and global to the process, so a shared reader would need the path as an
    argument to stay honest, and what it would buy back is a two-kilobyte read against
    a page that reads a clinical note.
    """
    return f"<style>{Path(path).read_text(encoding='utf-8')}</style>"


def cursor_seed(data: pd.DataFrame, estimate_field: str) -> int:
    """Which document the run opens on: the last one answered, or the first.

    The last answered rather than the one after it, so the annotator arrives on their
    own last answer and can see what it was before moving on. A run with no answer at
    all opens on the first document.
    """
    answered = data.index[data[estimate_field].ne(EMPTY)]
    return 0 if answered.empty else int(answered.max())


def exportable(config: AnnotConfig, data: pd.DataFrame) -> pd.DataFrame:
    """The rows and columns an export carries.

    A document with no answer is never in it, whatever the configuration says: an export
    is a gold standard, and the empty sentinel is the absence of an answer rather than
    one of the values a deployment may want to keep out.

    The selection is a membership test and not a `query` expression. That one takes the
    column name and every excluded value into a string it then parses, so a column named
    with a space or an answer carrying an apostrophe fails at the parse, inside the
    render, on labels a French deployment writes by hand.
    """
    excluded = (EMPTY, *config.export_excluded)
    kept = ~data[config.estimate_field].isin(excluded)
    return data.loc[kept, list(config.export_fields)]


def complete(config: AnnotConfig, data: pd.DataFrame) -> bool:
    """Whether every document carries one of the answers the estimate offers."""
    return bool(data[config.estimate_field].isin(config.estimate_options).all())


def tuto_form(tuto: Tuto) -> None:
    """Render the tutorial, which is the body of the dialog the sidebar opens.

    A plain function at module level, because `AppTest` runs no `st.dialog` body
    (streamlit#9786): what the decorated shell holds is the display, and this is what a
    test reaches.
    """
    st.markdown(tuto.text.read_text(encoding="utf-8"))

    with tuto.media.open("rb") as stream:
        st.video(stream)


def _field_key(name: str, index: int) -> str:
    """The session entry one field's widget holds its answer under, at one cursor.

    Derived rather than spelled out, because it is read from two places: the render that
    draws the widget, and the save that reads the answer back off it a callback later.
    Two spellings of one convention drift, and the drift is a save reading an entry
    nothing writes.
    """
    return f"key_{name}_{index}"


def render_field(note: NoteField, index: int, *, editable: bool) -> None:
    """Render one annotation widget and keep its answer under the field's own name.

    `editable` is the group's `persisted`, read at the call site where both words are
    visible together: a field is live exactly where its answer is written, and the two
    were declared apart until the pairing that renders a live widget nothing saves
    turned out to be what a group got by default. It is named for what this render
    reads rather than for where it comes from, and it takes no default, the caller
    being the one holding the group.

    Two names are in play and neither substitutes for the other. The widget's own key
    folds in the cursor, so moving to another document renders another widget rather
    than carrying this one's answer along, and it is the name a save reads: Streamlit
    applies the browser's values to the keyed entries and then calls the click
    callbacks, both ahead of this body, so at the instant a callback runs that entry
    holds what the annotator has on screen. The field name holds the same answer for
    the render that writes it, which is what everything drawn after this reads; a
    callback reading it would read the render before, and put a correction back into
    the gold standard as the value it corrects.

    An answer the options no longer offer selects nothing rather than raising, which is
    what a document annotated before a level was renamed holds. The annotator then sees
    an empty radio on a document that has an answer, and the answer is still in the file:
    it is a display of what is there and not a correction of it.

    `persist_state` is passed rather than inherited. A field of a group that is not
    rendered on this run loses its session value, which is the regime of a conditional
    render, and the default is `None` today: a version bump changing that would change
    what a widget remembers with nothing here to fail.
    """
    state = st.session_state
    key = _field_key(note.name, index)

    if note.kind == "radio":
        answer = state[note.name]
        state[note.name] = st.radio(
            label=note.label,
            options=note.options,
            index=note.options.index(answer) if answer in note.options else None,
            key=key,
            disabled=not editable,
            persist_state=None,
        )
        return

    state[note.name] = st.text_input(
        label=note.label,
        value=state[note.name],
        key=key,
        disabled=not editable,
        persist_state=None,
    )


def _render_tuto(tuto: Tuto) -> None:
    """Render the button that opens the tutorial, and the dialog behind it.

    The button is disabled where either file is missing, which answers for the instant
    it is drawn rather than for the click. `disabled` is a frontend attribute the server
    does not consult when dispatching, so the dialog carries its own boundary instead of
    resting on it: what a missing asset raises is a path, and an uncaught raise in a
    dialog reaches Streamlit's own handler, which paints it.

    That boundary cannot be the one around the render either. `st.dialog` is a fragment,
    and a fragment rerun calls the body Streamlit stored rather than the enclosing
    function again, so a failure on any rerun after the click lands outside `run_app`.
    """

    @st.dialog(title=" ", width="large")
    def dialog() -> None:
        try:
            tuto_form(tuto)
        except Exception:
            _logger.exception("The tutorial could not be rendered")
            st.error(MESSAGE_UNAVAILABLE)

    with st.sidebar.container(key="button-tuto"):
        clicked = st.button(
            label=LABEL_TUTO,
            icon=":material/videocam:",
            width="stretch",
            disabled=not (tuto.text.exists() and tuto.media.exists()),
        )

        if clicked:
            dialog()


def moved_to(rows: list[int], index: int) -> int | None:
    """Where a row selection moves the cursor, or nothing where it does not move.

    A selection is a position in the frame as displayed, and the cursor is a position
    too, so the one is the other: the guards have established that the output is
    labelled from zero without a gap, and the table is that frame in its own order. The
    counter column is displayed and never converted back into a position, which is what
    a run whose counter does not start at one would have made wrong without saying so.

    A selection landing on the document already shown moves nothing, since what follows
    a move is a rerun and that one would draw the same page again on every pass.

    It is a function of its own because `AppTest` offers no way to select a row, the
    same reason a dialog body is one: this is the shape that can be tested, and the
    shell around it holds the display.
    """
    if not rows:
        return None

    selected = rows[0]
    return None if selected == index else selected


def _render_table(config: AnnotConfig, df_output: pd.DataFrame, index: int) -> None:
    """Render the summary table and let a row selection move the cursor.

    The key folds in the number of saves as well as the cursor. A save rewrites the
    column this table displays, and a widget keyed on the cursor alone would hand back
    the selection state of a table drawn before the write.
    """
    state = st.session_state
    labels = {note.name: note.label for note in config.rendered}

    event = st.dataframe(
        data=df_output[list(config.table_fields)],
        on_select="rerun",
        selection_mode="single-row",
        column_config={
            name: st.column_config.Column(
                label=labels.get(name),
                width=config.column_widths.get(name),
            )
            for name in config.table_fields
        },
        key=f"df_{state.save_count}_{index}",
    )

    moved = moved_to(event["selection"]["rows"], index)

    if moved is not None:
        state.doc_index = moved
        st.rerun()


def _document_source(document: str) -> str:
    """The document as a source a frame loads, rather than as the frame's own markup.

    A raw string reaches the frame's inline-document attribute, which Streamlit renders
    under a sandbox granting `allow-same-origin` and `allow-scripts` at once: the two
    together leave the frame the app's own origin, and the framework's own reference
    text for that element names a database column as what must never be passed to it.
    A note is one. Markup the corpus carries would then read the annotator's
    authenticated page and drive the widgets that write the gold standard, at a layer
    none of the guards standing in front of that write observes.

    A document loaded from a `data:` URL carries an opaque origin whatever that sandbox
    allows, the scheme deciding it rather than the attribute, so the note reaches
    nothing outside its own frame. What that does not close is script running inside
    the frame at all, the sandbox being the frontend's and fixed.

    Nothing of the note is stripped on the way, where a sanitiser would repair a
    clinical document silently, on a page whose whole purpose is that a clinician reads
    what the record holds.
    """
    payload = b64encode(document.encode("utf-8")).decode("ascii")
    return f"data:text/html;charset=utf-8;base64,{payload}"


def _render_body(
    config: AnnotConfig,
    df_input: pd.DataFrame,
    df_output: pd.DataFrame,
    index: int,
    text_style: str,
) -> None:
    """Render the metadata line, the document, and the summary table beside it.

    The metadata are read from the output and the document from the input, which is the
    one column the output does not carry. Both frames designate the same document at
    this position, the alignment guard being what establishes that and the only thing
    that does.

    The document goes into a frame of its own with the project's text stylesheet
    prefixed to it. A clinical note is markup a page-level sheet does not reach, and it
    is not the app's own markup either: rendering it into the page would let a note
    style the widgets around it.

    The frame loads it rather than carrying it inline, which is what denies a note the
    app's own origin. Measuring content needs that origin, so the frame no longer sizes
    itself and the height below is the package's rather than the framework's fallback.
    """
    st.dataframe(
        data=(
            df_output.loc[:, list(config.meta_fields)]
            .iloc[[index]]
            .rename(columns=dict(config.meta_fields))
        ),
        hide_index=True,
        column_config={
            heading: st.column_config.Column(width=config.column_widths[name])
            for name, heading in config.meta_fields.items()
            if name in config.column_widths
        },
    )

    col_text, col_table = st.columns(_COLUMNS_BODY)

    with col_text:
        st.iframe(
            src=_document_source(text_style + df_input[config.text_field].iloc[index]),
            height=_HEIGHT_DOCUMENT,
        )

    with col_table:
        _render_table(config, df_output, index)


def _render_sidebar(
    config: AnnotConfig,
    df_output: pd.DataFrame,
    output_path: Path,
    index: int,
    title: str,
    text_style: str,
) -> None:
    """Render the sidebar: tutorial, cursor, fields, navigation, gauge, export, footer.

    The groups are what the layout reads, rather than the flat pairing: a divider is
    what separates one group from the next, and a render that flattened them would have
    to infer the boundary from a property that means something else.

    The document slider is a position throughout, offered one-based because that is what
    a person counts from. The counter column is what the table displays, and neither
    widget turns a value read from the frame back into a position.

    The gauge is keyed on the number of saves and the fields on the cursor, which is the
    difference between them: the gauge counts what reached the disk across the whole run
    and stands still on an arrow, where a field belongs to one document. It is a slider
    for its shape alone and never a control, so it is disabled: a value the browser
    submits under a key outlives every interaction that does not move that key, and this
    one moves on a save, so a gauge nudged between two saves would answer the sidebar's
    one question wrong until a reload.

    Every element the containers below hold is written as a bare `st.` call. An
    `st.sidebar.` one lands in the sidebar all the same, and reads as equivalent, but
    it resolves to the sidebar rather than to the enclosing `with`, so the container
    comes out empty and its key names nothing to style.

    The footer paints a value as markup and not as text, that value being a fragment
    lifted from the document and carrying the classes the text sheet declares for it;
    escaping it would paint the tags in letters where a clinician reads a highlighted
    excerpt. So the corpus reaches the page here, where the document itself reaches an
    opaque origin, and what stands between the two is `st.html`'s own sanitising rather
    than anything this package does. Measured against the shipped build, that sanitising
    holds a script and an event handler, a frame, a link and a `javascript:` or `data:`
    target, and it holds none of: a `style` element, whose CSS it does not read at all,
    a `style` attribute on any of the hundred and nineteen tags it allows, or an
    ordinary `http` resource a tag loads, which leaves the annotator's browser as a
    request whatever the origin. The frame is no better on that last one.

    None of it is observable from here. `AppTest` reads what this hands the frontend,
    and the sanitising is the browser's, so a version bump moving that profile would
    pass every test in this repository.

    The sheet is prefixed to each block rather than emitted once above them, so a block
    carries what paints it instead of resting on a neighbour emitted before it. What
    that repeats is the sheet's own bytes, 592 of them on the deployment measured, once
    per declared field: nothing at one field, 1776 more than a single emission at the
    four the same deployment's page sheet already names classes for, against a page
    carrying a clinical note and two tables.
    """
    state = st.session_state
    nrow = len(df_output)

    st.sidebar.space()

    if config.tuto is not None:
        _render_tuto(config.tuto)

    st.sidebar.space("stretch")

    with st.sidebar.container(key="slider-doc"):
        position = st.slider(
            label=" ",
            label_visibility="collapsed",
            value=index + 1,
            min_value=1,
            max_value=nrow,
            key=f"slider_doc_{index}",
            persist_state=None,
        )

        if position - 1 != index:
            state.doc_index = position - 1
            st.rerun()

    for rank, group in enumerate(config.groups):
        if rank:
            st.sidebar.divider()
            st.sidebar.space("stretch")

        pairs = zip(group.fields[::2], group.fields[1::2], strict=True)

        for estimate, comment in pairs:
            col_estimate, col_comment = st.sidebar.columns(_COLUMNS_ROW)

            with col_estimate:
                render_field(estimate, index, editable=group.persisted)

            with col_comment:
                render_field(comment, index, editable=group.persisted)

    st.sidebar.space("stretch")

    with st.sidebar.container(key="navigation"):
        navigation(
            output_path,
            nrow,
            {name: _field_key(name, index) for name in config.persisted_fields},
            columns=_COLUMNS_NAVIGATION,
            can_save=state[config.estimate_field] not in (EMPTY, None),
            save_label=LABEL_SAVE,
            id_field=config.id_field,
            doc_id=df_output[config.id_field].iloc[index],
        )

    with st.sidebar.container(key="slider-note"):
        st.slider(
            label=" ",
            label_visibility="collapsed",
            value=int(df_output[config.estimate_field].ne(EMPTY).sum()),
            max_value=nrow,
            key=f"slider_note_{state.save_count}",
            disabled=True,
            persist_state=None,
        )

    visibility = "visible" if config.download_visible else "hidden"

    with st.sidebar.container(key=f"button-download-{visibility}"):
        if config.download_visible:
            download(
                config.secrets,
                exportable(config, df_output),
                filename=title,
                label=LABEL_DOWNLOAD,
                info=config.export_info,
                button_type="primary" if complete(config, df_output) else "secondary",
            )

    st.sidebar.space("stretch")

    with st.sidebar.container(key="sidebar-footer"):
        for heading, name in config.footer_fields.items():
            st.html(
                text_style
                + f"<p class='sidebar-footer-title'>{heading}</p>"
                + f"<p class='sidebar-footer-{name}'>{df_output[name].iloc[index]}</p>"
            )


def run_app(config: AnnotConfig) -> None:
    """Render one annotation app, from its configuration and nothing else.

    A consuming entrypoint is an import, a configuration literal and this call. Whatever
    the two apps of a deployment differ by is expressed in that literal: the two file
    stems, the fields and their labels, the summary table's columns, how many groups the
    sidebar lays out, whether a tutorial is offered, and which groups are written back.

    The order is not cosmetic. The login gate runs first, since the answers it gives are
    what the column names are bound to, and a run with no gate binds the anonymous name
    rather than leaving the placeholder for a guard to catch. `st.set_page_config`
    follows it because the title it sets carries the annotator the gate returns, and not
    because it has to come first: Streamlit takes it at any point of a run and takes it
    more than once, each call overriding the parameters it names. So the run that
    withholds the app configures no page at all and the form takes the framework's own
    defaults, which it is written for, centring itself in a column of its own rather
    than resting on a layout. A deployment wanting that form under the app's layout
    splits the call in two rather than moving this one. The data path comes before the
    session because where the cursor opens is read off the output, and the guards have
    to have passed before a page is drawn on either frame.

    Nothing reaches the browser from here. `client.showErrorDetails` defaults to `full`,
    so an uncaught raise arrives with its traceback and the server's paths, to whoever is
    looking; the detail goes to the module log and one sentence to the page. The clause
    is broad on purpose, its point being that nothing escapes rather than that a known
    list is handled, and it is safe against the framework's own control flow: `st.rerun`
    and `st.stop` raise from `ScriptControlException`, which descends from
    `BaseException`, as does the stop every guard of the data path leaves through.

    The cursor is checked against the run before the page is drawn from it, which is
    the one input the render reads that no guard of the data path sees: those read the
    two files and the configuration, and this one is the session's. It cannot leave the
    range of the run it was moved in, every mover clamping, and it leaves the range of a
    later one, the data being regenerated smaller while a tab stays open, which is the
    operator action several of those guards prescribe on their own message. It stops
    rather than moving the cursor back in: what a repair would cost is an annotator
    carried to another patient with nothing said, where the page can say instead that
    the run changed and that reloading opens it again.

    A position below the run is what makes that check worth its line. `iloc` reads it as
    a position from the end, and `st.slider` widens its own bounds around a value below
    its minimum rather than raising, so the last document of the run rendered under the
    first position and every save was refused, by the label check that reads an index
    the frame does not carry.

    The title names the page, the exported file and the sheet inside it, and that last
    one is what bounds it: a sheet name takes at most thirty-one characters, so the two
    declared stems and their two separators leave the rest to the annotator's account
    name, nineteen of them on a deployment splitting by month and naming its project in
    three letters. Past that the export refuses, once the annotator has authenticated
    for it, and the log names the length while the page names nothing. It is left to
    refuse rather than truncated, a workbook whose sheet is quietly not the one asked
    for being what the export's own preconditions exist against, and a deployment whose
    accounts are longer shortens what it declares here.

    It asks for no menu either, where an empty mapping asked for one and got nothing:
    the framework builds each of the three entries only from the address or the text a
    page supplies for it, and this one supplies none, so they are already absent and a
    mapping naming them would raise a flag against entries that never render.

    The configuration object never enters `st.session_state`. Editing any file reimports
    the class module, so an `isinstance` against an instance stored there fails against
    its own class (streamlit#9638); it is an argument, and it stays one.

    What this call does not do is guard the cadence of what it calls. It runs whole on
    every click, so a validation added to a function it reaches is paid once per rerun
    per annotator, which is the reason the guards of the data path are where they are
    rather than spread through the render.
    """
    try:
        user = ANONYMOUS

        if config.auth:
            authenticated = login(config.secrets, info=config.login_info)

            if authenticated is None:
                return

            user = authenticated

        config = config.resolve(user)
        df_input, df_output, output_path = load_frames(config)
        title = f"{config.split}_{config.proj}_{user}"

        st.set_page_config(
            page_title=title,
            layout="wide",
            initial_sidebar_state="expanded",
        )
        st.markdown(stylesheet(config.styles.app), unsafe_allow_html=True)

        state = st.session_state

        if "doc_index" not in state:
            state.doc_index = cursor_seed(df_output, config.estimate_field)

        if "save_count" not in state:
            state.save_count = 0

        index = int(state.doc_index)

        if index not in range(len(df_output)):
            _logger.error(
                "The session holds position %d of a run of %d", index, len(df_output)
            )
            _halt(MESSAGE_POSITION)

        text_style = stylesheet(config.styles.text)

        for note in config.rendered:
            state[note.name] = df_output[note.name].iloc[index]

        _render_body(config, df_input, df_output, index, text_style)
        _render_sidebar(config, df_output, output_path, index, title, text_style)
    except Exception:
        _logger.exception("The annotation app could not be rendered")
        _halt(MESSAGE_UNAVAILABLE)
