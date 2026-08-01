"""The body an annotation app reduces to, driven by its `AnnotConfig`.

This module holds the data path: resolving the two parquet paths, reading the input,
building or resuming the output, and the guards that stand between a reshaped input
and a mislabelled gold standard. Rendering lives beside it.
"""

import dataclasses
import logging
from collections import Counter
from collections.abc import Iterable, Iterator, Mapping
from pathlib import Path

import pandas as pd
import streamlit as st
from filelock import FileLock

from edscrib.config import USER, AnnotConfig
from edscrib.io import build_output, data_path, read_data

MESSAGE_MISMATCH = (
    "The output was built on a different input: its document ids no longer match, "
    "in value or in order. Rebuild the input before annotating. The detail is in the "
    "server log, which the deployment operator reads."
)

MESSAGE_LABELS = (
    "The documents are not labelled from zero without a gap, so a position in the run "
    "no longer designates the row it shows. Rebuild the data before annotating. The "
    "detail is in the server log, which the deployment operator reads."
)

MESSAGE_DUPLICATES = (
    "The documents are not identified one by one: an identifier designates more than "
    "one of them, so a save can no longer tell which it was written on. Rebuild the "
    "data before annotating. The detail is in the server log, which the deployment "
    "operator reads."
)

MESSAGE_COLUMNS = (
    "The data no longer carries every column the annotation declares. The detail is in "
    "the server log, which the deployment operator reads."
)

MESSAGE_TYPES = (
    "The output and the input no longer read a column they share as the same type, so "
    "their values can no longer be compared. The annotation already recorded is intact. "
    "The detail is in the server log, which the deployment operator reads."
)

MESSAGE_VALUES = (
    "The output was built on different values: a column the input carries no longer "
    "matches what the output records. The detail is in the server log, which the "
    "deployment operator reads."
)

MESSAGE_UNAVAILABLE = (
    "The annotation data could not be opened. The detail is in the server log, which "
    "the deployment operator reads."
)

_logger = logging.getLogger(__name__)


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

    `resolve` binds it in the field names and in the table fields, and nowhere else,
    which is a decision the shape documents rather than an omission to patch: a
    placeholder in a working directory or in an identifier is an authoring mistake to
    name, never one to make work. So the binding stays narrow and this is what covers
    the rest, a label reaching the annotator with the braces beside the clinical
    question being the one that has no feedback path at all.

    The walk is generic rather than a list of the fields as they stand, since the shape
    is still growing and an enumeration stops covering whatever is added next. That is
    the very failure it is written against: `resolve`'s own two-item list is what left
    the other twelve fields uncovered.
    """
    return sorted({value for value in _strings(config) if USER in value})


def aligned(df_input: pd.DataFrame, df_output: pd.DataFrame, id_field: str) -> bool:
    """Whether the two frames designate the same documents, row for row.

    `Series.equals` compares the index alongside the values, so this also rejects the
    two frames drifting apart into non-contiguous labels together, which is what would
    put the cursor on a row label that no longer exists.

    Both frames are expected to carry the identifier, which is what the two column
    guards establish ahead of every call. A frame without it does not answer the
    question, it invalidates it, so the lookup is left to raise rather than folded into
    a `False` that would read as a mismatch and send the operator to rebuild an input
    that carries nothing wrong.
    """
    return df_input[id_field].equals(df_output[id_field])


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
    """The declared columns the two frames no longer read as the same type.

    `Series.equals` answers on the values and on the dtype at once, so a column whose
    type moved between the first build and now is reported by the value comparison as a
    value that changed, and by the alignment comparison as ids that stopped matching.
    Neither is what happened, and both prescribe rebuilding an input that is not wrong:
    a regeneration reproduces the type it was regenerated by, so the app stays stopped
    on the same message with the accumulated annotation intact and out of reach.

    That is not a hypothetical drift. The output is written once at the first build and
    read back a release later, and a round trip through parquet is where a pandas
    default or an arrow-backed read lands a column on another type for the same values.

    Asked before the two comparisons rather than beside them, since it is what makes
    either of them mean anything: they answer on values, and two types are not a value
    that changed. Nothing is loosened by it. An identifier that moved from an integer to
    a float still stops the app, which is the collapse past the mantissa that `save_notes`
    reads through the package's own reader to avoid; what changes is that the operator
    is told the type moved instead of being sent to rebuild against it.

    Naming the column is not enough for that, and the caller logs the two types beside
    it. The remedy is not at the data layer at all: the type a column reads as comes
    from the environment doing the reading, so what unsticks the app is a pinned library
    or a rewritten schema on the output, and neither is reachable from a message that
    says only which column disagreed. A type names no patient, so it is a detail the log
    can carry whole.
    """
    return sorted(
        name for name in fields if df_input[name].dtype != df_output[name].dtype
    )


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
    """
    return sorted(name for name in fields if not df_input[name].equals(df_output[name]))


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


def load_frames(config: AnnotConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read the input, build or resume the output, and gate on the guards.

    Returns the input frame, which carries the document text, and the output frame the
    app annotates. Both are labelled by position and aligned row for row once the
    guards have passed, which is what lets the rest of the app address a document by
    position, and they agree on every declared column outside the annotation, which is
    what makes the output's own copy of them the export.

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
    failure rather than the failure itself. The duplicate-column check goes with it,
    reading the configuration and nothing else on the same argument, and neither takes
    the lock: what they refuse is true of the deployment before any file is opened.

    The declared columns are checked against the output, which is what makes
    `table_fields` a description of the output's own columns rather than the input's.
    `build_output` drops the text and adds the annotation, so a table field naming the
    text stops the app on every run. Which of the two frames a table reads is the
    rendering half's to settle; until it does, this guard is what fixes it to the
    output.

    Every rendered field is declared, not only the persisted ones. A group shown for
    reference and never written owns columns `build_output` does not create: they reach
    the output through the input's first copy alone, so a regenerated input that drops
    or renames one passes every other guard and surfaces on whichever document the
    annotator reaches. Reading them off `persisted_fields` would leave them covered
    only when the consumer happens to repeat them among the table fields, which is a
    summary table's list and not a statement about what the annotation reads. So too
    the frozen-value guard covers them too: a reference answer the input has since
    corrected stops the app rather than being reconciled against in its stale form.

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
    that ordering is worth. It is why the alignment guard follows the input's own column
    check rather than preceding it: the identifier is one of the columns whose type can
    move, so the check that establishes the question has to run before the guard that
    would otherwise answer it wrong.

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
    The guards below stop the script through `ScriptControlException`, which descends
    from `BaseException` and so passes through the clause untouched.

    That raise is the runner's own and not `st.stop`'s, whose body requests a stop and
    forces a yield point where the enqueued message is handled. It is reached on the
    script thread and inside the exec alone, and each guard here is one statement
    followed by another, so every one of them rests on that single condition. With no
    run context
    the call does nothing whatever; from the worker thread of a `parallel=True` fragment
    the yield point consults a coordinator whose stop event the request never sets, and
    streamlit's own `_check_not_parallel_worker` is wired to nothing in 1.60. The guard
    then logs, renders and returns, and everything after it runs, down to the write:
    entered that way with two stems resolving to one file, this replaces the raw input
    with the merged frame, which is the outcome its first guard exists to prevent. The
    data path belongs on the script thread for that reason, and a worker pool would
    break its lock discipline before it broke its guards.
    """
    try:
        unresolved = unresolved_fields(config)

        if unresolved:
            _logger.error("The configuration was never resolved: %s", unresolved)
            st.error(MESSAGE_UNAVAILABLE)
            st.stop()

        declared_twice = sorted(
            name
            for name, count in Counter(field.name for field in config.rendered).items()
            if count > 1
        )

        if declared_twice:
            _logger.error("Two fields are declared on %s", declared_twice)
            st.error(MESSAGE_UNAVAILABLE)
            st.stop()

        paths = (config.work_dir, config.proj, config.split)
        input_path = data_path(*paths, config.input_stem, config.data_suffix)
        output_path = data_path(*paths, config.output_stem, config.data_suffix)

        if same_file(input_path, output_path):
            _logger.error("One file answers to %s and to %s", input_path, output_path)
            st.error(MESSAGE_UNAVAILABLE)
            st.stop()

        df_input = _read_input(input_path, _stamp(input_path))

        missing = missing_fields(df_input, (config.id_field, config.text_field))

        if missing:
            _logger.error("The input does not carry %s", missing)
            st.error(MESSAGE_COLUMNS)
            st.stop()

        repeated = repeated_ids(df_input, config.id_field)

        if repeated:
            _logger.error("One identifier designates the documents at %s", repeated)
            st.error(MESSAGE_DUPLICATES)
            st.stop()

        overwritten = sorted(set(config.persisted_fields) & set(df_input.columns))

        if overwritten:
            _logger.error("A field annotates the input column %s", overwritten)
            st.error(MESSAGE_UNAVAILABLE)
            st.stop()

        with FileLock(f"{output_path}.lock", is_singleton=True):
            df_output = build_output(
                df_input,
                output_path,
                config.text_field,
                config.persisted_fields,
            )

            mislabelled = [
                name
                for name, frame in (("input", df_input), ("output", df_output))
                if not labelled_by_position(frame)
            ]

            if mislabelled:
                _logger.error(
                    "The rows of the %s are not labelled by position", mislabelled
                )
                st.error(MESSAGE_LABELS)
                st.stop()

            declared = tuple(
                dict.fromkeys(
                    (
                        config.id_field,
                        *config.export_fields,
                        *config.table_fields,
                        *(field.name for field in config.rendered),
                    )
                )
            )
            missing = missing_fields(df_output, declared)

            if missing:
                _logger.error("The output does not carry %s", missing)
                st.error(MESSAGE_COLUMNS)
                st.stop()

            frozen = tuple(
                name for name in declared if name not in config.persisted_fields
            )
            missing = missing_fields(df_input, frozen)

            if missing:
                _logger.error("The input no longer carries %s", missing)
                st.error(MESSAGE_COLUMNS)
                st.stop()

            retyped = retyped_fields(df_input, df_output, frozen)

            if retyped:
                _logger.error(
                    "The input and the output read %s",
                    {
                        name: f"{df_input[name].dtype} against {df_output[name].dtype}"
                        for name in retyped
                    },
                )
                st.error(MESSAGE_TYPES)
                st.stop()

            if not aligned(df_input, df_output, config.id_field):
                _logger.error(
                    "The two frames designate different documents, %d rows against %d",
                    len(df_input),
                    len(df_output),
                )
                st.error(MESSAGE_MISMATCH)
                st.stop()

            drifted = frozen_fields(df_input, df_output, frozen)

            if drifted:
                _logger.error("The output was built on other values for %s", drifted)
                st.error(MESSAGE_VALUES)
                st.stop()

            on_disk = read_data(output_path) if Path(output_path).exists() else None

            if needs_write(on_disk, config.persisted_fields):
                write_output(df_output, output_path)
    except Exception:
        _logger.exception("Annotation data unusable, holding the app back")
        st.error(MESSAGE_UNAVAILABLE)
        st.stop()

    return df_input, df_output
