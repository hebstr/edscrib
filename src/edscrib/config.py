"""The declarative shape an annotation app reduces to.

A project describes its fields, its file stems and its column names here, and the
package derives everything the app body needs from that description. Nothing in this
module names a domain concept: every column name arrives from the consumer.

Construction is where this package raises, against every other module, which logs the
detail and stops the app on one message. Six refusals live here: a group that is not a
sequence of radio/text rows, a field whose options do not match its kind, an estimate
that does not name a radio of a persisted group, and a configuration, a tutorial pair
or a stylesheet pair built positionally. They raise because each one makes
a derivation unsound rather than a run wrong, `rows` resting on the pairing and the
export on the column names being the ones meant, so an object violating them must not
exist rather than exist and be caught a rerun later.

A consumer declares its configuration at module scope, outside every boundary this
package owns, so an uncaught refusal reaches the framework's own handler, which paints
its traceback and the server's paths into the browser of whoever is looking. What
escapes is that traceback and never the message: they name only what this module itself
declares, a widget kind, a field of this shape or an argument, and never a column, a
value or a path.
A consumer constructing a configuration inside a rendered page owes it a boundary of
its own.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Literal

USER = "{user}"

ANONYMOUS = "admin"
"""The name a run with no login gate annotates under.

It is not cosmetic: it binds the placeholder, so it is what the annotation columns are
named after and what the gold standard carries for the life of the file. A deployment
turning `auth` on later therefore starts fresh columns rather than continuing these.
"""


@dataclass(frozen=True)
class NoteField:
    """One annotation widget, and the output column it reads and writes.

    The default empty `options` is right for one of the two kinds and never for the
    other, which is what the guard says. A radio declared without them renders and holds
    `None`, rather than raising: that is a third value beside the empty sentinel and a
    real answer, and it is what a save would put in the column. Options on a text are
    copy nothing reads.

    Whether the annotator answers this field or reads it is not declared here. It is
    the group's `persisted`, that being the same decision said once: a field is live
    exactly where its answer is written, and the render derives one from the other.
    """

    name: str
    label: str
    kind: Literal["radio", "text"]
    options: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.kind == "radio" and not self.options:
            raise ValueError("a radio offers the options it declares, got none")

        if self.kind == "text" and self.options:
            raise ValueError("a text offers no option, got: " + ", ".join(self.options))


@dataclass(frozen=True)
class FieldGroup:
    """Fields laid out together, and written together or not at all.

    A group is a sequence of rows, each row a radio and the comment beside it. The
    pairing is positional, so the guard below turns that tacit invariant into a named
    failure rather than a silently wrong layout.

    What a group writes is also what it lets an annotator answer, and `persisted` says
    both at once. A live widget in a group that is never written takes an answer, shows
    it, and hands back the previous one as soon as the cursor moves, with nothing
    written, nothing rendered and nothing logged; the render reads this flag rather
    than a second one beside it, so that pairing cannot be declared at all.

    An empty one is refused by the same guard rather than by it: on no field at all the
    pairing holds vacuously, both slices being empty, and the group then contributes
    nothing to any derivation, so a section emptied by a typo renders as a section
    nobody declared.
    """

    fields: tuple[NoteField, ...]
    persisted: bool = True

    def __post_init__(self) -> None:
        if not self.fields:
            raise ValueError("a group is a sequence of radio/text rows, got none")

        kinds = [field.kind for field in self.fields]
        rows = len(kinds) // 2
        if kinds[::2] != ["radio"] * rows or kinds[1::2] != ["text"] * rows:
            raise ValueError(
                "a group is a sequence of radio/text rows, got: " + ", ".join(kinds),
            )


@dataclass(frozen=True, kw_only=True)
class Tuto:
    """The two files a tutorial dialog renders, each named for what it is.

    Keyword-only for the reason `AnnotConfig` is: two consecutive fields of one type,
    and a transposition hands a markdown file to a video player and the bytes of a
    recording to a markdown renderer, at the click and inside a dialog where nothing
    else is looking. A pair of positions says which file comes first and nothing about
    which is which; this says it at the only place a consumer writes it.

    Neither file is opened here, and their existence is not asked. A configuration is
    declared at module scope, so a check here would raise at import on a deployment
    whose asset is not in place yet, and it would answer for the instant before the
    click rather than the click: what the render disables the button on is what decides.
    """

    text: Path
    media: Path


@dataclass(frozen=True, kw_only=True)
class Styles:
    """The two stylesheets a page carries, each named for where it is injected.

    Keyword-only and named for the reason `Tuto` is: two consecutive fields of one type,
    and a transposition puts the page chrome inside the frame holding a clinical note
    and the note's own rules over every widget of the app, which renders and is wrong
    rather than raising.

    The package ships neither, and the two roles are the whole of what it knows about
    them. `app` reaches the page itself and styles what the render lays out, keyed on
    the container names below. `text` is the sheet the document's own markup is painted
    under, and it is prefixed to the two places that markup reaches a reader: the frame
    holding the document, and the sidebar footer, which paints a fragment lifted from
    the document and carries the document's classes with it.

    Only the first of those two is an iframe. The footer goes through `st.html`, which
    renders into the page, so a `text` sheet reaches the page there and whatever it
    declares outside its own class selectors lands on the app.

    A `run_app` page emits ten container keys, which Streamlit turns into
    `st-key-<name>` classes. Six are the render's own: `button-tuto`, `slider-doc`,
    `navigation`, `slider-note`, `sidebar-footer`, and `button-download-visible` or
    `button-download-hidden`. Four are the default parameters of the two functions the
    render calls, `navigation` and `download`: `button-backward`, `button-save`,
    `button-forward` and `button-export`. A direct caller of those two names its own, a
    `run_app` consumer cannot, so all ten are the seam `app` styles against and part of
    what this package promises rather than an implementation detail. Two are emitted
    only where what they name is offered, `button-tuto` and `button-export`; the
    download's own container is named on every page, under whichever of its two names
    the configuration declares.

    Nothing else is. Every `key` becomes such a class, and the render's own widget keys
    fold in the cursor or the number of saves, so they name a position in a run rather
    than a place on a page; the login form's two fields are keyed so the session can
    hold and drop the password, not so a sheet can find them.
    """

    app: Path
    text: Path


@dataclass(frozen=True, kw_only=True)
class AnnotConfig:
    """Everything one annotation app declares, and what the package derives from it.

    Keyword-only where the two classes above are not, because four of the twenty-three
    fields are consecutive column names of one type and transposing two of them is
    silent all the way down: the export then carries the document identifier the row
    counter is there to keep out of it, and both identity guards, the alignment one and
    the one the save re-runs under its lock, compare a row counter against itself and
    pass on any two frames of equal length. A transposition in either of the others
    fails loudly, on the group's own validation or on the column the render cannot
    find. A consumer meets this one in its own type checker rather than at render.

    `estimate_field` names the one answer the run is measured by, and five derivations
    read it: where the cursor opens, whether the save button is live, what the progress
    gauge shows, whether the export button reads as complete, and which rows the export
    carries. It is declared rather than taken as the first persisted field, the shape
    being keyword-only precisely because a column name resting on a position is what
    goes wrong quietly; the guard below is what keeps the declaration honest.

    `table_fields` describes the output's own columns, the summary table being sourced
    from it, and it carries the counter like any other column. The two heading mappings
    run in opposite directions and each is written the way its own render reads it:
    `meta_fields` is keyed on the column, a line of metadata being a frame selected then
    relabelled, and `footer_fields` on the heading, a footer being a heading painted then
    filled. The column side is what the declared-column guard reads in both cases.

    `column_widths` is keyed on column names rather than on the headings they render
    under, so it shares a key space with the fields above and a width outlives a
    relabelling. A column it does not name takes whatever the framework gives it.

    `export_excluded` holds the answers that keep a document out of the export. The
    unanswered sentinel is not among them and never needs to be: the package excludes it
    on its own, an export being a gold standard and an empty answer being no answer.

    `download_visible` decides whether the export is offered at all, and not only how
    its container is named. Both names are written either way, a stylesheet hooking on
    them, and the widget behind exists only where the flag says so: that sheet lives in
    the deployment rather than here, so a rule dropped from it, a file not yet in place
    or a class prefix a version bump moves would otherwise put the export back on a page
    that declared it away, with nothing here to notice. What it withholds is the entry
    point and never the bytes, which the dialog's own credential check holds.
    """

    work_dir: str | Path
    proj: str
    split: str
    input_stem: str
    output_stem: str
    secrets: str | Path
    styles: Styles
    groups: tuple[FieldGroup, ...]
    table_fields: tuple[str, ...]
    index_field: str
    id_field: str
    text_field: str
    estimate_field: str
    meta_fields: Mapping[str, str]
    footer_fields: Mapping[str, str] = field(default_factory=dict)
    column_widths: Mapping[str, int] = field(default_factory=dict)
    export_excluded: tuple[str, ...] = ()
    login_info: str = ""
    export_info: str = ""
    auth: bool = False
    download_visible: bool = False
    tuto: Tuto | None = None
    data_suffix: str = ""

    def __post_init__(self) -> None:
        estimates = {
            note.name
            for group in self.groups
            if group.persisted
            for note in group.fields
            if note.kind == "radio"
        }

        if self.estimate_field not in estimates:
            raise ValueError(
                "estimate_field names a radio of a persisted group, and none matches"
            )

    def resolve(self, user: str) -> "AnnotConfig":
        """Bind the authenticated user into the column names that carry it.

        The only late binding the shape allows, and it happens once, after the login
        gate. Anything else computed at render time would run on every rerun.
        """
        groups = tuple(
            replace(
                group,
                fields=tuple(
                    replace(note, name=note.name.replace(USER, user))
                    for note in group.fields
                ),
            )
            for group in self.groups
        )
        table = tuple(name.replace(USER, user) for name in self.table_fields)
        return replace(
            self,
            groups=groups,
            table_fields=table,
            estimate_field=self.estimate_field.replace(USER, user),
        )

    @property
    def rendered(self) -> tuple[NoteField, ...]:
        return tuple(note for group in self.groups for note in group.fields)

    @property
    def persisted_fields(self) -> tuple[str, ...]:
        return tuple(
            note.name for group in self.groups if group.persisted for note in group.fields
        )

    @property
    def rows(self) -> tuple[tuple[NoteField, ...], ...]:
        """The pairs to lay out, across every group and flat.

        Slicing the flattened fields two by two never straddles two groups, because a
        group of odd length is refused: the pairing guard compares slices whose lengths
        already differ there. So this rests on that guard rather than on the slicing,
        and relaxing the one silently mispairs the other.

        Which group a pair came from is dropped, since nothing lays out per group and
        `FieldGroup` carries nothing to render as a section of its own. A render that
        comes to need it reads `groups` directly, which is what holds the nesting.
        """
        fields = self.rendered
        return tuple(tuple(fields[i : i + 2]) for i in range(0, len(fields), 2))

    @property
    def estimate_options(self) -> tuple[str, ...]:
        """The answers the estimate offers, which construction guarantees there are.

        A radio is refused without options and `estimate_field` is refused unless it
        names one, so the search below always finds a non-empty tuple. Both refusals are
        what this rests on rather than a fallback of its own: an empty tuple handed back
        here would make every document read as unanswered and the export as complete.
        """
        return next(
            note.options for note in self.rendered if note.name == self.estimate_field
        )

    @property
    def export_fields(self) -> tuple[str, ...]:
        """The columns an export carries, in order and each of them once.

        The three sources can name one column twice, the counter being declarable among
        the metadata as well. A caller selects on this, and a repeated label selects a
        frame carrying the column twice, which reaches the workbook as two columns of
        one content. The two guards reading a column list fold theirs into a set and so
        never saw it; this one is the list itself.
        """
        return tuple(
            dict.fromkeys((self.index_field, *self.meta_fields, *self.persisted_fields))
        )
