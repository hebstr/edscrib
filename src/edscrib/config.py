"""The declarative shape an annotation app reduces to.

A project describes its fields, its file stems and its column names here, and the
package derives everything the app body needs from that description. Nothing in this
module names a domain concept: every column name arrives from the consumer.

Construction is where this package raises, against every other module, which logs the
detail and stops the app on one message. Four refusals live here: a group that is not
a sequence of radio/text rows, a field whose options do not match its kind, and a
configuration or a tutorial pair built positionally. They raise because each one makes
a derivation unsound rather than a run wrong, `rows` resting on the pairing and the
export on the column names being the ones meant, so an object violating them must not
exist rather than exist and be caught a rerun later.

A consumer declares its configuration at module scope, outside every boundary this
package owns, so an uncaught refusal reaches the framework's own handler, which paints
its traceback and the server's paths into the browser of whoever is looking. What
escapes is that traceback and never the message: the three name only what this module
itself declares, a widget kind or an argument, and never a column, a value or a path.
A consumer constructing a configuration inside a rendered page owes it a boundary of
its own.
"""

from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal

USER = "{user}"


@dataclass(frozen=True)
class NoteField:
    """One annotation widget, and the output column it reads and writes.

    The default empty `options` is right for one of the two kinds and never for the
    other, which is what the guard says. A radio declared without them renders and holds
    `None`, rather than raising: that is a third value beside the empty sentinel and a
    real answer, and it is what a save would put in the column. Options on a text are
    copy nothing reads.
    """

    name: str
    label: str
    kind: Literal["radio", "text"]
    options: tuple[str, ...] = ()
    editable: bool = True

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
class AnnotConfig:
    """Everything one annotation app declares, and what the package derives from it.

    Keyword-only where the two classes above are not, because three of the fourteen
    fields are consecutive column names of one type and transposing two of them is
    silent all the way down: the export then carries the document identifier the row
    counter is there to keep out of it, and both identity guards, the alignment one and
    the one the save re-runs under its lock, compare a row counter against itself and
    pass on any two frames of equal length. A transposition in either of the others
    fails loudly, on the group's own validation or on the column the render cannot
    find. A consumer meets this one in its own type checker rather than at render.
    """

    work_dir: str | Path
    proj: str
    split: str
    input_stem: str
    output_stem: str
    groups: tuple[FieldGroup, ...]
    table_fields: tuple[str, ...]
    index_field: str
    id_field: str
    text_field: str
    meta_fields: Mapping[str, str]
    auth: bool = False
    tuto: Tuto | None = None
    data_suffix: str = ""

    def resolve(self, user: str) -> "AnnotConfig":
        """Bind the authenticated user into the column names that carry it.

        The only late binding the shape allows, and it happens once, after the login
        gate. Anything else computed at render time would run on every rerun.
        """
        groups = tuple(
            replace(
                group,
                fields=tuple(
                    replace(field, name=field.name.replace(USER, user))
                    for field in group.fields
                ),
            )
            for group in self.groups
        )
        table = tuple(name.replace(USER, user) for name in self.table_fields)
        return replace(self, groups=groups, table_fields=table)

    @property
    def rendered(self) -> tuple[NoteField, ...]:
        return tuple(field for group in self.groups for field in group.fields)

    @property
    def persisted_fields(self) -> tuple[str, ...]:
        return tuple(
            field.name
            for group in self.groups
            if group.persisted
            for field in group.fields
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
