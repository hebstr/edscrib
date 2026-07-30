"""The declarative shape an annotation app reduces to.

A project describes its fields, its file suffixes and its column names here, and the
package derives everything the app body needs from that description. Nothing in this
module names a domain concept: every column name arrives from the consumer.
"""

from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal

USER = "{user}"


@dataclass(frozen=True)
class NoteField:
    """One annotation widget, and the output column it reads and writes."""

    name: str
    label: str
    kind: Literal["radio", "text"]
    options: tuple[str, ...] = ()
    editable: bool = True


@dataclass(frozen=True)
class FieldGroup:
    """Fields laid out together, and written together or not at all.

    A group is a sequence of rows, each row a radio and the comment beside it. The
    pairing is positional, so the guard below turns that tacit invariant into a named
    failure rather than a silently wrong layout.
    """

    fields: tuple[NoteField, ...]
    persisted: bool = True

    def __post_init__(self) -> None:
        kinds = [field.kind for field in self.fields]
        rows = len(kinds) // 2
        if kinds[::2] != ["radio"] * rows or kinds[1::2] != ["text"] * rows:
            raise ValueError(
                "a group is a sequence of radio/text rows, got: " + ", ".join(kinds),
            )


@dataclass(frozen=True)
class AnnotConfig:
    """Everything one annotation app declares, and what the package derives from it."""

    work_dir: str | Path
    proj: str
    split: str
    input_suffix: str
    output_suffix: str
    groups: tuple[FieldGroup, ...]
    table_fields: tuple[str, ...]
    index_field: str
    id_field: str
    text_field: str
    meta_fields: Mapping[str, str]
    auth: bool = False
    tuto: tuple[Path, Path] | None = None
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
    def persisted(self) -> tuple[str, ...]:
        return tuple(
            field.name
            for group in self.groups
            if group.persisted
            for field in group.fields
        )

    @property
    def rows(self) -> tuple[tuple[NoteField, ...], ...]:
        fields = self.rendered
        return tuple(tuple(fields[i : i + 2]) for i in range(0, len(fields), 2))

    @property
    def export_fields(self) -> tuple[str, ...]:
        return (self.index_field, *self.meta_fields, *self.persisted)
