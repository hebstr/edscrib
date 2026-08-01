"""Everything the package renders, and the single module it is written in."""

import importlib
import pkgutil
import re

import edscrib
import edscrib.messages
from edscrib.app import MESSAGE_UNAVAILABLE as UNAVAILABLE_DATA
from edscrib.auth import MESSAGE_UNAVAILABLE as UNAVAILABLE_SECRETS
from edscrib.messages import MESSAGE_UNAVAILABLE

RENDERED = ("MESSAGE_", "LABEL_", "TITLE_")


def modules():
    """Every module of the package, imported."""
    return [
        importlib.import_module(f"edscrib.{info.name}")
        for info in pkgutil.iter_modules(edscrib.__path__)
    ]


def test_one_message_answers_for_a_service_nobody_can_use():
    """Two failures the annotator can do nothing about, and one sentence for both.

    The secrets file and the annotation data break independently and the reader is the
    same person, who is told the same thing either way. Two spellings under one name
    were what a shell rendered depending on which subsystem failed, in two languages.
    """
    assert UNAVAILABLE_SECRETS is MESSAGE_UNAVAILABLE
    assert UNAVAILABLE_DATA is MESSAGE_UNAVAILABLE


def test_every_rendered_string_is_written_in_one_module():
    """The copy has one home, so a second spelling of one sentence cannot be declared.

    Two modules each holding a `MESSAGE_UNAVAILABLE` collided under one name without
    raising, and what broke was what the annotator saw. This is that failure made
    mechanical: whatever a module renders, it imports.
    """
    for module in modules():
        for name, value in vars(module).items():
            if name.startswith(RENDERED):
                assert value is getattr(edscrib.messages, name), (
                    f"{module.__name__} declares its own {name}"
                )


def test_the_copy_is_the_language_of_the_deployment():
    """The reader is a French-speaking clinician, and the log is the operator's.

    Nothing here can assert a translation, so this pins the mechanical trace of one: no
    message carries an English function word. A string rewritten or added in English
    fails here rather than reaching an annotator, which is how the two languages came to
    live side by side in the first place.
    """
    english = re.compile(r"\b(the|is|are|was|were|and|not|with|which|could)\b")
    messages = {
        name: value
        for name, value in vars(edscrib.messages).items()
        if name.startswith("MESSAGE_")
    }

    assert len(messages) == 10

    for name, text in messages.items():
        assert not english.search(text.lower()), f"{name} reads as English"
