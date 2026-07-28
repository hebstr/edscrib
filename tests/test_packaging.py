"""Packaging smoke: the src layout resolves and the type marker sits beside the import.

Whether the marker reaches a built wheel is a build concern, not an import one, and this
says nothing about it: under an editable install both assertions read the working tree.
"""

from importlib.resources import files

import edscrib


def test_public_api_is_declared():
    assert isinstance(edscrib.__all__, list)


def test_type_marker_is_installed():
    assert files("edscrib").joinpath("py.typed").is_file()
