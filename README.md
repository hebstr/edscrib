# edscrib

edscrib carries the manual side of clinical text annotation: the Streamlit apps a human uses to annotate, correct, and reconcile what an extraction pipeline or an LLM produced.
It is the companion of [edstr](https://github.com/hebstr/edstr), which handles the machine side (Oracle import, cleaning, concept matching).

An app is a configuration plus a call: in its finished shape the package holds the rendering engine, the parquet input/output, the navigation under lock, the HMAC authentication, and the Excel export.
Everything naming a column, a label, or a classification level stays in the consuming project.

## Status

Bootstrapping.
The code descends symbol by symbol from its first consumer.
`__all__` is still empty: what has landed is imported from its own module, `edscrib.io` for the parquet paths, the reading and the output creation, `edscrib.navigation` for the document cursor and the save under lock, and `edscrib.auth` for the login form and the credential check.

## Installation

edscrib is not on PyPI.
Install from GitHub, pinned to a tag:

```sh
uv add git+https://github.com/hebstr/edscrib --tag v0.1.0
```

## Development

```sh
uv sync
prek install
uv run pytest
```

`prek install` lays down the two hook types the project declares: the staged-file pass (prose, file hygiene, secret scan, then the lint, format and type gate) and the Conventional Commits check on the message.

A consuming project exercises an unreleased state without declaring the dependency:

```sh
uv run --with-editable ../path/to/py-edscrib pytest
```

That overlay stops being enough once the consumer's own code imports edscrib: it is ephemeral, so a type checker pinned to the consumer's `.venv` reports the import as missing.
Install into that venv instead, and replay it after each `uv sync`, which purges it:

```sh
uv pip install -e ../path/to/py-edscrib
```

## License

[MIT](LICENSE.md)
