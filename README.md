# edscrib

edscrib carries the manual side of clinical text annotation: the Streamlit apps a human uses to annotate, correct, and reconcile what an extraction pipeline or an LLM produced.
It is the companion of [edstr](https://github.com/hebstr/edstr), which handles the machine side (Oracle import, cleaning, concept matching).

An app is a configuration plus a call: the package holds the rendering engine, the parquet input/output, the navigation under lock, the HMAC authentication, and the Excel export.
Everything naming a column, a label, or a classification level stays in the consuming project.

## Status

Bootstrapping.
The public API is empty and the code descends symbol by symbol from its first consumer.

## Installation

edscrib is not on PyPI.
Install from GitHub, pinned to a tag:

```sh
uv add git+https://github.com/hebstr/edscrib --tag v0.1.0
```

## Development

```sh
uv sync
uv run pytest
```

A consuming project exercises an unreleased state without declaring the dependency:

```sh
uv run --with-editable ../path/to/py-edscrib pytest
```

## License

[MIT](LICENSE.md)
