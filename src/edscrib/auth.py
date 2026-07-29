"""Gating an annotation app behind the credentials of a deployment."""

import hmac
import logging
import tomllib
import unicodedata
from pathlib import Path

import streamlit as st

_logger = logging.getLogger(__name__)

_DEPLOYMENT_ERROR = "Login unavailable: contact the application administrator."


class SecretsError(Exception):
    """The deployment secrets cannot be read, or hold nothing anyone can log in with.

    One type for every way the file fails, so that a caller guarding a deployment
    writes one `except` rather than three: the conditions differ but the answer to
    all of them is the same, no annotator gets in until an operator intervenes. What
    raised underneath stays reachable through `__cause__`.

    The type is what a guard catches, so it has to be narrower than the builtins it
    replaces: an `except KeyError` standing where this one stands would swallow a
    plain bug in the reading and report it as a broken deployment.
    """


def load_secrets(path: str | Path) -> dict[str, str]:
    """Read the `[users]` table of the TOML file holding the annotator passwords.

    A file that cannot be read, an unparseable one, and a `[users]` table that is
    absent, not a table or empty all raise, naming the path: an app started against the
    wrong deployment directory would otherwise reject every annotator with the message
    of a wrong password.

    Reading is what decides whether the file is there, rather than a prior `exists`:
    the check answers for the instant before the read and not for the read itself, and
    it passes a path naming a directory or a file the server may not open, both of
    which would then leave as an `OSError` no guard here converts and no caller expects.

    Every way the read can fail arrives as this one type, which takes three clauses
    rather than two: `tomllib` decodes the bytes before parsing them, so a file an
    editor wrote in latin-1 to hold an accented password raises `UnicodeDecodeError`,
    a `ValueError` and no `TOMLDecodeError`. The parse clause comes first for being
    the narrower of the two.

    An annotator whose password is not a non-empty string is dropped and logged, not
    raised on. Dropping is what closes the bypass, `compare_digest` holding two empty
    values equal, and it closes it for that name alone; raising would take the whole
    deployment down over one half-typed line, on the very operation the operator
    performs on the running server to add a colleague. The name is then unknown to
    `verify_credentials`, which answers it as it answers any other unknown one.

    The table is rebuilt rather than handed over, which is what earns the return type:
    a value read out of a parsed TOML document is `Any`, and an annotation asserting
    otherwise would type-check the callers against a promise nothing verified.
    """
    secrets_path = Path(path)

    try:
        with secrets_path.open("rb") as f:
            secrets = tomllib.load(f)
    except OSError as error:
        raise SecretsError(f"{secrets_path} cannot be read: {error.strerror}") from error
    except tomllib.TOMLDecodeError as error:
        raise SecretsError(f"{secrets_path} is not readable TOML: {error}") from error
    except ValueError as error:
        raise SecretsError(f"{secrets_path} cannot be read: {error}") from error

    users = secrets.get("users")

    if not isinstance(users, dict):
        raise SecretsError(f"{secrets_path} has no [users] table")

    passwords: dict[str, str] = {}
    unusable: list[str] = []

    for name, word in users.items():
        if isinstance(word, str) and word:
            passwords[str(name)] = word
        else:
            unusable.append(str(name))

    if unusable:
        _logger.warning(
            "%s holds no usable password for %s",
            secrets_path,
            ", ".join(sorted(unusable)),
        )

    if not passwords:
        raise SecretsError(f"{secrets_path} has an empty [users] table")

    return passwords


def _opaque(word: str) -> bytes:
    """Render a password as the bytes two spellings of it agree on.

    NFC is what PRECIS mandates for a password (RFC 8265, the `OpaqueString`
    profile): an accented character has a composed and a decomposed spelling, and
    the two carry different bytes, so an operator whose editor writes the file in one
    and an annotator whose browser submits the other would never match. The rejection
    naming neither field, that annotator is locked out with nothing to go on.
    """
    return unicodedata.normalize("NFC", word).encode()


def verify_credentials(path: str | Path, username: str, password: str) -> bool:
    """Tell whether a username and password pair matches the deployment secrets.

    An unknown username is a plain false, not a raise: the caller shows one message
    for both failures, so that neither tells which of the two was wrong.

    The pair is compared as UTF-8 bytes: `compare_digest` refuses two strings as soon
    as either holds a non-ASCII character, so an accented password would raise where
    it should simply not match.
    """
    users = load_secrets(path)

    if username in users:
        return hmac.compare_digest(_opaque(password), _opaque(users[username]))

    return False


def login(path: str | Path, *, title: str = "Connexion", info: str = "") -> bool:
    """Render the login form and tell whether the session is authenticated.

    Returns True on a session that already authenticated, without rendering anything,
    so the caller guards its own body on the result and stops on False.

    The secrets are read on render, before anyone types, so a deployment pointed at a
    missing or shapeless file withholds the form rather than answering a submission.
    The read sits behind the authenticated return: it guards a session on its way in,
    and an annotator already through would otherwise lose their work on the rerun
    following any edit of the file on the server, which the operator makes to add a
    colleague.

    Which is also to say a session is never re-checked once it is through, and a
    Streamlit session lives as long as its browser tab. Striking a name out of the file
    does not close what that annotator already has open, however long they leave it
    open; the operator restarts the server to end it.

    What `load_secrets` raises names the file and, on a half-written table, the
    annotators it holds no password for. Both reads are therefore caught here and sent
    to the module logger, where the operator reads them, against a single message that
    tells an unauthenticated visitor nothing: `client.showErrorDetails` defaults to
    `full`, so anything uncaught reaches the browser with its traceback, and this is a
    library whose deployments cannot each be relied on to have set it otherwise. The
    functions themselves keep raising, so a caller wanting the detail asks them.

    Nothing here limits how often a pair may be tried. A counter kept in the session
    would bound nobody, a new tab starting a session of its own, so the limit belongs
    to whatever fronts the deployment and is named here to be placed there rather than
    assumed.

    Four keys are written in `st.session_state`: `username` and `password` by the two
    inputs, then `password_correct` and, on success, `user`, which carries the
    annotator through the rest of the app.

    Both inputs carry `persist_state` rather than inherit it: the value that leaves the
    session when the form stops being rendered is a password in clear, so an upstream
    change of default would otherwise keep it in server memory without a failing test.

    The password is taken out of the session before it is checked, so that a value the
    annotator typed by mistake does not sit in server memory for the life of the
    session: the check reads the secrets file, which raises on a deployment whose file
    an operator is editing, and a removal placed after it would be skipped on exactly
    the failure the rest of this function exists to survive. It also clears the field,
    without which a rejected annotator correcting their identifier alone submits no
    change, so the callback never fires and the form does not answer.
    """
    state = st.session_state

    def password_entered() -> None:
        username = state["username"]
        password = state.pop("password")

        try:
            correct = verify_credentials(path, username, password)
        except SecretsError:
            _logger.exception("Secrets unusable while checking a submitted pair")
            return

        if correct:
            state["password_correct"] = True
            state["user"] = username
        else:
            state["password_correct"] = False

    if state.get("password_correct"):
        if "user" not in state:
            raise KeyError("'password_correct' is set without 'user' in st.session_state")

        return True

    try:
        load_secrets(path)
    except SecretsError:
        _logger.exception("Secrets unusable, holding the login form back")
        st.error(_DEPLOYMENT_ERROR)
        return False

    _, col, _ = st.columns([1, 3, 1])

    with col:
        st.markdown(f"### {title}")

        if info:
            st.info(info)

        st.text_input(label="Identifiant", key="username", persist_state=None)

        st.text_input(
            label="Mot de passe",
            type="password",
            on_change=password_entered,
            key="password",
            persist_state=None,
        )

    if "password_correct" in state and not state["password_correct"]:
        st.error("😕 Nom d'utilisateur ou mot de passe incorrect")

    return False
