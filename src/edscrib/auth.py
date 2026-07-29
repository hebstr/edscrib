"""Gating an annotation app behind the credentials of a deployment."""

import hmac
import tomllib
from pathlib import Path

import streamlit as st


def load_secrets(path: str | Path) -> dict[str, str]:
    """Read the `[users]` table of the TOML file holding the annotator passwords.

    A missing file, an unparseable one, a `[users]` table that is absent, not a table
    or empty, and an annotator whose password is not a non-empty string all raise,
    naming the path: an app started against the wrong deployment directory would
    otherwise reject every annotator with the message of a wrong password, and an unset
    password would admit anyone knowing the name, `compare_digest` holding two empty
    values equal.

    The table is rebuilt rather than handed over, which is what earns the return type:
    a value read out of a parsed TOML document is `Any`, and an annotation asserting
    otherwise would type-check the callers against a promise nothing verified.
    """
    secrets_path = Path(path)

    if not secrets_path.exists():
        raise FileNotFoundError(f"Secrets file not found at {secrets_path}")

    with secrets_path.open("rb") as f:
        try:
            secrets = tomllib.load(f)
        except tomllib.TOMLDecodeError as error:
            raise ValueError(f"{secrets_path} is not readable TOML: {error}") from error

    users = secrets.get("users")

    if not isinstance(users, dict):
        raise KeyError(f"{secrets_path} has no [users] table")

    passwords: dict[str, str] = {}
    unusable: list[str] = []

    for name, word in users.items():
        if isinstance(word, str) and word:
            passwords[str(name)] = word
        else:
            unusable.append(str(name))

    if unusable:
        raise ValueError(
            f"{secrets_path} holds no usable password for {', '.join(sorted(unusable))}"
        )

    if not passwords:
        raise ValueError(f"{secrets_path} has an empty [users] table")

    return passwords


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
        return hmac.compare_digest(password.encode(), users[username].encode())

    return False


def login(path: str | Path, *, title: str = "Connexion", info: str = "") -> bool:
    """Render the login form and tell whether the session is authenticated.

    Returns True on a session that already authenticated, without rendering anything,
    so the caller guards its own body on the result and stops on False.

    The secrets are read on render, before anyone types, so a deployment pointed at a
    missing or shapeless file fails there rather than on a password submission. The
    read sits behind the authenticated return: it guards a session on its way in, and
    an annotator already through would otherwise lose their work on the rerun following
    any edit of the file on the server, which the operator makes to add a colleague.

    Four keys are written in `st.session_state`: `username` and `password` by the two
    inputs, then `password_correct` and, on success, `user`, which carries the
    annotator through the rest of the app.

    Both inputs carry `persist_state` rather than inherit it: the value that leaves the
    session when the form stops being rendered is a password in clear, so an upstream
    change of default would otherwise keep it in server memory without a failing test.

    The password is dropped from the session on every submission, accepted or not, so
    that a value the annotator typed by mistake does not sit in server memory for the
    life of the session. It also clears the field, without which a rejected annotator
    correcting their identifier alone submits no change, so the callback never fires
    and the form does not answer.
    """
    state = st.session_state

    def password_entered() -> None:
        username = state["username"]

        if verify_credentials(path, username, state["password"]):
            state["password_correct"] = True
            state["user"] = username
        else:
            state["password_correct"] = False

        del state["password"]

    if state.get("password_correct"):
        if "user" not in state:
            raise KeyError("'password_correct' is set without 'user' in st.session_state")

        return True

    load_secrets(path)

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
