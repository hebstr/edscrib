"""Gating an annotation app behind the credentials of a deployment."""

import hmac
import tomllib
from pathlib import Path
from typing import Any

import streamlit as st


def load_secrets(path: str | Path) -> dict[str, Any]:
    """Read the TOML file holding the annotators and their passwords.

    A missing file and a file without a `[users]` table both raise, naming the path:
    an app started against the wrong deployment directory would otherwise reject
    every annotator with the message of a wrong password.
    """
    secrets_path = Path(path)

    if not secrets_path.exists():
        raise FileNotFoundError(f"Secrets file not found at {secrets_path}")

    with secrets_path.open("rb") as f:
        secrets = tomllib.load(f)

    if "users" not in secrets:
        raise KeyError(f"{secrets_path} has no [users] table")

    return secrets


def verify_credentials(path: str | Path, username: str, password: str) -> bool:
    """Tell whether a username and password pair matches the deployment secrets.

    An unknown username is a plain false, not a raise: the caller shows one message
    for both failures, so that neither tells which of the two was wrong.

    The pair is compared as UTF-8 bytes: `compare_digest` refuses two strings as soon
    as either holds a non-ASCII character, so an accented password would raise where
    it should simply not match.
    """
    users = load_secrets(path)["users"]

    if username in users:
        return hmac.compare_digest(password.encode(), users[username].encode())

    return False


def login(path: str | Path, *, title: str = "Connexion", info: str = "") -> bool:
    """Render the login form and tell whether the session is authenticated.

    Returns True on a session that already authenticated, without rendering anything,
    so the caller guards its own body on the result and stops on False.

    The secrets are read on render, before anyone types, so a deployment pointed at a
    missing or shapeless file fails there rather than on a password submission.

    Four keys are written in `st.session_state`: `username` and `password` by the two
    inputs, then `password_correct` and, on success, `user`, which carries the
    annotator through the rest of the app. The password is dropped from the session as
    soon as it is accepted.
    """
    state = st.session_state
    load_secrets(path)

    def password_entered() -> None:
        username = state["username"]

        if verify_credentials(path, username, state["password"]):
            state["password_correct"] = True
            state["user"] = username
            del state["password"]
        else:
            state["password_correct"] = False

    if state.get("password_correct"):
        return True

    _, col, _ = st.columns([1, 3, 1])

    with col:
        st.markdown(f"### {title}")

        if info:
            st.info(info)

        st.text_input(label="Identifiant", key="username")

        st.text_input(
            label="Mot de passe",
            type="password",
            on_change=password_entered,
            key="password",
        )

    if "password_correct" in state and not state["password_correct"]:
        st.error("😕 Nom d'utilisateur ou mot de passe incorrect")

    return False
