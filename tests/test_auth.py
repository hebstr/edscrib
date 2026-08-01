"""Credential checking and the login shell, on synthetic secrets files."""

import logging
import tomllib
import unicodedata

import pytest
from streamlit.testing.v1 import AppTest

from edscrib.auth import SecretsError, load_secrets, verify_credentials

USERS = {"annotator_a": "sept-chevaux", "annotator_b": "trois-oranges"}


@pytest.fixture
def secrets(tmp_path):
    path = tmp_path / "secrets.toml"
    table = "\n".join(f'{name} = "{word}"' for name, word in USERS.items())
    path.write_text(f"[users]\n{table}\n")
    return path


def test_load_secrets_reads_the_users_table(secrets):
    assert load_secrets(secrets) == USERS


def test_load_secrets_names_the_file_it_cannot_find(tmp_path):
    absent = tmp_path / "absent.toml"

    with pytest.raises(SecretsError, match=r"absent\.toml"):
        load_secrets(absent)


def test_load_secrets_names_a_path_it_cannot_open(tmp_path):
    directory = tmp_path / "secrets.toml"
    directory.mkdir()

    with pytest.raises(SecretsError, match=r"secrets\.toml"):
        load_secrets(directory)


def test_load_secrets_names_a_file_an_editor_wrote_in_another_encoding(tmp_path):
    path = tmp_path / "secrets.toml"
    path.write_bytes('[users]\nannotator_a = "cl\xe9-de-vo\xfbte"\n'.encode("latin-1"))

    with pytest.raises(SecretsError, match=r"secrets\.toml"):
        load_secrets(path)


def test_load_secrets_refuses_a_file_without_a_users_table(tmp_path):
    path = tmp_path / "secrets.toml"
    path.write_text("[server]\nport = 8501\n")

    with pytest.raises(SecretsError, match="users"):
        load_secrets(path)


def test_verify_credentials_accepts_a_matching_pair(secrets):
    assert verify_credentials(secrets, "annotator_a", USERS["annotator_a"])


def test_verify_credentials_rejects_a_wrong_password(secrets):
    assert not verify_credentials(secrets, "annotator_a", USERS["annotator_b"])
    assert not verify_credentials(secrets, "annotator_a", "")


def test_verify_credentials_compares_a_non_ascii_password(tmp_path):
    path = tmp_path / "secrets.toml"
    path.write_text('[users]\nannotator_a = "clé-de-voûte"\n')

    assert verify_credentials(path, "annotator_a", "clé-de-voûte")
    assert not verify_credentials(path, "annotator_a", "cle-de-voute")


def test_verify_credentials_accepts_the_other_spelling_of_an_accented_password(tmp_path):
    path = tmp_path / "secrets.toml"
    decomposed = unicodedata.normalize("NFD", "clé-de-voûte")
    composed = unicodedata.normalize("NFC", "clé-de-voûte")
    path.write_text(f'[users]\nannotator_a = "{decomposed}"\n')

    assert decomposed != composed
    assert verify_credentials(path, "annotator_a", composed)


def test_verify_credentials_rejects_a_non_ascii_password_on_an_ascii_secret(secrets):
    assert not verify_credentials(secrets, "annotator_a", "sept-chevaüx")


def test_verify_credentials_rejects_an_unknown_user_without_raising(secrets):
    assert not verify_credentials(secrets, "annotator_c", USERS["annotator_a"])


def _render(path, options, seed):
    import streamlit as st

    from edscrib.auth import login

    for name, value in seed.items():
        st.session_state.setdefault(name, value)

    st.session_state["authenticated"] = login(path, **options)


def run_login(secrets, **kwargs):
    app = AppTest.from_function(
        _render,
        kwargs={"path": str(secrets), "options": {}, "seed": {}} | kwargs,
    )
    return app.run()


def test_login_renders_the_form_and_holds_the_session_out(secrets):
    app = run_login(secrets)

    assert not app.exception
    assert app.session_state["authenticated"] is None
    assert [field.key for field in app.text_input] == ["username", "password"]
    assert not app.error


def test_login_titles_the_form_with_its_default(secrets):
    app = run_login(secrets)

    assert "Connexion" in app.markdown[0].value


def test_login_titles_the_form_and_shows_the_deployment_notice(secrets):
    app = run_login(
        secrets, options={"title": "Accès", "info": "Le prénom en minuscule."}
    )

    assert "Accès" in app.markdown[0].value
    assert app.info[0].value == "Le prénom en minuscule."


def test_login_shows_no_notice_when_it_was_given_none(secrets):
    app = run_login(secrets)

    assert not app.info


def test_login_admits_a_matching_pair_and_leaves_no_password_in_the_session(secrets):
    app = run_login(secrets)

    app.text_input(key="username").input("annotator_a")
    app.text_input(key="password").input(USERS["annotator_a"]).run()

    assert app.session_state["authenticated"] == "annotator_a"
    assert app.session_state["user"] == "annotator_a"
    assert "password" not in app.session_state


def _reject(secrets, username, password):
    app = run_login(secrets)

    app.text_input(key="username").input(username)
    app.text_input(key="password").input(password).run()

    return app


def test_login_rejects_a_wrong_password_without_naming_which_field(secrets):
    app = _reject(secrets, "annotator_a", USERS["annotator_b"])

    assert app.session_state["authenticated"] is None
    assert not app.session_state["password_correct"]
    assert "user" not in app.session_state

    message = app.error[0].value.lower()

    assert "utilisateur" in message
    assert "mot de passe" in message


def test_login_rejects_an_unknown_user_with_the_same_message(secrets):
    unknown = _reject(secrets, "annotator_c", USERS["annotator_a"])
    wrong = _reject(secrets, "annotator_a", USERS["annotator_b"])

    assert unknown.session_state["authenticated"] is None
    assert unknown.error[0].value == wrong.error[0].value


def test_login_lets_an_authenticated_session_through_without_a_form(secrets):
    app = run_login(secrets, seed={"password_correct": True, "user": "annotator_a"})

    assert app.session_state["authenticated"] == "annotator_a"
    assert app.session_state["user"] == "annotator_a"
    assert not app.text_input


def test_login_refuses_a_session_marked_correct_without_its_user(secrets):
    app = run_login(secrets, seed={"password_correct": True})

    assert app.exception
    assert "user" in app.exception[0].value
    assert not app.text_input


def test_login_refuses_a_session_whose_user_is_not_a_name(secrets):
    """What earns the annotation, on the one key the return is read out of.

    A value taken out of `st.session_state` is `Any`, so returning it would type-check
    every caller against a promise nothing verified. The narrowing is that verification,
    and it covers the same class of setup mistake as the guard above: a session marked
    correct that carries no name anyone can annotate under.
    """
    app = run_login(secrets, seed={"password_correct": True, "user": 42})

    assert app.exception
    assert "user" in app.exception[0].value
    assert not app.text_input


def test_login_withholds_the_form_on_a_missing_secrets_file(tmp_path):
    absent = tmp_path / "absent.toml"

    app = run_login(absent)

    assert not app.exception
    assert not app.text_input
    assert app.session_state["authenticated"] is None
    assert str(absent) not in app.error[0].value


def test_login_names_neither_the_secrets_file_nor_what_it_could_not_parse(tmp_path):
    path = tmp_path / "secrets.toml"
    path.write_text('[users]\nannotator_a = "unclosed\n')

    app = run_login(path)
    message = app.error[0].value

    assert not app.exception
    assert not app.text_input
    assert str(path) not in message
    assert "unclosed" not in message


def test_login_leaks_nothing_when_the_path_cannot_be_opened(tmp_path):
    directory = tmp_path / "secrets.toml"
    directory.mkdir()

    app = run_login(directory)

    assert not app.exception
    assert not app.text_input
    assert str(directory) not in app.error[0].value


def test_login_leaks_nothing_when_the_file_is_in_another_encoding(tmp_path):
    path = tmp_path / "secrets.toml"
    path.write_bytes('[users]\nannotator_a = "cl\xe9-de-vo\xfbte"\n'.encode("latin-1"))

    app = run_login(path)

    assert not app.exception
    assert not app.text_input
    assert str(path) not in app.error[0].value


def test_login_still_serves_the_form_when_one_annotator_entry_is_unusable(tmp_path):
    path = tmp_path / "secrets.toml"
    path.write_text('[users]\nannotator_a = "sept-chevaux"\nannotator_c = ""\n')

    app = run_login(path)

    assert not app.exception
    assert not app.error
    assert [field.key for field in app.text_input] == ["username", "password"]


def test_load_secrets_keeps_what_raised_underneath_it(tmp_path):
    path = tmp_path / "secrets.toml"
    path.write_text('[users]\nannotator_a = "unclosed\n')

    with pytest.raises(SecretsError) as caught:
        load_secrets(path)

    assert isinstance(caught.value.__cause__, tomllib.TOMLDecodeError)


def test_load_secrets_drops_an_empty_password_and_names_it_in_the_log(tmp_path, caplog):
    path = tmp_path / "secrets.toml"
    path.write_text('[users]\nannotator_a = "sept-chevaux"\nannotator_c = ""\n')

    with caplog.at_level(logging.WARNING, logger="edscrib.auth"):
        assert load_secrets(path) == {"annotator_a": "sept-chevaux"}

    assert "annotator_c" in caplog.text


def test_load_secrets_drops_a_non_string_password(tmp_path):
    path = tmp_path / "secrets.toml"
    path.write_text('[users]\nannotator_a = "sept-chevaux"\nannotator_c = 1234\n')

    assert load_secrets(path) == {"annotator_a": "sept-chevaux"}


def test_load_secrets_still_serves_the_annotators_it_can_read(tmp_path):
    path = tmp_path / "secrets.toml"
    path.write_text('[users]\nannotator_a = "sept-chevaux"\nannotator_c = ""\n')

    assert verify_credentials(path, "annotator_a", "sept-chevaux")


def test_verify_credentials_never_admits_an_empty_password(tmp_path):
    path = tmp_path / "secrets.toml"
    path.write_text('[users]\nannotator_a = "sept-chevaux"\nannotator_c = ""\n')

    assert not verify_credentials(path, "annotator_c", "")


def test_load_secrets_refuses_an_empty_users_table(tmp_path):
    path = tmp_path / "secrets.toml"
    path.write_text("[users]\n")

    with pytest.raises(SecretsError, match=r"secrets\.toml"):
        load_secrets(path)


def test_load_secrets_refuses_a_users_key_that_is_not_a_table(tmp_path):
    path = tmp_path / "secrets.toml"
    path.write_text('users = "annotator_a"\n')

    with pytest.raises(SecretsError, match="users"):
        load_secrets(path)


def test_load_secrets_names_the_file_it_cannot_parse(tmp_path):
    path = tmp_path / "secrets.toml"
    path.write_text('[users]\nannotator_a = "unclosed\n')

    with pytest.raises(SecretsError, match=r"secrets\.toml"):
        load_secrets(path)


def test_login_drops_the_password_from_the_session_when_it_is_rejected(secrets):
    app = run_login(secrets)

    app.text_input(key="username").input("annotator_TYPO")
    app.text_input(key="password").input(USERS["annotator_a"]).run()

    assert not app.session_state["password_correct"]
    assert not app.session_state["password"]


def test_login_drops_the_password_when_the_secrets_file_breaks_before_submission(
    secrets,
):
    app = run_login(secrets)

    secrets.write_text("[users]\n")

    app.text_input(key="username").input("annotator_a")
    app.text_input(key="password").input(USERS["annotator_a"]).run()

    assert not app.exception
    assert not app.text_input
    assert "password" not in app.session_state


def test_login_still_admits_a_correct_pair_after_a_rejected_attempt(secrets):
    app = run_login(secrets)

    app.text_input(key="username").input("annotator_TYPO")
    app.text_input(key="password").input(USERS["annotator_a"]).run()

    app.text_input(key="username").input("annotator_a")
    app.text_input(key="password").input(USERS["annotator_a"]).run()

    assert app.session_state["authenticated"] == "annotator_a"
    assert app.session_state["user"] == "annotator_a"


def test_login_lets_an_authenticated_session_survive_a_broken_secrets_file(secrets):
    app = run_login(secrets)

    app.text_input(key="username").input("annotator_a")
    app.text_input(key="password").input(USERS["annotator_a"]).run()

    assert app.session_state["authenticated"]

    secrets.write_text("[users]\n")
    app.run()

    assert not app.exception
    assert app.session_state["authenticated"]

    secrets.write_text('[users]\nannotator_a = "unclosed\n')
    app.run()

    assert not app.exception
    assert app.session_state["authenticated"]
