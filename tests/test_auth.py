"""Credential checking and the login shell, on synthetic secrets files."""

import pytest
from streamlit.testing.v1 import AppTest

from edscrib.auth import load_secrets, verify_credentials

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

    with pytest.raises(FileNotFoundError, match=r"absent\.toml"):
        load_secrets(absent)


def test_load_secrets_refuses_a_file_without_a_users_table(tmp_path):
    path = tmp_path / "secrets.toml"
    path.write_text("[server]\nport = 8501\n")

    with pytest.raises(KeyError, match="users"):
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
    assert not app.session_state["authenticated"]
    assert [field.key for field in app.text_input] == ["username", "password"]


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

    assert app.session_state["authenticated"]
    assert app.session_state["user"] == "annotator_a"
    assert "password" not in app.session_state


def test_login_rejects_a_wrong_password_without_naming_which_field(secrets):
    app = run_login(secrets)

    app.text_input(key="username").input("annotator_a")
    app.text_input(key="password").input(USERS["annotator_b"]).run()

    assert not app.session_state["authenticated"]
    assert not app.session_state["password_correct"]
    assert "user" not in app.session_state
    assert app.error


def test_login_rejects_an_unknown_user_with_the_same_message(secrets):
    app = run_login(secrets)

    app.text_input(key="username").input("annotator_c")
    app.text_input(key="password").input(USERS["annotator_a"]).run()

    assert not app.session_state["authenticated"]
    assert app.error


def test_login_lets_an_authenticated_session_through_without_a_form(secrets):
    app = run_login(secrets, seed={"password_correct": True, "user": "annotator_a"})

    assert app.session_state["authenticated"]
    assert app.session_state["user"] == "annotator_a"
    assert not app.text_input


def test_login_refuses_a_session_marked_correct_without_its_user(secrets):
    app = run_login(secrets, seed={"password_correct": True})

    assert app.exception
    assert "user" in app.exception[0].value
    assert not app.text_input


def test_login_fails_on_a_missing_secrets_file_before_anything_is_typed(tmp_path):
    app = run_login(tmp_path / "absent.toml")

    assert app.exception
    assert not app.text_input


def test_load_secrets_refuses_an_empty_password(tmp_path):
    path = tmp_path / "secrets.toml"
    path.write_text('[users]\nannotator_a = "sept-chevaux"\nannotator_c = ""\n')

    with pytest.raises(ValueError, match="annotator_c"):
        load_secrets(path)


def test_load_secrets_refuses_a_non_string_password(tmp_path):
    path = tmp_path / "secrets.toml"
    path.write_text("[users]\nannotator_a = 1234\n")

    with pytest.raises(ValueError, match="annotator_a"):
        load_secrets(path)


def test_verify_credentials_never_admits_an_empty_password(tmp_path):
    path = tmp_path / "secrets.toml"
    path.write_text('[users]\nannotator_c = ""\n')

    with pytest.raises(ValueError):
        verify_credentials(path, "annotator_c", "")


def test_load_secrets_refuses_an_empty_users_table(tmp_path):
    path = tmp_path / "secrets.toml"
    path.write_text("[users]\n")

    with pytest.raises(ValueError, match=r"secrets\.toml"):
        load_secrets(path)


def test_load_secrets_refuses_a_users_key_that_is_not_a_table(tmp_path):
    path = tmp_path / "secrets.toml"
    path.write_text('users = "annotator_a"\n')

    with pytest.raises(KeyError, match="users"):
        load_secrets(path)


def test_load_secrets_names_the_file_it_cannot_parse(tmp_path):
    path = tmp_path / "secrets.toml"
    path.write_text('[users]\nannotator_a = "unclosed\n')

    with pytest.raises(ValueError, match=r"secrets\.toml"):
        load_secrets(path)


def test_login_drops_the_password_from_the_session_when_it_is_rejected(secrets):
    app = run_login(secrets)

    app.text_input(key="username").input("annotator_TYPO")
    app.text_input(key="password").input(USERS["annotator_a"]).run()

    assert not app.session_state["password_correct"]
    assert not app.session_state["password"]


def test_login_still_admits_a_correct_pair_after_a_rejected_attempt(secrets):
    app = run_login(secrets)

    app.text_input(key="username").input("annotator_TYPO")
    app.text_input(key="password").input(USERS["annotator_a"]).run()

    app.text_input(key="username").input("annotator_a")
    app.text_input(key="password").input(USERS["annotator_a"]).run()

    assert app.session_state["authenticated"]
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
