"""Installing the bundled agent skill."""

import importlib.util
from pathlib import Path

import pytest

import bixverse as bv
from bixverse._cli import main
from bixverse._core import agent_skill
from bixverse._core.agent_skill import (
    SOURCE_DIR,
    STAMP_FILE,
    skill_dir,
    stale_agent_skills,
)

GENERATOR = Path(__file__).resolve().parent.parent / "dev" / "gen_skill_api_index.py"


def _load_generator():
    spec = importlib.util.spec_from_file_location("gen_skill_api_index", GENERATOR)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def fake_home(tmp_path, monkeypatch):
    """Point the home directory and the working directory into tmp_path."""
    home = tmp_path / "home"
    project = tmp_path / "project"
    home.mkdir()
    project.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    monkeypatch.chdir(project)
    return home, project


def test_skill_is_bundled():
    assert (SOURCE_DIR / "SKILL.md").is_file()
    references = {p.name for p in (SOURCE_DIR / "references").glob("*.md")}
    assert {"api-index.md", "conventions.md", "single-cell.md"} <= references


@pytest.mark.parametrize(
    ("agent", "scope", "expected"),
    [
        ("claude", "user", "home/.claude/skills"),
        ("claude", "project", "project/.claude/skills"),
        ("codex", "user", "home/.agents/skills"),
        ("codex", "project", "project/.agents/skills"),
    ],
)
def test_install_locations(fake_home, agent, scope, expected):
    home, _ = fake_home
    target = bv.install_agent_skill(agent=agent, scope=scope)
    assert target.resolve() == (home.parent / expected / "bixverse-python").resolve()
    assert target.parent == skill_dir(agent, scope)


def test_every_agent_gets_skill_md_with_frontmatter(tmp_path):
    for agent in ("claude", "codex"):
        target = bv.install_agent_skill(tmp_path / agent, agent=agent)
        entry = (target / "SKILL.md").read_text()
        assert entry.startswith("---\nname: bixverse-python\n")
        assert "\ndescription: " in entry
        assert (target / "references" / "api-index.md").is_file()


def test_dest_overrides_agent_and_scope(tmp_path):
    target = bv.install_agent_skill(tmp_path, agent="codex", scope="project")
    assert target == tmp_path / "bixverse-python"


def test_existing_install_needs_overwrite(tmp_path):
    bv.install_agent_skill(tmp_path)
    with pytest.raises(FileExistsError, match="overwrite"):
        bv.install_agent_skill(tmp_path)
    marker = tmp_path / "bixverse-python" / "stale.md"
    marker.write_text("old")
    bv.install_agent_skill(tmp_path, overwrite=True)
    assert not marker.exists()


def test_overwrite_replaces_a_symlink_not_its_target(tmp_path):
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    (checkout / "keep.md").write_text("dev copy")
    link = tmp_path / "skills" / "bixverse-python"
    link.parent.mkdir()
    link.symlink_to(checkout)
    bv.install_agent_skill(tmp_path / "skills", overwrite=True)
    assert not link.is_symlink()
    assert (checkout / "keep.md").read_text() == "dev copy"


def test_install_stamps_the_package_version(tmp_path):
    target = bv.install_agent_skill(tmp_path)
    assert (target / STAMP_FILE).read_text().strip() == bv.__version__


def test_fresh_install_is_not_stale(fake_home):
    bv.install_agent_skill()
    bv.install_agent_skill(agent="codex", scope="project")
    assert stale_agent_skills() == []


def test_older_or_unstamped_install_is_stale(fake_home):
    user = bv.install_agent_skill()
    project = bv.install_agent_skill(agent="codex", scope="project")
    (user / STAMP_FILE).write_text("0.0.1\n")
    (project / STAMP_FILE).unlink()
    stale = {p.resolve() for p in stale_agent_skills()}
    assert stale == {user.resolve(), project.resolve()}


def test_symlinked_install_is_not_stale(fake_home):
    home, _ = fake_home
    checkout = home / "checkout"
    checkout.mkdir()
    link = home / ".claude" / "skills" / "bixverse-python"
    link.parent.mkdir(parents=True)
    link.symlink_to(checkout)
    assert stale_agent_skills() == []


def test_stale_install_warns(fake_home, caplog):
    target = bv.install_agent_skill()
    (target / STAMP_FILE).write_text("0.0.1\n")
    with caplog.at_level("WARNING", logger="bixverse"):
        agent_skill.warn_stale_agent_skills()
    assert "0.0.1" in caplog.text
    assert "--force" in caplog.text


def test_cli_install_and_status(fake_home, capsys):
    _, project = fake_home
    assert main(["skill", "install", "--project", "--agent", "codex"]) == 0
    assert (project / ".agents" / "skills" / "bixverse-python" / "SKILL.md").is_file()
    assert main(["skill", "status"]) == 0
    out = capsys.readouterr().out
    assert "ok" in out
    assert bv.__version__ in out


def test_cli_status_flags_a_stale_install(fake_home, capsys):
    target = bv.install_agent_skill()
    (target / STAMP_FILE).write_text("0.0.1\n")
    assert main(["skill", "status"]) == 1
    assert "stale" in capsys.readouterr().out


def test_cli_install_without_force_fails_on_existing(fake_home):
    main(["skill", "install"])
    with pytest.raises(SystemExit) as exit_info:
        main(["skill", "install"])
    assert exit_info.value.code == 1
    assert main(["skill", "install", "--force"]) == 0


def test_api_index_is_current():
    generator = _load_generator()
    rendered = generator.render()
    if generator.NOT_IMPORTABLE in rendered:
        pytest.skip("an optional module does not import, so the index differs")
    assert rendered == generator.OUT.read_text(), (
        "references/api-index.md is stale: run "
        "`uv run python dev/gen_skill_api_index.py`"
    )
