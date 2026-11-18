"""Installing the bundled agent skill."""

import pytest

import bixverse as bv
from bixverse._core.agent_skill import SOURCE_DIR, _strip_frontmatter


def test_skill_is_bundled():
    assert (SOURCE_DIR / "SKILL.md").is_file()
    references = {p.name for p in (SOURCE_DIR / "references").glob("*.md")}
    assert {"api-index.md", "conventions.md", "single-cell.md"} <= references


def test_claude_install_keeps_frontmatter(tmp_path):
    target = bv.install_agent_skill(tmp_path, agent="claude")
    assert target == tmp_path / "bixverse-python"
    entry = (target / "SKILL.md").read_text()
    assert entry.startswith("---\nname: bixverse-python\n")
    assert (target / "references" / "api-index.md").is_file()


def test_generic_install_writes_agents_md_without_frontmatter(tmp_path):
    target = bv.install_agent_skill(tmp_path, agent="generic")
    assert not (target / "SKILL.md").exists()
    entry = (target / "AGENTS.md").read_text()
    assert entry.startswith("# Using bixverse in Python")


def test_generic_needs_dest():
    with pytest.raises(ValueError, match="dest"):
        bv.install_agent_skill(agent="generic")


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


def test_strip_frontmatter():
    assert _strip_frontmatter("---\nname: x\n---\n\n# T\n") == "# T\n"
    assert _strip_frontmatter("# T\n") == "# T\n"
    assert _strip_frontmatter("---\nunclosed\n") == "---\nunclosed\n"
