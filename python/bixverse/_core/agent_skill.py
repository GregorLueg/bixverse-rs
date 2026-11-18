"""The bixverse agent skill: install the bundled skill where an agent finds it.

The skill (`bixverse/_skill/`: `SKILL.md` plus `references/`) teaches coding
agents how to use the package. It ships in the wheel, so the installed copy
always matches the installed version.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Literal

from beartype import beartype

###########
# Globals #
###########

logger = logging.getLogger(__name__)

SKILL_NAME: str = "bixverse-python"
SOURCE_DIR: Path = Path(__file__).resolve().parent.parent / "_skill"
ENTRY_FILE: str = "SKILL.md"
GENERIC_ENTRY_FILE: str = "AGENTS.md"
DEFAULT_DEST: dict[str, str] = {
    "claude": "~/.claude/skills",
    "codex": "~/.codex/skills",
}
FRONTMATTER_FENCE: str = "---"

###########
# Helpers #
###########


@beartype
def _strip_frontmatter(text: str) -> str:
    """Strip a leading YAML frontmatter block from markdown.

    Claude Code reads `name` and `description` from the frontmatter; other
    agents do not, so for them it is noise at the top of the file.

    Args:
        text: The file contents.

    Returns:
        The contents without the frontmatter and the blank lines after it.
        Text without a closed frontmatter block is returned as is.

    Examples:
        >>> _strip_frontmatter("---\\nname: x\\n---\\n\\n# Title\\n")
        '# Title\\n'
        >>> _strip_frontmatter("# Title\\n")
        '# Title\\n'
    """
    lines = text.splitlines(keepends=True)
    if not lines or lines[0].strip() != FRONTMATTER_FENCE:
        return text
    for i, line in enumerate(lines[1:], start=1):
        if line.strip() == FRONTMATTER_FENCE:
            rest = lines[i + 1 :]
            while rest and not rest[0].strip():
                rest = rest[1:]
            return "".join(rest)
    return text


########
# Main #
########


@beartype
def install_agent_skill(
    dest: str | Path | None = None,
    *,
    agent: Literal["claude", "codex", "generic"] = "claude",
    overwrite: bool = False,
) -> Path:
    """Install the bixverse agent skill where your coding agent looks for it.

    The skill is versioned with the package; re-run this after upgrading
    bixverse to keep the agent in sync with the code. The reference files are
    plain markdown and work with any agent. Only the entry file differs:
    `"claude"` writes `SKILL.md` with the frontmatter Claude Code uses to decide
    when to load the skill; `"codex"` and `"generic"` write `AGENTS.md` without
    it, and the agent has to be pointed at the directory.

    Args:
        dest: Directory to install into; the skill lands in a
            `bixverse-python` subdirectory. `None` uses `~/.claude/skills` for
            `"claude"` and `~/.codex/skills` for `"codex"`; required for
            `"generic"`.
        agent: Which agent the entry file is written for.
        overwrite: Replace an existing installation (a directory or a symlink).

    Returns:
        The installed skill directory.

    Raises:
        ValueError: If `agent="generic"` without `dest`.
        FileNotFoundError: If the installed package has no bundled skill.
        FileExistsError: If the skill is already installed and `overwrite` is
            false.

    Examples:
        >>> bv.install_agent_skill()  # doctest: +SKIP
        PosixPath('/home/me/.claude/skills/bixverse-python')
        >>> bv.install_agent_skill("agents", agent="generic")  # doctest: +SKIP
        PosixPath('agents/bixverse-python')
    """
    if dest is None:
        if agent not in DEFAULT_DEST:
            raise ValueError('`dest` is required when `agent="generic"`.')
        dest = DEFAULT_DEST[agent]
    if not (SOURCE_DIR / ENTRY_FILE).is_file():
        raise FileNotFoundError(
            f"The agent skill was not found at {SOURCE_DIR}. The package was "
            "probably built without it."
        )

    target = Path(dest).expanduser() / SKILL_NAME
    if target.exists() or target.is_symlink():
        if not overwrite:
            raise FileExistsError(
                f"A bixverse skill already exists at {target}. "
                "Re-run with overwrite=True."
            )
        if target.is_symlink() or target.is_file():
            target.unlink()
        else:
            shutil.rmtree(target)

    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(
        SOURCE_DIR, target, ignore=shutil.ignore_patterns("__pycache__", "*.py[co]")
    )

    entry = target / ENTRY_FILE
    if agent != "claude":
        generic = target / GENERIC_ENTRY_FILE
        generic.write_text(_strip_frontmatter(entry.read_text()))
        entry.unlink()
        entry = generic

    logger.info("bixverse agent skill installed to %s", target)
    if agent == "claude":
        logger.info("Start a new Claude Code session to pick it up.")
    else:
        logger.info(
            "No auto-discovery outside Claude Code. Point your agent at %s, or "
            "symlink it into the project you are working in.",
            entry,
        )
    return target
