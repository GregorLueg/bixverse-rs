"""The bixverse agent skill: install the bundled skill where an agent finds it.

The skill (`bixverse/_skill/`: `SKILL.md` plus `references/`) teaches coding
agents how to use the package. It ships in the wheel, so the installed copy
always matches the installed version. `SKILL.md` follows the Agent Skills open
standard (agentskills.io), so the same files serve every agent; only the
directory they are discovered in differs.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Literal

from beartype import beartype

from .._bixverse import __version__

###########
# Globals #
###########

logger = logging.getLogger(__name__)

Agent = Literal["claude", "codex"]
Scope = Literal["user", "project"]

SKILL_NAME: str = "bixverse-python"
SOURCE_DIR: Path = Path(__file__).resolve().parent.parent / "_skill"
ENTRY_FILE: str = "SKILL.md"
STAMP_FILE: str = ".bixverse-version"
# Codex scans the open standard's `.agents/skills`, not `~/.codex/skills`.
# Project paths are relative to the working directory.
SKILL_DIRS: dict[tuple[str, str], str] = {
    ("claude", "user"): "~/.claude/skills",
    ("claude", "project"): ".claude/skills",
    ("codex", "user"): "~/.agents/skills",
    ("codex", "project"): ".agents/skills",
}

###########
# Helpers #
###########


@beartype
def skill_dir(agent: Agent = "claude", scope: Scope = "user") -> Path:
    """Return the directory an agent discovers skills in.

    Args:
        agent: The coding agent.
        scope: `"user"` for every project of this user, `"project"` for the
            project in the working directory.

    Returns:
        The skills directory, without the `bixverse-python` subdirectory.

    Examples:
        >>> skill_dir("codex", "project").as_posix()
        '.agents/skills'
    """
    return Path(SKILL_DIRS[(agent, scope)]).expanduser()


@beartype
def installed_version(target: Path) -> str | None:
    """Read the package version an installed skill was copied from.

    Args:
        target: The installed skill directory.

    Returns:
        The stamped version, or `None` for an install without a stamp.

    Examples:
        >>> installed_version(Path("nowhere")) is None
        True
    """
    stamp = target / STAMP_FILE
    return stamp.read_text().strip() if stamp.is_file() else None


@beartype
def installed_agent_skills() -> list[Path]:
    """List the copied skill installs in the user and project locations.

    Symlinked installs track their source, so they are left out.

    Returns:
        One path per installed `bixverse-python` directory.

    Examples:
        >>> installed_agent_skills()  # doctest: +SKIP
        [PosixPath('/home/me/.claude/skills/bixverse-python')]
    """
    found = []
    for dest in SKILL_DIRS.values():
        target = Path(dest).expanduser() / SKILL_NAME
        try:
            if target.is_dir() and not target.is_symlink():
                found.append(target)
        except OSError:
            continue
    return found


@beartype
def stale_agent_skills() -> list[Path]:
    """List installed skills that lag the package.

    A copied skill does not follow a package upgrade, so an agent would keep
    reading the API of the old version. Called once at import; it only stats a
    handful of paths.

    Returns:
        The skill directories whose stamp differs from `bixverse.__version__`,
        including installs without a stamp.

    Examples:
        >>> stale_agent_skills()  # doctest: +SKIP
        [PosixPath('/home/me/.claude/skills/bixverse-python')]
    """
    stale = []
    for target in installed_agent_skills():
        try:
            if installed_version(target) != __version__:
                stale.append(target)
        except OSError:
            continue
    return stale


@beartype
def warn_stale_agent_skills() -> None:
    """Log one warning per installed skill that lags the package.

    Examples:
        >>> warn_stale_agent_skills()  # doctest: +SKIP
    """
    for target in stale_agent_skills():
        logger.warning(
            "The bixverse agent skill at %s is from version %s, the package is "
            "%s. Run `bixverse skill install --force` (with `--project` for a "
            "project install) to update it.",
            target,
            installed_version(target) or "unknown",
            __version__,
        )


########
# Main #
########


@beartype
def install_agent_skill(
    dest: str | Path | None = None,
    *,
    agent: Agent = "claude",
    scope: Scope = "user",
    overwrite: bool = False,
) -> Path:
    """Install the bixverse agent skill where your coding agent looks for it.

    The skill is versioned with the package; re-run this after upgrading
    bixverse to keep the agent in sync with the code. The install is stamped
    with the package version, and `import bixverse` warns when an installed
    skill is older than the package.

    A project install ties the skill to the project's environment, so two
    projects on different bixverse versions each see their own API. Run it
    from the project root.

    Args:
        dest: Skills directory to install into; the skill lands in a
            `bixverse-python` subdirectory. Overrides `agent` and `scope`.
        agent: `"claude"` installs to `.claude/skills`; `"codex"` to
            `.agents/skills`, the open standard's location, which Codex and
            other agents scan.
        scope: `"user"` installs under the home directory, `"project"` under
            the working directory.
        overwrite: Replace an existing installation (a directory or a symlink).

    Returns:
        The installed skill directory.

    Raises:
        FileNotFoundError: If the installed package has no bundled skill.
        FileExistsError: If the skill is already installed and `overwrite` is
            false.

    Examples:
        >>> bv.install_agent_skill()  # doctest: +SKIP
        PosixPath('/home/me/.claude/skills/bixverse-python')
        >>> bv.install_agent_skill(agent="codex", scope="project")  # doctest: +SKIP
        PosixPath('.agents/skills/bixverse-python')
    """
    if not (SOURCE_DIR / ENTRY_FILE).is_file():
        raise FileNotFoundError(
            f"The agent skill was not found at {SOURCE_DIR}. The package was "
            "probably built without it."
        )

    base = Path(dest).expanduser() if dest is not None else skill_dir(agent, scope)
    target = base / SKILL_NAME
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
    (target / STAMP_FILE).write_text(__version__ + "\n")

    logger.info("bixverse agent skill %s installed to %s", __version__, target)
    logger.info("Start a new agent session to pick it up.")
    return target
