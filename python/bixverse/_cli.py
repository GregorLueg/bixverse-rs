"""The `bixverse` command line: manage the bundled agent skill.

    bixverse skill install [--agent claude|codex] [--project] [--dest DIR] [--force]
    bixverse skill status

Run it through the project's environment (`uv run bixverse ...`) so the skill
matches the bixverse version the project uses; `uvx bixverse` installs the
skill of the latest release instead.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from beartype import beartype

from ._bixverse import __version__
from ._core.agent_skill import (
    install_agent_skill,
    installed_agent_skills,
    installed_version,
)

###########
# Helpers #
###########


@beartype
def _parser() -> argparse.ArgumentParser:
    """Build the argument parser.

    Returns:
        The parser for `bixverse skill install` and `bixverse skill status`.
    """
    parser = argparse.ArgumentParser(prog="bixverse")
    parser.add_argument("--version", action="version", version=__version__)
    commands = parser.add_subparsers(dest="command", required=True)
    skill = commands.add_parser("skill", help="manage the bundled agent skill")
    actions = skill.add_subparsers(dest="action", required=True)

    install = actions.add_parser("install", help="install or update the skill")
    install.add_argument(
        "--agent",
        choices=["claude", "codex"],
        default="claude",
        help="claude: .claude/skills; codex: .agents/skills (default: claude)",
    )
    install.add_argument(
        "--project",
        action="store_true",
        help="install under the working directory instead of the home directory",
    )
    install.add_argument("--dest", help="skills directory; overrides --agent")
    install.add_argument(
        "--force", action="store_true", help="replace an existing install"
    )
    actions.add_parser("status", help="list installed skills and their versions")
    return parser


@beartype
def _status() -> int:
    """Print every installed skill with its version.

    Returns:
        1 if any install lags the package, else 0.
    """
    installs = installed_agent_skills()
    if not installs:
        print("No bixverse agent skill installed. Run `bixverse skill install`.")
        return 0
    stale = False
    for target in installs:
        found = installed_version(target) or "unknown"
        current = found == __version__
        stale |= not current
        print(f"{'ok' if current else 'stale':<6} {found:<10} {target}")
    print(f"package: {__version__}")
    return int(stale)


########
# Main #
########


@beartype
def main(argv: Sequence[str] | None = None) -> int:
    """Run the `bixverse` command line.

    Args:
        argv: Arguments without the program name; `None` reads `sys.argv`.

    Returns:
        The exit code.

    Examples:
        >>> main(["skill", "install", "--project"])  # doctest: +SKIP
        0
    """
    parser = _parser()
    args = parser.parse_args(argv)
    if args.action == "status":
        return _status()
    try:
        target = install_agent_skill(
            args.dest,
            agent=args.agent,
            scope="project" if args.project else "user",
            overwrite=args.force,
        )
    except FileExistsError:
        parser.exit(1, "The skill is already installed there. Pass --force.\n")
    print(target)
    return 0
