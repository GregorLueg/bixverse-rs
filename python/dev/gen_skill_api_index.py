"""Generate `bixverse/_skill/references/api-index.md` from the package.

Run from `python/`, with the package installed in the environment:

    uv run python dev/gen_skill_api_index.py

The index lists every name in each public module's `__all__` with the first
line of its docstring, grouped by module. The output is committed and never
edited by hand. `tests/test_agent_skill.py` fails when the committed index
differs from a fresh render, so re-run this after any change to a public name,
a docstring summary or the version.
"""

from __future__ import annotations

import importlib
import inspect
from pathlib import Path
from types import ModuleType

from beartype import beartype

###########
# Globals #
###########

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "bixverse" / "_skill" / "references" / "api-index.md"
# Public modules in reading order; `bixverse` itself comes first.
MODULES: tuple[str, ...] = (
    "io",
    "qc",
    "doublets",
    "pp",
    "graph",
    "embed",
    "integrate",
    "de",
    "annotate",
    "pathways",
    "lr",
    "nmf",
    "dialogue",
    "traj",
    "trees",
    "metacells",
    "mm",
    "atac",
    "pipeline",
    "pl",
    "loader",
    "nn",
    "params",
)
# Modules that need an optional extra; a failed import is listed, not fatal.
OPTIONAL: frozenset[str] = frozenset({"embed", "pl", "nn"})
NOT_IMPORTABLE = "Not importable here"
HEADER = """# bixverse (Python) API index

Every public entry point, grouped by module. Generated from each module's
`__all__` and docstrings by `dev/gen_skill_api_index.py`. Do not edit by hand.

Grep this file to check whether a function exists before calling it, then
`help(bv.<module>.<name>)` or `inspect.signature()` for how to call it.
"""

###########
# Helpers #
###########


@beartype
def first_line(obj: object) -> str:
    """Return the first line of an object's docstring, or an empty string.

    Args:
        obj: Any object.

    Returns:
        The summary line, stripped.

    Examples:
        >>> first_line(first_line)
        "Return the first line of an object's docstring, or an empty string."
    """
    doc = inspect.getdoc(obj) or ""
    return doc.strip().splitlines()[0].strip() if doc.strip() else ""


@beartype
def section(title: str, module: ModuleType) -> list[str]:
    """Render one module as a markdown section.

    Args:
        title: Section heading, e.g. `bv.qc`.
        module: The imported module.

    Returns:
        Markdown lines: heading, module summary, one bullet per public name.

    Examples:
        >>> import bixverse as bv
        >>> section("bv", bv)[0]
        '## bv'
    """
    lines = [f"## {title}", ""]
    summary = first_line(module)
    if summary:
        lines += [summary, ""]
    names = list(getattr(module, "__all__", None) or [])
    if not names:
        # Without `__all__`, only names defined in the module are public; the
        # rest are imports.
        names = sorted(
            n
            for n, obj in vars(module).items()
            if not n.startswith("_")
            and getattr(obj, "__module__", None) == module.__name__
        )
    for name in names:
        obj = getattr(module, name, None)
        if obj is None or isinstance(obj, ModuleType):
            continue
        text = first_line(obj) if callable(obj) else ""
        lines.append(f"- `{name}`: {text}" if text else f"- `{name}`")
    lines.append("")
    return lines


########
# Main #
########


@beartype
def render() -> str:
    """Render the API index from the installed package.

    Returns:
        The markdown text, as written to `OUT`. An optional module that does
        not import here is listed as such, so the text depends on the extras
        installed.

    Examples:
        >>> render().startswith("# bixverse (Python) API index")  # doctest: +SKIP
        True
    """
    import bixverse

    lines = [HEADER, f"Version: {bixverse.__version__}.", ""]
    lines += section("bv", bixverse)
    for name in MODULES:
        try:
            module = importlib.import_module(f"bixverse.{name}")
        except ImportError as err:
            if name not in OPTIONAL:
                raise
            lines += [f"## bv.{name}", "", f"{NOT_IMPORTABLE}: {err}.", ""]
            continue
        lines += section(f"bv.{name}", module)
    return "\n".join(lines).rstrip() + "\n"


@beartype
def main() -> None:
    """Write the API index.

    Examples:
        >>> main()  # doctest: +SKIP
    """
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(render())
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
