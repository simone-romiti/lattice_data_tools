#!/usr/bin/env python3
"""
generate_docs.py
================
Pipeline to auto-generate a Quarto **book** documentation project from a
Python library.

Usage
-----
    python generate_docs.py --src /path/to/library --out /path/to/docs_output

Options
-------
--src      Root directory of the Python library (required)
--out      Output directory for the Quarto project (created if absent, required)
--github   Override the GitHub base URL for source links.
           If omitted the script reads it from pyproject.toml in --src.
--branch   Git branch name used in source links (default: main)

The script:
1. Reads pyproject.toml for the project name and GitHub repository URL
2. Walks the library tree and parses every .py file with ``ast``
3. Extracts module-level docstrings, class/function signatures and docstrings
4. Writes one .qmd file per .py file, mirroring the folder structure
5. __init__.py files become chapter/section index pages (index.qmd)
6. Builds _quarto.yml as a Quarto *book* project with a nested chapter tree
"""

import ast
import argparse
import os
import re
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Runtime config  (populated by read_pyproject + CLI)
# ---------------------------------------------------------------------------
_CONFIG: dict = {
    "project_name": "lattice_data_tools",
    "github_repo":  "https://github.com/simone-itpkaon/lattice_data_tools/",
    "branch":       "main",
}

# Directories to skip entirely
IGNORE_NAMES = {
    "__pycache__",
    ".git",
    "doc",
    "legacy",
    "old",
}


# ---------------------------------------------------------------------------
# pyproject.toml reader  (stdlib tomllib / tomli fallback)
# ---------------------------------------------------------------------------

def _load_toml(path: Path) -> dict:
    """Return parsed TOML dict, or {} on failure."""
    # Python 3.11+ has tomllib in stdlib
    try:
        import tomllib                          # type: ignore
        with open(path, "rb") as fh:
            return tomllib.load(fh)
    except ModuleNotFoundError:
        pass
    # Python 3.10 and earlier: try tomli (pip install tomli)
    try:
        import tomli as tomllib                 # type: ignore
        with open(path, "rb") as fh:
            return tomllib.load(fh)
    except ModuleNotFoundError:
        pass
    # Last resort: minimal hand-rolled key=value parser (no arrays/tables)
    return _simple_toml_parse(path)


def _simple_toml_parse(path: Path) -> dict:
    """Very small TOML parser that handles only [sections] and key = "value"."""
    result: dict = {}
    section: dict = result
    section_path: list = []
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("[") and not line.startswith("[["):
                # e.g. [project.urls]
                header = line.strip("[]").strip()
                keys = header.split(".")
                section = result
                for k in keys:
                    section = section.setdefault(k, {})
                section_path = keys
                continue
            if "=" in line:
                k, _, v = line.partition("=")
                k = k.strip()
                v = v.strip().strip('"').strip("'")
                section[k] = v
    except Exception:
        pass
    return result


def read_pyproject(src_dir: Path) -> tuple[str, str]:
    """
    Return (project_name, github_url) parsed from src_dir/pyproject.toml.
    github_url is the bare repo URL, e.g. https://github.com/user/repo
    Returns ("", "") if not found.
    """
    toml_path = src_dir / "pyproject.toml"
    if not toml_path.exists():
        return "", ""

    data = _load_toml(toml_path)
    if not data:
        return "", ""

    # Project name: [project] name  or  [tool.poetry] name
    name = (
        data.get("project", {}).get("name", "")
        or data.get("tool", {}).get("poetry", {}).get("name", "")
        or ""
    )

    # GitHub URL: check several common conventions
    url = ""
    candidates = [
        # PEP 621 [project.urls]
        data.get("project", {}).get("urls", {}).get("repository", ""),
        data.get("project", {}).get("urls", {}).get("Repository", ""),
        data.get("project", {}).get("urls", {}).get("Homepage", ""),
        data.get("project", {}).get("urls", {}).get("homepage", ""),
        data.get("project", {}).get("urls", {}).get("Source", ""),
        data.get("project", {}).get("urls", {}).get("source", ""),
        # Poetry
        data.get("tool", {}).get("poetry", {}).get("repository", ""),
        data.get("tool", {}).get("poetry", {}).get("homepage", ""),
    ]
    for c in candidates:
        if c and "github.com" in c:
            url = c.rstrip("/")
            break

    return name, url


# ---------------------------------------------------------------------------
# GitHub URL helpers
# ---------------------------------------------------------------------------

def github_file_url(rel_path: Path, lineno: int | None = None) -> str:
    """
    Return a GitHub URL pointing to a specific file (blob view).
    rel_path is relative to the repo root (= src_dir).
    """
    repo = _CONFIG["github_repo"]
    branch = _CONFIG["branch"]
    if not repo:
        return ""
    url = f"{repo}/blob/{branch}/{rel_path.as_posix()}"
    if lineno is not None:
        url += f"#L{lineno}"
    return url


def github_file_badge(rel_path: Path) -> str:
    """Return a callout-note block with the source file link, or '' if no repo."""
    url = github_file_url(rel_path)
    if not url:
        return ""
    return (
        '::: {.callout-note appearance="minimal"}\n'
        f"**Source:** [`{rel_path.as_posix()}`]({url})\n"
        ":::\n"
    )


# ---------------------------------------------------------------------------
# AST helpers
# ---------------------------------------------------------------------------

def parse_file(path: Path):
    try:
        return ast.parse(path.read_text(encoding="utf-8", errors="replace"))
    except SyntaxError as exc:
        print(f"  [WARN] syntax error in {path}: {exc}", file=sys.stderr)
        return None


def module_docstring(tree) -> str:
    return ast.get_docstring(tree) or ""


def _sig(node) -> str:
    if isinstance(node, ast.ClassDef):
        bases = ", ".join(ast.unparse(b) for b in node.bases)
        return f"class {node.name}({bases})" if bases else f"class {node.name}"
    prefix = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
    try:
        args_str = ast.unparse(node.args)
    except Exception:
        args_str = "..."
    ret = ""
    if getattr(node, "returns", None):
        try:
            ret = f" -> {ast.unparse(node.returns)}"
        except Exception:
            pass
    return f"{prefix} {node.name}({args_str}){ret}"


def iter_definitions(tree):
    """
    Yield (kind, name, lineno, doc, sig) for top-level and one-level-deep defs.
    kind in {"class", "method", "function", "async_function"}
    """
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            yield ("class", node.name, node.lineno,
                   ast.get_docstring(node) or "", _sig(node))
            for child in node.body:
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    yield ("method", f"{node.name}.{child.name}", child.lineno,
                           ast.get_docstring(child) or "", _sig(child))
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            kind = ("async_function" if isinstance(node, ast.AsyncFunctionDef)
                    else "function")
            yield (kind, node.name, node.lineno,
                   ast.get_docstring(node) or "", _sig(node))


# ---------------------------------------------------------------------------
# QMD content builder
# ---------------------------------------------------------------------------

def build_qmd_content(rel_py: Path, module_doc: str,
                       definitions: list, is_init: bool = False) -> str:
    lines: list[str] = []

    # YAML front matter
    if is_init:
        title = (rel_py.parent.name if rel_py.parent != Path(".")
                 else _CONFIG["project_name"])
    else:
        title = rel_py.stem
    lines += ["---", f'title: "{title}"', "---", ""]

    # Module / section intro
    if module_doc:
        lines += [module_doc.strip(), ""]

    if is_init:
        return "\n".join(lines)

    # Source link badge
    badge = github_file_badge(rel_py)
    if badge:
        lines += [badge, ""]

    if not definitions:
        lines.append("*No public definitions found in this module.*")
        return "\n".join(lines)

    lines += ["## API Reference", ""]

    for kind, name, lineno, doc, sig in definitions:
        anchor = re.sub(r"[^a-zA-Z0-9_-]", "-", name.lower())
        src_line_url = github_file_url(rel_py, lineno)

        heading = "###" if kind == "class" else "####"
        lines += [f"{heading} `{name}` {{#{anchor}}}", ""]
        lines += ["```python", sig, "```", ""]

        if doc:
            lines += [doc.strip(), ""]

        if src_line_url:
            lines += [
                f"[View source ↗]({src_line_url})"
                "{.source-link style='font-size:0.8em;'}",
                "",
            ]

        lines += ["---", ""]

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Quarto book _quarto.yml builder
# ---------------------------------------------------------------------------
#
# Quarto books only allow `part:` at the TOP level of the chapters list —
# nesting part: inside chapters: is invalid.  We therefore flatten the whole
# tree into a two-level structure:
#
#   chapters:
#     - index.qmd                  # top-level files
#     - part: section/index.qmd    # every folder becomes a top-level part …
#       chapters:                  # … with ALL its descendant files listed flat
#         - section/file1.qmd
#         - section/sub/file2.qmd  # sub-folder files included here, not nested
#
# The sub-folder index pages (sub/index.qmd) are prepended to their group
# so they appear as the first chapter of that part.

def _collect_flat(node: dict, prefix: str = "") -> tuple[list[str], list[tuple[str, list[str]]]]:
    """
    Walk the nav tree and return:
      top_files  – qmd paths that live at the current level (no sub-section)
      parts      – list of (part_index_or_label, [flat list of all chapter qmds])

    This is called once on the root nav_tree.
    """
    top_files: list[str] = []
    parts: list[tuple[str, list[str]]] = []

    for key in sorted(node.keys()):
        if key == "__index__":
            continue
        val = node[key]
        if isinstance(val, str):
            top_files.append(val)
        elif isinstance(val, dict):
            index_qmd = val.get("__index__", "")
            # Recursively gather every leaf qmd under this subtree, flat
            leaves = _flatten_leaves(val)
            parts.append((index_qmd or key, leaves))

    return top_files, parts


def _flatten_leaves(node: dict) -> list[str]:
    """Return every leaf qmd in the subtree, depth-first, index pages first."""
    result: list[str] = []
    # Sub-section index pages first (they act as mini-introductions)
    for key in sorted(node.keys()):
        if key == "__index__":
            continue
        val = node[key]
        if isinstance(val, dict):
            idx = val.get("__index__")
            if idx:
                result.append(idx)
    # Then leaf files and recurse into sub-dicts
    for key in sorted(node.keys()):
        if key == "__index__":
            continue
        val = node[key]
        if isinstance(val, str):
            result.append(val)
        elif isinstance(val, dict):
            result.extend(_flatten_leaves(val))
    return result


def _chapter_entries(nav_tree: dict) -> list[str]:
    """Return the YAML lines for the book chapters block (flat, Quarto-valid)."""
    top_files, parts = _collect_flat(nav_tree)
    lines: list[str] = []

    for f in top_files:
        lines.append(f"    - {f}")

    for part_label, chapters in parts:
        # part_label is either a qmd path (index page) or a plain string name
        lines.append(f"    - part: {part_label}")
        if chapters:
            lines.append("      chapters:")
            for ch in chapters:
                lines.append(f"        - {ch}")

    return lines


def build_quarto_yml(nav_tree: dict) -> str:
    chapter_lines = _chapter_entries(nav_tree)
    chapters_str = "\n".join(chapter_lines)
    name = _CONFIG["project_name"]
    repo = _CONFIG["github_repo"]

    repo_block = ""
    if repo:
        repo_block = (
            f"  repo-url: {repo}\n"
            f"  repo-actions: [source, issue]\n"
        )

    return f"""\
project:
  type: book

book:
  title: "{name}"
  sidebar:
    collapse-level: 1
  search: true
{repo_block}\
  chapters:
    - index.qmd
{chapters_str}

format:
  html:
    theme: cosmo
    toc: true
    toc-depth: 3
    code-fold: false
    highlight-style: github


execute:
  freeze: auto  # re-render only when source changes

"""


# ---------------------------------------------------------------------------
# Navigation tree helpers
# ---------------------------------------------------------------------------

def insert_nav(tree: dict, parts: list, value):
    if len(parts) == 1:
        tree[parts[0]] = value
        return
    section = parts[0]
    if section not in tree or not isinstance(tree[section], dict):
        tree[section] = {}
    insert_nav(tree[section], parts[1:], value)


# ---------------------------------------------------------------------------
# File collection
# ---------------------------------------------------------------------------

def collect_py_files(src_dir: Path) -> list:
    result = []
    for root, dirs, files in os.walk(src_dir):
        dirs[:] = sorted(
            d for d in dirs
            if d not in IGNORE_NAMES and not d.startswith(".")
        )
        root_path = Path(root)
        for f in sorted(files):
            if f.endswith(".py"):
                rel = (root_path / f).relative_to(src_dir)
                result.append(rel)
    return result


def rel_py_to_qmd(rel_py: Path) -> Path:
    if rel_py.name == "__init__.py":
        parent = rel_py.parent
        return Path("index.qmd") if parent == Path(".") else parent / "index.qmd"
    return rel_py.with_suffix(".qmd")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def generate(src_dir: Path, out_dir: Path):
    src_dir = src_dir.resolve()
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Read pyproject.toml
    proj_name, github_url = read_pyproject(src_dir)
    if proj_name:
        _CONFIG["project_name"] = proj_name
        print(f"  Project name from pyproject.toml: {proj_name}")
    if github_url and not _CONFIG["github_repo"]:
        # Only use toml value if not overridden by --github CLI flag
        _CONFIG["github_repo"] = github_url
        print(f"  GitHub URL from pyproject.toml:   {github_url}")
    if not _CONFIG["github_repo"]:
        print("  [WARN] No GitHub URL found. Source links will be omitted.")
        print("         Add [project.urls] repository = '...' to pyproject.toml,")
        print("         or pass --github https://github.com/user/repo")

    py_files = collect_py_files(src_dir)
    print(f"\nFound {len(py_files)} Python files under {src_dir}\n")

    nav_tree: dict = {}

    for rel_py in py_files:
        full_py = src_dir / rel_py
        is_init = rel_py.name == "__init__.py"

        tree = parse_file(full_py)
        module_doc = module_docstring(tree) if tree else ""
        definitions = list(iter_definitions(tree)) if (tree and not is_init) else []

        rel_qmd = rel_py_to_qmd(rel_py)
        out_qmd = out_dir / rel_qmd
        out_qmd.parent.mkdir(parents=True, exist_ok=True)
        out_qmd.write_text(
            build_qmd_content(rel_py, module_doc, definitions, is_init),
            encoding="utf-8",
        )
        print(f"  ✓  {rel_qmd}")

        # Register in nav tree
        if is_init:
            if rel_py.parent == Path("."):
                pass  # top-level __init__ → index.qmd (first chapter)
            else:
                parts = list(rel_py.parent.parts) + ["__index__"]
                insert_nav(nav_tree, parts, rel_qmd.as_posix())
        else:
            parts = list(rel_py.parent.parts) + [rel_py.stem]
            insert_nav(nav_tree, parts, rel_qmd.as_posix())

    # _quarto.yml
    (out_dir / "_quarto.yml").write_text(build_quarto_yml(nav_tree), encoding="utf-8")
    print(f"\n  ✓  _quarto.yml")

    # Placeholder index.qmd if no top-level __init__.py
    index_path = out_dir / "index.qmd"
    if not index_path.exists():
        index_path.write_text(
            f'---\ntitle: "{_CONFIG["project_name"]}"\n---\n\n'
            f"Welcome to the **{_CONFIG['project_name']}** documentation.\n",
            encoding="utf-8",
        )
        print("  ✓  index.qmd  (placeholder — add a top-level __init__.py for custom text)")

    print(f"\nDone! Quarto book written to: {out_dir}")
    print(f"Render with:  cd '{out_dir}' && quarto render")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate a Quarto book documentation project from a Python library."
    )
    parser.add_argument("--src", required=True, type=Path,
                        help="Root directory of the Python library")
    parser.add_argument("--out", required=True, type=Path,
                        help="Output directory for the Quarto book (created if absent)")
    parser.add_argument(
        "--github",
        default="",
        help=(
            "GitHub repository URL, e.g. https://github.com/user/repo  "
            "(overrides pyproject.toml; omit to auto-detect)"
        ),
    )
    parser.add_argument(
        "--branch",
        default="main",
        help="Git branch used in source file links (default: main)",
    )
    args = parser.parse_args()

    if args.github:
        _CONFIG["github_repo"] = args.github.rstrip("/")
    _CONFIG["branch"] = args.branch

    generate(args.src, args.out)


if __name__ == "__main__":
    main()
