#!/usr/bin/env python3
"""
generate_docs.py
================
Pipeline to auto-generate a Quarto documentation project from a Python library.

Usage
-----
    python generate_docs.py --src /path/to/library --out /path/to/docs_output

The script:
1. Walks the library tree and parses every .py file with ``ast``
2. Extracts module-level docstrings, class/function signatures and docstrings
3. Writes one .qmd file per .py file, mirroring the folder structure
4. Builds ``_quarto.yml`` with a nested sidebar that reflects the folder hierarchy
5. ``__init__.py`` files supply section-level introductory text (index.qmd)
"""

import ast
import argparse
import os
import re
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Runtime configuration (overridden by CLI --github flag)
# ---------------------------------------------------------------------------
_CONFIG = {
    "github_base": "https://github.com/YOUR_ORG/lattice_data_tools/blob/main",
}

# ---------------------------------------------------------------------------
# Layout constants
# ---------------------------------------------------------------------------
IGNORE_NAMES = {
    "__pycache__",
    ".git",
    "doc",      # existing docs – ignored as requested
    "legacy",
    "old",
}


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
# QMD content builders
# ---------------------------------------------------------------------------

def github_link(rel_path: Path) -> str:
    return f"{_CONFIG['github_base']}/{rel_path.as_posix()}"


def build_qmd_content(rel_py: Path, module_doc: str,
                       definitions: list, is_init: bool = False) -> str:
    lines = []

    # YAML front matter
    if is_init:
        title = (rel_py.parent.name if rel_py.parent != Path(".")
                 else "lattice_data_tools")
    else:
        title = rel_py.stem
    lines += ["---", f'title: "{title}"', "---", ""]

    # Module / section intro
    if module_doc:
        lines += [module_doc.strip(), ""]

    if is_init:
        return "\n".join(lines)

    # Source link callout
    src_url = github_link(rel_py)
    lines += [
        '::: {.callout-note appearance="minimal"}',
        f"**Source:** [`{rel_py.as_posix()}`]({src_url})",
        ":::", "",
    ]

    if not definitions:
        lines.append("*No public definitions found in this module.*")
        return "\n".join(lines)

    lines += ["## API Reference", ""]

    for kind, name, lineno, doc, sig in definitions:
        anchor = re.sub(r"[^a-zA-Z0-9_-]", "-", name.lower())
        src_line_url = f"{src_url}#L{lineno}"

        # Class → h3, everything else → h4
        heading = "###" if kind == "class" else "####"
        lines += [f"{heading} `{name}` {{#{anchor}}}", ""]
        lines += ["```python", sig, "```", ""]

        if doc:
            lines += [doc.strip(), ""]

        lines += [
            f"[View source ↗]({src_line_url})"
            "{.source-link style='font-size:0.8em;'}",
            "", "---", "",
        ]

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Quarto _quarto.yml builder
# ---------------------------------------------------------------------------

def _sidebar_entries(node: dict, indent: int = 8) -> list:
    pad = " " * indent
    entries = []
    for key in sorted(node.keys()):
        val = node[key]
        if key == "__index__":
            continue
        if isinstance(val, str):
            entries.append(f'{pad}- "{val}"')
        elif isinstance(val, dict):
            index_qmd = val.get("__index__")
            if index_qmd:
                sec_head = (f'{pad}- section: "{key}"\n'
                            f'{pad}  href: "{index_qmd}"')
            else:
                sec_head = f'{pad}- section: "{key}"'
            children = _sidebar_entries(
                {k: v for k, v in val.items() if k != "__index__"},
                indent + 2,
            )
            entries.append(sec_head)
            if children:
                entries.append(f"{pad}  contents:")
                entries += children
    return entries


def build_quarto_yml(nav_tree: dict) -> str:
    sidebar_lines = _sidebar_entries(nav_tree, indent=8)
    sidebar_str = "\n".join(sidebar_lines)
    return f"""\
project:
  type: website
  output-dir: _site

website:
  title: "lattice_data_tools"
  navbar:
    left:
      - href: index.qmd
        text: Home
  sidebar:
    style: "docked"
    search: true
    contents:
        - "index.qmd"
{sidebar_str}

format:
  html:
    theme: cosmo
    toc: true
    toc-depth: 3
    code-fold: false
    highlight-style: github
"""


# ---------------------------------------------------------------------------
# Navigation tree
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

    py_files = collect_py_files(src_dir)
    print(f"Found {len(py_files)} Python files under {src_dir}\n")

    nav_tree: dict = {}
    has_index = False

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
                has_index = True  # top-level __init__ → index.qmd (Home)
            else:
                parts = list(rel_py.parent.parts) + ["__index__"]
                insert_nav(nav_tree, parts, rel_qmd.as_posix())
        else:
            parts = list(rel_py.parent.parts) + [rel_py.stem]
            insert_nav(nav_tree, parts, rel_qmd.as_posix())

    # _quarto.yml
    (out_dir / "_quarto.yml").write_text(build_quarto_yml(nav_tree), encoding="utf-8")
    print(f"\n  ✓  _quarto.yml")

    # Placeholder index if no top-level __init__.py
    index_path = out_dir / "index.qmd"
    if not index_path.exists():
        index_path.write_text(
            '---\ntitle: "lattice_data_tools"\n---\n\n'
            "Welcome to the **lattice_data_tools** documentation.\n",
            encoding="utf-8",
        )
        print("  ✓  index.qmd  (placeholder — add a top-level __init__.py for custom text)")

    print(f"\nDone! Quarto project written to: {out_dir}")
    print(f"Render with:  cd '{out_dir}' && quarto render")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate a Quarto documentation project from a Python library."
    )
    parser.add_argument("--src", required=True, type=Path,
                        help="Root directory of the Python library")
    parser.add_argument("--out", required=True, type=Path,
                        help="Output directory for the Quarto project (created if absent)")
    parser.add_argument(
        "--github",
        default="https://github.com/YOUR_ORG/lattice_data_tools/blob/main",
        help="GitHub base URL for source links",
    )
    args = parser.parse_args()
    _CONFIG["github_base"] = args.github.rstrip("/")
    generate(args.src, args.out)


if __name__ == "__main__":
    main()
