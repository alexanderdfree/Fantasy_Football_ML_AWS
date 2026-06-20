#!/usr/bin/env python3
"""Generate (and backfill) the Claude auto-memory ``MEMORY.md`` index from the topic files.

``MEMORY.md`` is the index Claude auto-loads each session. It is a SHARED MUTABLE file that
every session rewrites wholesale, so concurrent / cross-platform sessions race on it and drop
each other's index lines: the topic file survives (the per-file S3 sync is additive) but its
index line is lost -> an "orphan" the recall layer never surfaces. This module makes
``MEMORY.md`` a *generated projection* of the topic files instead. Each topic file carries its
curated index line in a frontmatter ``index_line`` block scalar; the index is regenerated from
those at SessionStart. Combined with excluding ``MEMORY.md`` from the S3 sync (see
``scripts/agent-memory-sync.sh``), the index is no longer shared mutable state -> not racy.

Stdlib only, by design: the SessionStart hook runs before any venv is active, so we cannot
depend on PyYAML. ``index_line`` is stored as a ``|-`` block scalar, which a tiny line-based
parser reads/writes robustly -- real entries contain brackets, embedded double-quotes, em-dashes
and colons that naive YAML string handling would corrupt.

Usage:
  python scripts/memory_index.py generate <memory_dir>   # print regenerated index to stdout
  python scripts/memory_index.py backfill <memory_dir>   # write index_line into each topic file
"""

from __future__ import annotations

import os
import re
import sys

# ~24.4 KiB: above this the auto-loader truncates MEMORY.md (dropping the NEWEST entries).
CAP_BYTES = 24985
INDEX = "MEMORY.md"
_LINK_RE = re.compile(r"\]\(([^)]+\.md)\)")  # slug from "...](slug.md)..."


def split_frontmatter(text):
    """Return (frontmatter_lines, body_lines); frontmatter excludes the ``---`` fences.

    A file with no leading ``---`` fence (or an unterminated one) has no frontmatter.
    """
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return [], lines
    for i in range(1, len(lines)):
        if lines[i].strip() == "---":
            return lines[1:i], lines[i + 1 :]
    return [], lines


def read_key(fm_lines, key):
    """Read a top-level frontmatter ``key``: handles ``|``/``|-`` block scalars and inline values.

    Returns the value (str) or None if the key is absent.
    """
    pat = re.compile(rf"^{re.escape(key)}:\s*(.*)$")
    for idx, line in enumerate(fm_lines):
        m = pat.match(line)
        if not m:
            continue
        rest = m.group(1).strip()
        if rest.startswith("|"):  # block scalar -> collect the indented continuation
            cont = []
            for nxt in fm_lines[idx + 1 :]:
                if nxt.strip() == "" or nxt[:1] in (" ", "\t"):
                    cont.append(nxt)
                else:
                    break
            indents = [len(c) - len(c.lstrip()) for c in cont if c.strip()]
            n = min(indents) if indents else 0
            return "\n".join(c[n:] for c in cont).strip()
        if len(rest) >= 2 and rest[0] == rest[-1] and rest[0] in ("'", '"'):
            return rest[1:-1]
        return rest
    return None


def _line_parts(path):
    """Return (prefix, hook, fell_back, warning); the rendered index line is ``prefix + hook``.

    For an `index_line` file the whole curated line is the hook (prefix ``"- "``). For a fallback
    the prefix is the ``"- [title](slug) — "`` scaffold and the hook is the *untruncated*
    description (or first body line) — generate_index trims it to a cap-aware budget.
    """
    with open(path, encoding="utf-8") as fh:
        fm, body = split_frontmatter(fh.read())
    val = read_key(fm, "index_line")
    if val:
        return "- ", " ".join(val.split()), False, None
    slug = os.path.basename(path)
    title = read_key(fm, "name") or slug[:-3]
    desc = read_key(fm, "description")
    if desc:
        return (
            f"- [{title}]({slug}) — ",
            " ".join(desc.split()),
            True,
            f"{slug}: no index_line, used description",
        )
    first = next((ln.strip() for ln in body if ln.strip()), slug[:-3])
    return (
        f"- [{title}]({slug}) — ",
        " ".join(first.lstrip("# ").split()),
        True,
        f"{slug}: no index_line/description, used body",
    )


def _fit(prefix, hook, budget):
    """Render ``prefix + hook`` truncated so its UTF-8 size is <= ``budget`` bytes (… if cut)."""
    if len((prefix + hook).encode("utf-8")) <= budget:
        return prefix + hook
    avail = budget - len(prefix.encode("utf-8")) - len("…".encode())
    if avail <= 0:
        return f"{prefix}…"
    cut = hook.encode("utf-8")[:avail].decode("utf-8", "ignore").rstrip()
    return f"{prefix}{cut}…"


def generate_index(memdir):
    """Rebuild the index text from every topic file. Returns (text, warnings).

    Deterministic (slug-sorted) and idempotent. Curated `index_line` lines are emitted in full;
    fallback (description/body) lines are trimmed to a DYNAMIC per-line budget so the total can
    never exceed the auto-load cap. A bulk-fallback state (e.g. mid-migration, when a concurrent
    pull has stripped `index_line` from many files) thus degrades to a short-but-complete index
    instead of an over-cap one the loader would silently truncate. Still warns near/over the cap
    (the over case means too many *curated* lines — those are never trimmed; prune instead).
    """
    files = sorted(
        f
        for f in os.listdir(memdir)
        if f.endswith(".md") and f != INDEX and os.path.isfile(os.path.join(memdir, f))
    )
    parts, warnings = [], []
    for f in files:
        prefix, hook, fell_back, warn = _line_parts(os.path.join(memdir, f))
        parts.append((prefix, hook, fell_back))
        if warn:
            warnings.append(warn)

    def nbytes(prefix, hook):
        return len((prefix + hook + "\n").encode("utf-8"))

    target = int(CAP_BYTES * 0.93)  # leave margin below the hard cap (newlines + safety)
    fixed = sum(nbytes(p, h) for p, h, fb in parts if not fb)
    fallback = [(p, h) for p, h, fb in parts if fb]
    budget = ((target - fixed) // len(fallback)) if fallback else 0  # bytes per fallback line

    lines = [(_fit(p, h, max(1, budget)) if fb else p + h) for p, h, fb in parts]
    out = "\n".join(lines) + ("\n" if lines else "")

    size = len(out.encode("utf-8"))
    if size >= CAP_BYTES:
        warnings.append(
            f"index {size} B >= cap {CAP_BYTES} B -- too many curated entries; prune/consolidate"
        )
    elif size >= int(CAP_BYTES * 0.92):
        warnings.append(f"index {size} B is near the {CAP_BYTES} B cap ({CAP_BYTES - size} B left)")
    return out, warnings


def _strip_index_line(fm_lines):
    """Drop an existing ``index_line`` key + its block continuation (so backfill is idempotent)."""
    out, skipping = [], False
    for line in fm_lines:
        if re.match(r"^index_line:\s*", line):
            skipping = True
            continue
        if skipping:
            if line.strip() == "" or line[:1] in (" ", "\t"):
                continue
            skipping = False
        out.append(line)
    return out


def _guess_type(slug):
    if slug.startswith("feedback_"):
        return "feedback"
    if slug.startswith("project_"):
        return "project"
    return "reference"


def backfill(memdir):
    """Write each topic file's current ``MEMORY.md`` line into its frontmatter ``index_line``.

    Source of truth is the existing curated index. Returns (changed, missing) basenames;
    ``missing`` = files with no current index line (orphans) -> left untouched + reported.
    """
    with open(os.path.join(memdir, INDEX), encoding="utf-8") as fh:
        index_text = fh.read()
    slug_to_line = {}
    for raw in index_text.splitlines():
        line = raw.strip()
        if not line.startswith("- "):
            continue
        m = _LINK_RE.search(line)
        if m:
            slug_to_line[m.group(1)] = line[2:]  # text after "- "
    changed, missing = [], []
    for f in sorted(os.listdir(memdir)):
        if not f.endswith(".md") or f == INDEX:
            continue
        path = os.path.join(memdir, f)
        if not os.path.isfile(path):
            continue
        if f not in slug_to_line:
            missing.append(f)
            continue
        block = ["index_line: |-", f"  {slug_to_line[f]}"]
        with open(path, encoding="utf-8") as fh:
            text = fh.read()
        fm, body = split_frontmatter(text)
        if fm or text.lstrip().startswith("---"):
            new = ["---", *_strip_index_line(fm), *block, "---", *body]
        else:
            title_desc = slug_to_line[f].split(" — ", 1)
            desc = title_desc[1] if len(title_desc) > 1 else slug_to_line[f]
            new = [
                "---",
                f"name: {f[:-3]}",
                f"description: {desc}",
                "metadata:",
                f"  type: {_guess_type(f)}",
                *block,
                "---",
                "",
                *body,
            ]
        out = "\n".join(new).rstrip("\n") + "\n"
        tmp = f"{path}.tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            fh.write(out)
        os.replace(tmp, path)
        changed.append(f)
    return changed, missing


def main(argv):
    if len(argv) != 3 or argv[1] not in ("generate", "backfill"):
        sys.stderr.write("usage: memory_index.py {generate|backfill} <memory_dir>\n")
        return 2
    cmd, memdir = argv[1], argv[2]
    if not os.path.isdir(memdir):
        sys.stderr.write(f"memory_index: not a directory: {memdir}\n")
        return 1
    if cmd == "generate":
        text, warnings = generate_index(memdir)
        sys.stdout.write(text)
        for w in warnings:
            sys.stderr.write(f"[memory-index] WARN: {w}\n")
        return 0
    changed, missing = backfill(memdir)
    sys.stderr.write(f"[memory-index] backfill: wrote index_line into {len(changed)} file(s)\n")
    for m in missing:
        sys.stderr.write(f"[memory-index] WARN: {m} not in {INDEX}, no index_line written\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
