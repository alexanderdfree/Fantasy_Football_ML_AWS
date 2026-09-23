"""Verify the narrowly scoped WR/TE observer-identity source bridge."""

from __future__ import annotations

import ast
import hashlib
import io
import json
import subprocess
import tarfile
from pathlib import Path

ALLOWED_RUNTIME = {
    "src/tuning/audit_development.py",
    "src/tuning/audit_stint_candidate.py",
    "src/tuning/ab_audit_stint_development.py",
}
EXTRA_CORE = {
    "requirements.txt",
    "requirements-gpu.txt",
    "src/batch/requirements.txt",
    "src/batch/Dockerfile.train",
}
IDENTITY_FUNCTION = '''def configured_position(config):
    """Use the native loader identity; WR and TE deliberately share targets."""
    module = getattr(config.get("filter_fn"), "__module__", "")
    expected = {f"src.{position.lower()}.data": position for position in ("QB", "RB", "WR", "TE", "K", "DST")}
    if module not in expected:
        raise ValueError(f"Unknown production position filter identity: {module!r}")
    return expected[module]
'''


def verify_routing_only(parent, changed):
    """Reverse exactly the identity fix, then require AST equality with its parent."""
    before, after = ast.parse(parent), ast.parse(changed)
    functions = [
        n for n in after.body if isinstance(n, ast.FunctionDef) and n.name == "configured_position"
    ]
    if len(functions) != 1 or ast.dump(functions[0]) != ast.dump(
        ast.parse(IDENTITY_FUNCTION).body[0]
    ):
        raise ValueError("Source bridge contains an unrecognized position resolver")
    after.body.remove(functions[0])
    configure = next(
        n for n in after.body if isinstance(n, ast.FunctionDef) and n.name == "configure"
    )
    expected = ast.dump(ast.parse("position = configured_position(config)").body[0])
    statements = [n for n in configure.body if ast.dump(n) == expected]
    if len(statements) != 1:
        raise ValueError("Source bridge does not contain the declared identity replacement")
    offset = configure.body.index(statements[0])
    configure.body[offset] = ast.parse("position = infer_position(config['targets'])").body[0]
    configure.body.insert(
        0, ast.parse("from src.shared.aggregate_targets import infer_position").body[0]
    )
    if ast.dump(before) != ast.dump(after):
        raise ValueError("Source bridge changes additional experiment behavior")


def verify_bridge(source, declaration, *, repo_root=None):
    """Read immutable Git objects only; never fetch, checkout or modify a branch."""
    root = Path(repo_root or Path(__file__).resolve().parents[2])

    def git(*args):
        try:
            return subprocess.check_output(["git", "-C", str(root), *args], stderr=subprocess.PIPE)
        except subprocess.CalledProcessError as error:
            raise ValueError(
                f"Unavailable source-bridge Git objects: {error.stderr.decode().strip()}"
            ) from error

    path = declaration["path"]
    if path != "todo/audit-source-bridge-20260923.json":
        raise ValueError("Unrecognized source-bridge record path")
    raw = git("show", f"{source}:{path}")
    digest = hashlib.sha256(raw).hexdigest()
    if digest != declaration["sha256"]:
        raise ValueError("Source-bridge record checksum mismatch")
    proof = json.loads(raw)
    if proof["schema"] != "audit-observer-source-bridge/v1":
        raise ValueError("Unknown source-bridge schema")
    base, parent = proof["baseline_source_sha"], proof["source_parent_sha"]
    names = git("ls-tree", "-r", "--name-only", base, "--", "src").decode().splitlines()
    core = {p for p in names if p.endswith(".py") and p not in ALLOWED_RUNTIME} | EXTRA_CORE
    if set(proof["core_files_sha256"]) != core or proof["verified_core_files"] != len(core):
        raise ValueError("Source bridge omits core training/data/model files")
    if git("diff", base, source, "--", *sorted(core)):
        raise ValueError("Source bridge changes core training/data/model code")
    with tarfile.open(fileobj=io.BytesIO(git("archive", base, *sorted(core)))) as archive:
        hashes = {
            member.name: hashlib.sha256(archive.extractfile(member).read()).hexdigest()
            for member in archive.getmembers()
            if member.isfile()
        }
    fingerprint = hashlib.sha256(
        json.dumps(hashes, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if hashes != proof["core_files_sha256"] or fingerprint != proof["core_fingerprint"]:
        raise ValueError("Source-bridge core fingerprints do not reproduce")
    if proof["count_candidate_sha256"] != hashes["src/tuning/audit_count_candidate.py"]:
        raise ValueError("Source bridge changed the count candidate")
    changed = set(git("diff", "--name-only", base, source, "--", "src").decode().splitlines())
    if not changed <= ALLOWED_RUNTIME or changed != set(proof["changed_runtime_files"]):
        raise ValueError("Unexpected source-bridge runtime changes")
    for filename, expected in proof["changed_runtime_files"].items():
        if hashlib.sha256(git("show", f"{source}:{filename}")).hexdigest() != expected:
            raise ValueError("Source-bridge runtime hash mismatch")
    if git("diff", "--name-only", parent, source, "--", "src").decode().splitlines() != [
        "src/tuning/audit_development.py"
    ]:
        raise ValueError("Observer correction changes other runtime files")
    verify_routing_only(
        git("show", f"{parent}:src/tuning/audit_development.py").decode(),
        git("show", f"{source}:src/tuning/audit_development.py").decode(),
    )
    return {
        "source_sha": source,
        "baseline_source_sha": base,
        "source_parent_sha": parent,
        "bridge_sha256": digest,
        "core_fingerprint": fingerprint,
        "core_files": len(core),
        "observer_only_verified": True,
    }
