"""Render the checked-in ff-train bootstrap wrapper as an atomic SSM update."""

from __future__ import annotations

import argparse
import json
import re
import shlex
from pathlib import Path

BOOTSTRAP = Path(__file__).resolve().parents[2] / "infra/ec2/user-data.sh"
START = "cat > /usr/local/bin/ff-train <<EOF\n"
END = "\nEOF\nchmod 755 /usr/local/bin/ff-train"


def wrapper_template(bootstrap: Path = BOOTSTRAP) -> str:
    text = bootstrap.read_text()
    if text.count(START) != 1 or text.count(END) != 1:
        raise ValueError("Cannot identify the unique ff-train bootstrap template")
    return text.split(START, 1)[1].split(END, 1)[0]


def render_update(
    region: str,
    bucket: str,
    *,
    target="/usr/local/bin/ff-train",
    cache_dir="/opt/ff/cache",
    cache_owner="ubuntu:ubuntu",
) -> str:
    if not re.fullmatch(r"[a-z0-9-]+", region) or not re.fullmatch(
        r"[a-z0-9][a-z0-9.-]{1,61}[a-z0-9]", bucket
    ):
        raise ValueError("Invalid region or bucket for ff-train wrapper")
    quote = shlex.quote
    lines = [
        # AWS-RunShellScript invokes /bin/sh (dash on Ubuntu). The generated
        # ff-train wrapper itself uses Bash, but its installer must be POSIX.
        "set -eu",
        f"REGION={quote(region)}",
        f"BUCKET={quote(bucket)}",
        f"_ec2_wrapper_target={quote(str(target))}",
        f"_ec2_cache_dir={quote(str(cache_dir))}",
        'mkdir -p "$(dirname "$_ec2_wrapper_target")" "$_ec2_cache_dir"',
    ]
    if cache_owner is not None:
        lines.append(f'chown {quote(cache_owner)} "$_ec2_cache_dir"')
    lines.extend(
        [
            '_ec2_wrapper_tmp=$(mktemp "$(dirname "$_ec2_wrapper_target")/.ff-train.XXXXXX")',
            "trap 'rm -f \"$_ec2_wrapper_tmp\"' EXIT",
            'cat > "$_ec2_wrapper_tmp" <<EOF',
            wrapper_template(),
            "EOF",
            'bash -n "$_ec2_wrapper_tmp"',
            'chmod 755 "$_ec2_wrapper_tmp"',
            'if [ -f "$_ec2_wrapper_target" ] && cmp -s "$_ec2_wrapper_tmp" "$_ec2_wrapper_target"; then',
            '  echo "ff-train wrapper already current"',
            "else",
            '  mv -f "$_ec2_wrapper_tmp" "$_ec2_wrapper_target"',
            '  echo "ff-train wrapper updated atomically"',
            "fi",
        ]
    )
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--region", required=True)
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.write_text(json.dumps({"commands": [render_update(args.region, args.bucket)]}))


if __name__ == "__main__":
    main()
