import json
import os
import re
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
pytestmark = pytest.mark.unit


def test_scheduled_triage_selects_unlabelled_or_requested_open_issues(tmp_path):
    workflow = (ROOT / ".github/workflows/gemini-scheduled-triage.yml").read_text()
    match = re.search(
        r"      - name: 'Find untriaged issues'.*?        run: \|-\n(.*?)(?=\n      - name:)",
        workflow,
        re.S,
    )
    assert match
    script = textwrap.dedent(match.group(1))
    issues = [
        {"number": 1, "state": "open", "repo": "fixture/repo", "labels": []},
        {
            "number": 2,
            "state": "open",
            "repo": "fixture/repo",
            "labels": ["bug", "status/needs-triage"],
        },
        {"number": 3, "state": "open", "repo": "fixture/repo", "labels": ["enhancement"]},
        {"number": 4, "state": "closed", "repo": "fixture/repo", "labels": []},
        {"number": 5, "state": "closed", "repo": "fixture/repo", "labels": ["status/needs-triage"]},
        {"number": 6, "state": "open", "repo": "another/repo", "labels": []},
        {"number": 7, "state": "open", "repo": "another/repo", "labels": ["status/needs-triage"]},
    ]
    fixture = tmp_path / "issues.json"
    fixture.write_text(json.dumps(issues))
    stub = tmp_path / "gh"
    stub.write_text(
        f"#!{sys.executable}\n"
        + textwrap.dedent("""
        import json, os, shlex, sys
        args = sys.argv[1:]
        assert args[:2] == ['issue', 'list']
        option = lambda name: args[args.index(name) + 1]
        query = shlex.split(option('--search'))
        terms = [term for term in query if term != 'OR']
        def matches(issue, term):
            if term == 'no:label':
                return not issue['labels']
            assert term.startswith('label:')
            return term.removeprefix('label:') in issue['labels']
        selected = []
        for issue in json.load(open(os.environ['TRIAGE_FIXTURE'])):
            if issue['state'] != option('--state') or issue['repo'] != option('--repo'):
                continue
            predicate = any if 'OR' in query else all
            if predicate(matches(issue, term) for term in terms):
                selected.append({'number': issue['number'], 'title': 'Fixture', 'body': 'Fixture'})
        print(json.dumps(selected[:int(option('--limit'))], separators=(',', ':')))
    """)
    )
    stub.chmod(0o755)
    output = tmp_path / "output"
    result = subprocess.run(
        ["bash", "-euo", "pipefail", "-c", script],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        env={
            **os.environ,
            "PATH": f"{tmp_path}{os.pathsep}{os.environ['PATH']}",
            "GITHUB_REPOSITORY": "fixture/repo",
            "GITHUB_OUTPUT": str(output),
            "TRIAGE_FIXTURE": str(fixture),
        },
    )
    assert result.returncode == 0, result.stderr
    values = dict(line.split("=", 1) for line in output.read_text().splitlines())
    assert [issue["number"] for issue in json.loads(values["issues_to_triage"])] == [1, 2]
    assert values["issue_numbers"] == "1,2"
