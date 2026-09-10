"""Exercise Batch teardown against local AWS command stubs."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path
from urllib.parse import urlparse

import pytest

pytestmark = pytest.mark.unit
_ROOT = Path(__file__).resolve().parents[2]


def _stub_aws(tmp_path, source):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    aws = bin_dir / "aws"
    aws.write_text(f"#!{sys.executable}\n" + source)
    aws.chmod(0o755)
    sleep = bin_dir / "sleep"
    sleep.write_text("#!/bin/sh\nexit 0\n")
    sleep.chmod(0o755)
    return str(bin_dir) + os.pathsep + os.environ["PATH"]


def test_bootstrap_requests_certificate_for_deployed_application_host(tmp_path):
    source = (_ROOT / "infra/aws/bootstrap.sh").read_text()
    declarations = "\n".join(
        line for line in source.splitlines() if re.match(r"^(DOMAIN|DOMAIN_WWW|DNS_ZONE)=", line)
    )
    request = re.search(r"CERT_ARN=\$\((aws acm request-certificate.*?)\)\n", source, re.S)[1]
    path = _stub_aws(
        tmp_path,
        "import json, os, pathlib, sys\n"
        "pathlib.Path(os.environ['TRACE']).write_text(json.dumps(sys.argv[1:]))\n"
        "print('arn:certificate')\n",
    )
    trace = tmp_path / "certificate-call.json"
    subprocess.run(
        ["bash", "-euc", declarations + "\n" + request],
        env={**os.environ, "PATH": path, "TRACE": str(trace), "REGION": "us-east-1"},
        check=True,
        capture_output=True,
        text=True,
    )
    args = json.loads(trace.read_text())
    domains = [args[args.index("--domain-name") + 1]]
    if "--subject-alternative-names" in args:
        domains.append(args[args.index("--subject-alternative-names") + 1])
    workflow = (_ROOT / ".github/workflows/deploy.yml").read_text()
    deployed_host = urlparse(re.search(r"SERVICE_URL:\s*(https://\S+)", workflow)[1]).hostname
    assert deployed_host in domains


def test_bootstrap_dns_instructions_preserve_the_portfolio_apex(tmp_path):
    source = (_ROOT / "infra/aws/bootstrap.sh").read_text()
    declarations = "\n".join(
        line for line in source.splitlines() if re.match(r"^(DOMAIN|DOMAIN_WWW|DNS_ZONE)=", line)
    )
    summary = source.split("# Summary\n", 1)[1]
    result = subprocess.run(
        ["bash", "-euc", declarations + "\n" + summary],
        env={**os.environ, "OUT_FILE": "unused", "ALB_DNS": "app.elb.amazonaws.com"},
        check=True,
        capture_output=True,
        text=True,
    )
    assert re.search(r"CNAME\s+fantasy\s+-> app.elb.amazonaws.com", result.stdout)
    assert "ALIAS  @" not in result.stdout
    assert "CNAME  www" not in result.stdout


def test_bootstrap_reconciles_existing_https_certificate(tmp_path):
    source = (_ROOT / "infra/aws/bootstrap.sh").read_text()
    https = "HTTPS_ARN=" + source.split("HTTPS_ARN=", 1)[1].split("\nHTTP_ARN=", 1)[0]
    path = _stub_aws(
        tmp_path,
        "import json, os, pathlib, sys\n"
        "args = sys.argv[1:]\n"
        "with pathlib.Path(os.environ['TRACE']).open('a') as f: f.write(json.dumps(args)+'\\n')\n"
        "print('arn:existing-listener' if args[1] == 'describe-listeners' else '{}')\n",
    )
    trace = tmp_path / "listener-calls.jsonl"
    subprocess.run(
        ["bash", "-euc", "log() { :; }; out() { :; };\n" + https],
        env={
            **os.environ,
            "PATH": path,
            "TRACE": str(trace),
            "REGION": "us-east-1",
            "ALB_ARN": "arn:alb",
            "DOMAIN": "fantasy.alexfree.me",
            "CERT_ARN": "arn:app-certificate",
            "TG_ARN": "arn:target",
        },
        check=True,
        capture_output=True,
        text=True,
    )
    calls = [json.loads(line) for line in trace.read_text().splitlines()]
    assert any(
        call[:2] == ["elbv2", "modify-listener"] and "CertificateArn=arn:app-certificate" in call
        for call in calls
    )


def test_batch_teardown_removes_cpu_resources_and_preserves_serving_role(tmp_path):
    path = _stub_aws(
        tmp_path,
        """import json, os, pathlib, sys
args = sys.argv[1:]
with pathlib.Path(os.environ['TRACE']).open('a') as f:
    f.write(json.dumps(args) + '\\n')
state_path = pathlib.Path(os.environ['STATE'])
state = json.loads(state_path.read_text())
def value(flag):
    return args[args.index(flag) + 1] if flag in args else ''
operation = args[1]
if operation.startswith('delete-'):
    for flag in ('--job-queue', '--compute-environment'):
        if flag in args:
            state[value(flag)] = True
    state_path.write_text(json.dumps(state))
if operation == 'describe-job-definitions':
    print('1')
elif operation in ('describe-job-queues', 'describe-compute-environments'):
    name = value('--job-queues') or value('--compute-environments')
    print('None' if state.get(name) else 'VALID' if value('--query').endswith('.status') else name)
elif operation == 'describe-vpcs':
    print('vpc-test')
elif operation == 'describe-security-groups':
    print('sg-test')
elif operation == 'list-attached-role-policies':
    print('arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy')
elif operation == 'list-role-policies':
    print('inline-workload')
""",
    )
    trace, state = tmp_path / "trace", tmp_path / "state"
    state.write_text("{}")
    result = subprocess.run(
        ["bash", str(_ROOT / "infra/batch/teardown.sh")],
        env={**os.environ, "PATH": path, "TRACE": str(trace), "STATE": str(state)},
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    calls = [json.loads(line) for line in trace.read_text().splitlines()]
    expected = [
        ("deregister-job-definition", "--job-definition", "ff-training-cpu-job:1"),
        ("delete-job-queue", "--job-queue", "ff-cpu-training-queue"),
        ("delete-compute-environment", "--compute-environment", "ff-cpu-spot"),
    ]
    for operation, flag, resource in expected:
        assert any(
            call[1] == operation and flag in call and call[call.index(flag) + 1] == resource
            for call in calls
        )
    serving_role = json.loads((_ROOT / "infra/aws/task-definition.json").read_text())[
        "executionRoleArn"
    ].split("/")[-1]
    assert not any(serving_role in call for call in calls if call[0] == "iam")
