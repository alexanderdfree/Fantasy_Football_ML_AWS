"""Exercise the bake boundary without launching any AWS resources."""

import copy
import json
import os
import subprocess
from pathlib import Path
from unittest.mock import Mock

import pytest

pytestmark = pytest.mark.unit
ROOT = Path(__file__).resolve().parents[2]


def paired_evidence():
    from infra.batch.warm_ami import POSITIONS, TYPES

    state = {"bake": {"source_ami": "ami-stock", "candidate_ami": "ami-warm"}, "resources": []}
    for kind in TYPES:
        for pair in range(3):
            for arm in ["cold", "warm"]:
                timing = {
                    "instance_id": f"{kind}-{pair}-{arm}",
                    "instance_type": kind,
                    "ami": "ami-warm" if arm == "warm" else "ami-stock",
                    "pull_seconds": 10 if arm == "warm" else 100,
                    "total_seconds": 180 if arm == "warm" else 300,
                }
                metrics = {
                    "gpu_name": "NVIDIA L4" if kind == "g6.xlarge" else "NVIDIA A10G",
                    "nn_metrics": {"total": {"mae": 1.5, "rmse": 2.0}},
                    "cohorts": {"elite_top24": {"mae": 1.0}},
                }
                state["resources"].append(
                    {
                        "instance_type": kind,
                        "pair": pair,
                        "arm": arm,
                        "jobs": {
                            pos: {
                                "status": "SUCCEEDED",
                                "lifecycle": copy.deepcopy(timing),
                                "metrics": copy.deepcopy(metrics),
                            }
                            for pos in POSITIONS
                        },
                    }
                )
    return state


def test_canary_requires_separate_cold_hosts_full_coverage_and_measured_savings():
    from infra.batch.warm_ami import assess

    assert assess(paired_evidence())["passed"]
    missing = paired_evidence()
    missing["resources"].pop()
    assert not assess(missing)["passed"]
    reused = paired_evidence()
    reused["resources"][1]["jobs"]["RB"]["lifecycle"]["instance_id"] = reused["resources"][0][
        "jobs"
    ]["RB"]["lifecycle"]["instance_id"]
    assert any("reused" in error for error in assess(reused)["errors"])


@pytest.mark.parametrize(
    "mutation",
    [
        lambda state: [
            r["jobs"]["RB"]["lifecycle"].update(pull_seconds=80)
            for r in state["resources"]
            if r["arm"] == "warm"
        ],
        lambda state: [
            r["jobs"]["RB"]["lifecycle"].update(total_seconds=500)
            for r in state["resources"]
            if r["arm"] == "warm"
        ],
        lambda state: state["resources"][1]["jobs"]["K"]["metrics"]["nn_metrics"]["total"].update(
            mae=1.6
        ),
        lambda state: state["resources"][1]["jobs"]["DST"]["metrics"]["cohorts"][
            "elite_top24"
        ].update(mae=1.2),
        lambda state: state["resources"][0]["jobs"]["RB"]["lifecycle"].pop("total_seconds"),
        lambda state: [r["jobs"].pop("K") for r in state["resources"]],
    ],
)
def test_failed_canary_gate_is_not_accepted(mutation):
    from infra.batch.warm_ami import assess

    state = paired_evidence()
    mutation(state)
    assert not assess(state)["passed"]


def test_numeric_template_version_preserves_existing_userdata():
    from infra.batch.warm_ami import template

    ec2 = Mock()
    ec2.describe_launch_template_versions.return_value = {
        "LaunchTemplateVersions": [
            {
                "LaunchTemplateId": "lt-one",
                "VersionNumber": 17,
                "LaunchTemplateData": {"UserData": "existing"},
            }
        ]
    }
    selected, data = template(
        ec2, {"launchTemplate": {"launchTemplateName": "current", "version": "$Latest"}}
    )
    assert selected == {"launchTemplateId": "lt-one", "version": "17"}
    assert data == {"UserData": "existing"}


def test_cleanup_cannot_touch_another_runs_resources(tmp_path):
    from infra.batch.warm_ami import cleanup

    batch, ec2 = Mock(), Mock()
    batch.describe_compute_environments.return_value = {
        "computeEnvironments": [{"tags": {"ff-warm-ami-run": "another-run"}}]
    }
    batch.describe_job_queues.return_value = {"jobQueues": []}
    state = {"id": "our-run", "resources": [{"name": "unrelated", "jobs": {}}]}
    with pytest.raises(ValueError, match="not owned"):
        cleanup({"batch": batch, "ec2": ec2}, state, tmp_path / "state.json")
    batch.delete_compute_environment.assert_not_called()
    batch.terminate_job.assert_not_called()
    ec2.delete_launch_template.assert_not_called()


def test_operator_workflow_is_manual_and_its_shell_blocks_parse():
    import yaml

    workflow = yaml.safe_load((ROOT / ".github/workflows/warm-ami.yml").read_text())
    assert set(workflow.get("on", workflow.get(True))) == {"workflow_dispatch"}
    assert workflow["permissions"] == {"contents": "read"}
    for step in workflow["jobs"]["warm-ami"]["steps"]:
        if "run" in step:
            checked = subprocess.run(
                ["bash", "-n"], input=step["run"], text=True, capture_output=True
            )
            assert checked.returncode == 0, checked.stderr
    assert "--checkpoint-prefix" in (ROOT / ".github/workflows/warm-ami.yml").read_text()


def test_cleanup_deletes_invalid_disabled_environment(tmp_path):
    from infra.batch.warm_ami import cleanup

    batch, ec2 = Mock(), Mock()
    invalid = {"status": "INVALID", "state": "DISABLED", "tags": {"ff-warm-ami-run": "ours"}}
    batch.describe_compute_environments.side_effect = [
        {"computeEnvironments": [invalid]},
        {"computeEnvironments": [invalid]},
        {"computeEnvironments": []},
    ]
    batch.describe_job_queues.return_value = {"jobQueues": []}
    ec2.describe_launch_templates.return_value = {"LaunchTemplates": []}
    state = {"id": "ours", "resources": [{"name": "our-ce", "jobs": {}}]}
    cleanup({"batch": batch, "ec2": ec2}, state, tmp_path / "state.json")
    batch.delete_compute_environment.assert_called_once_with(computeEnvironment="our-ce")
    assert state["resources"][0]["cleaned"]


def test_cleanup_already_deleted_template_does_not_block_other_resources(tmp_path):
    from botocore.exceptions import ClientError

    from infra.batch.warm_ami import cleanup

    batch, ec2 = Mock(), Mock()
    batch.describe_compute_environments.return_value = {"computeEnvironments": []}
    batch.describe_job_queues.return_value = {"jobQueues": []}
    ec2.describe_launch_templates.side_effect = [
        ClientError(
            {"Error": {"Code": "InvalidLaunchTemplateId.NotFoundException"}},
            "DescribeLaunchTemplates",
        ),
        {
            "LaunchTemplates": [
                {
                    "LaunchTemplateId": "lt-two",
                    "Tags": [{"Key": "ff-warm-ami-run", "Value": "ours"}],
                }
            ]
        },
    ]
    ec2.delete_launch_template.side_effect = ClientError(
        {"Error": {"Code": "InvalidLaunchTemplateId.NotFoundException"}}, "DeleteLaunchTemplate"
    )
    state = {
        "id": "ours",
        "resources": [
            {"name": name, "jobs": {}, "launch_template": template}
            for name, template in [("one", "lt-one"), ("two", "lt-two")]
        ],
    }
    cleanup({"batch": batch, "ec2": ec2}, state, tmp_path / "state.json")
    assert all(row["cleaned"] for row in state["resources"])
    ec2.delete_launch_template.assert_called_once_with(LaunchTemplateId="lt-two")


def test_cleanup_verifies_even_a_recorded_templates_live_ownership(tmp_path):
    from infra.batch.warm_ami import cleanup

    batch, ec2 = Mock(), Mock()
    batch.describe_compute_environments.return_value = {"computeEnvironments": []}
    batch.describe_job_queues.return_value = {"jobQueues": []}
    ec2.describe_launch_templates.return_value = {
        "LaunchTemplates": [{"LaunchTemplateId": "lt-other", "Tags": []}]
    }
    state = {
        "id": "ours",
        "resources": [{"name": "gone", "jobs": {}, "launch_template": "lt-other"}],
    }
    with pytest.raises(ValueError, match="unowned launch template"):
        cleanup({"batch": batch, "ec2": ec2}, state, tmp_path / "state.json")
    ec2.delete_launch_template.assert_not_called()


def test_stock_rollback_explicitly_removes_template_and_refreshes_ami(tmp_path, monkeypatch):
    from infra.batch import warm_ami

    selected = {"launchTemplateId": "lt-current", "version": "4"}
    resources = {
        "type": "SPOT",
        "minvCpus": 0,
        "maxvCpus": 64,
        "desiredvCpus": 0,
        "instanceTypes": ["g6.xlarge", "g5.xlarge"],
        "subnets": ["subnet-one"],
        "securityGroupIds": ["sg-one"],
    }
    current = {"status": "VALID", "ecsClusterArn": "cluster", "computeResources": resources}
    batch, ec2, ecs = Mock(), Mock(), Mock()
    batch.describe_compute_environments.return_value = {"computeEnvironments": [current]}
    ecs.list_tasks.return_value = {"taskArns": []}
    monkeypatch.setattr(warm_ami, "template", Mock(side_effect=[(selected, {}), ({}, {})]))
    monkeypatch.setattr(warm_ami, "wait_until", lambda *args, **kwargs: current)
    evidence = {"selected_template": selected, "previous_template": {}, "candidate_ami": "ami-warm"}
    warm_ami.activate(
        {"batch": batch, "ec2": ec2, "ecs": ecs},
        evidence,
        tmp_path / "rollback.json",
        rollback=True,
    )
    assert batch.update_compute_environment.call_args.kwargs["computeResources"] == {
        "launchTemplate": {"launchTemplateId": ""},
        "updateToLatestImageVersion": True,
    }


def test_one_pair_smoke_requires_metric_parity_but_cannot_pass_full_gate():
    from infra.batch.warm_ami import assess, smoke_assessment

    state = paired_evidence()
    state["resources"] = [
        r for r in state["resources"] if r["instance_type"] == "g5.xlarge" and r["pair"] == 0
    ]
    for resource in state["resources"]:
        resource["jobs"] = {"RB": resource["jobs"]["RB"]}
    assert smoke_assessment(state, "g5.xlarge")["passed"]
    assert not assess(state)["passed"]
    state["resources"][1]["jobs"]["RB"]["metrics"]["nn_metrics"]["total"]["mae"] += 0.2
    assert not smoke_assessment(state, "g5.xlarge")["passed"]


def test_freshness_catches_base_ami_updates_even_when_dependency_layers_match(monkeypatch):
    from infra.batch import warm_ami
    from src.scripts import resolve_training_image

    monkeypatch.setattr(
        resolve_training_image,
        "resolve_ec2",
        lambda *args: {"image_uri": "image", "image_sha": "sha"},
    )
    monkeypatch.setattr(warm_ami, "recipe", lambda sha: "recipe")
    monkeypatch.setattr(warm_ami, "layers", lambda *args: ["base", "deps", "application", "stamp"])
    bake = {
        "eligible": True,
        "source_ami": "ami-old",
        "dependency_recipe": "recipe",
        "image_uri": "image",
    }
    ssm = Mock()
    ssm.get_parameter.return_value = {"Parameter": {"Value": "ami-old"}}
    assert warm_ami.freshness(bake, Mock(), ssm=ssm)["fresh"]
    ssm.get_parameter.return_value = {"Parameter": {"Value": "ami-new"}}
    result = warm_ami.freshness(bake, Mock(), ssm=ssm)
    assert result["dependencies_current"]
    assert not result["base_ami_current"] and not result["fresh"]


def test_ecr_digest_can_have_multiple_tags_but_one_manifest():
    from infra.batch.warm_ami import layers

    digest = "sha256:" + "a" * 64
    manifest = json.dumps({"layers": [{"digest": "dependency"}, {"digest": "code"}]})
    client = Mock()
    client.batch_get_image.return_value = {
        "images": [
            {"imageId": {"imageDigest": digest, "imageTag": tag}, "imageManifest": manifest}
            for tag in ["latest", "b" * 40]
        ]
    }
    assert layers(client, "registry/repository@" + digest) == ["dependency", "code"]
    client.batch_get_image.return_value["images"][1]["imageManifest"] = json.dumps({"layers": []})
    with pytest.raises(ValueError, match="inconsistent"):
        layers(client, "registry/repository@" + digest)


def test_builder_preserves_ssm_newlines_pins_inputs_and_cleans_up(tmp_path):
    shim = tmp_path / "aws"
    shim.write_text(
        "#!/usr/bin/env python3\n"
        "import json, os, pathlib, sys\n"
        "args=sys.argv[1:]; service, operation=args[:2]\n"
        "def arg(flag): return args[args.index(flag)+1]\n"
        "with open(os.environ['WARM_TEST_CALLS'], 'a') as f: f.write(json.dumps(args)+'\\n')\n"
        "if operation == 'get-parameters': print('ami-0123456789abcdef0')\n"
        "elif operation == 'describe-images' and service == 'ec2':\n"
        "    print('al2023-ami-ecs-gpu-x86_64' if 'Name' in arg('--query') else 'available')\n"
        "elif operation == 'describe-images':\n"
        "    print(json.dumps([os.environ['WARM_TEST_SHA']]) if 'imageTags' in arg('--query') else 'sha256:'+'a'*64)\n"
        "elif operation == 'describe-security-groups': print('sg-0123')\n"
        "elif operation == 'run-instances': print('i-0123')\n"
        "elif operation == 'describe-instance-information': print('Online')\n"
        "elif operation == 'send-command':\n"
        "    source=arg('--parameters'); assert source.startswith('file://')\n"
        "    commands=json.loads(pathlib.Path(source[7:]).read_text())['commands']\n"
        "    assert len(commands)==1 and commands[0].splitlines()[0]=='set -euo pipefail'\n"
        "    assert '\\ndocker pull --quiet ' in commands[0]\n"
        "    assert '\\ndocker logout ' in commands[0]\n"
        "    assert '\\ncloud-init clean --logs\\n' in commands[0]\n"
        "    assert '--machine-id' not in commands[0]\n"
        "    assert '/etc/machine-id' in commands[0]\n"
        "    print('command-id')\n"
        "elif operation == 'get-command-invocation':\n"
        "    print('Success' if arg('--query')=='Status' else json.dumps(['sha256:'+str(i)*64 for i in range(5)]))\n"
        "elif operation == 'create-image': print('ami-0aaaaaaaaaaaaaaaa')\n"
        "elif operation in {'wait','stop-instances','terminate-instances','create-tags'}: pass\n"
        "else: raise AssertionError(args)\n"
    )
    shim.chmod(0o755)
    sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    manifest = tmp_path / "manifest.json"
    calls = tmp_path / "calls.jsonl"
    env = {
        **os.environ,
        "PATH": str(tmp_path) + os.pathsep + os.environ["PATH"],
        "WARM_TEST_CALLS": str(calls),
        "WARM_TEST_SHA": sha,
        "FF_WARM_AMI_SOURCE_AMI": "ami-0123456789abcdef0",
        "FF_WARM_AMI_MANIFEST": str(manifest),
    }
    result = subprocess.run(
        [
            "bash",
            "infra/batch/build-warm-ami.sh",
            "123456789012.dkr.ecr.us-east-1.amazonaws.com/ff-training:latest",
        ],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    baked = json.loads(manifest.read_text())
    assert baked["eligible"]
    assert baked["source_sha"] == sha
    assert baked["image_uri"].endswith("@sha256:" + "a" * 64)
    assert len(baked["dependency_layers"]) == 3
    operations = [json.loads(line) for line in calls.read_text().splitlines()]
    assert operations[-1][:2] == ["ec2", "terminate-instances"]
    assert "i-0123" in operations[-1]
    assert not any(args[:2] == ["batch", "update-compute-environment"] for args in operations)
