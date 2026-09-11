"""Explicit remote boundaries for CLI flow tests; identity validation stays real."""

import os


def mock_submission_boundaries(monkeypatch, launch):
    for name in ("FF_DATA_RELEASE", "FF_DATASET_ID", "FF_DATA_FORMAT"):
        monkeypatch.setenv(name, "")
        monkeypatch.delenv(name, raising=False)

    def binding(*_, **__):
        sha = os.environ["FF_TRAIN_GIT_SHA"]
        return {
            "image_sha": sha,
            "gpu_definition": launch._job_definition_for("QB"),
            "cpu_definition": (
                launch._job_definition_for("QB", "cpu")
                if launch.JOB_DEFINITION_CPU and launch.JOB_QUEUE_CPU
                else ""
            ),
            "gpu_image": "registry/train:" + sha,
            "cpu_image": "registry/train:" + sha,
        }

    def pin(*_, **__):
        from src.orchestration.datasets import bind_data_release

        return bind_data_release("f" * 64)

    monkeypatch.setattr(launch, "resolve_launch_binding", binding)
    monkeypatch.setattr(launch, "pin_data_release", pin)
    monkeypatch.setattr(launch, "validate_local_publish", lambda *_: None)
