"""No-fit orchestration contracts; producer/remote operations are fakes."""

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from src.analysis import correctness_batch_gates as gates


class CorrectnessGatesTests(unittest.TestCase):
    def test_prepare_data_rejects_existing_inputs_before_producer(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "data/raw").mkdir(parents=True)
            run = Mock()
            with self.assertRaisesRegex(ValueError, "clean"):
                gates.prepare_data(root, "a" * 40, object(), "bucket", root / "log", run=run)
            run.assert_not_called()

    def test_prepare_data_rebuilds_before_staging_without_promotion(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = "a" * 40

            def run(command, **kwargs):
                self.assertEqual(command[-1], "src.data.maintenance_build")
                self.assertEqual(kwargs["env"]["NFLREADPY_TIMEOUT"], "120")
                self.assertNotIn("FF_DATA_RELEASE", kwargs["env"])
                self.assertNotIn("FF_DATASET_ID", kwargs["env"])
                self.assertEqual(kwargs["env"]["GITHUB_SHA"], source)
                (root / "data/splits").mkdir(parents=True)
                (root / "data/splits/release-inputs.json").write_text(
                    json.dumps({"git_sha": source, "data_producer_sha256": "b" * 64, "files": {}})
                )

            publish = Mock(return_value="c" * 64)
            with patch.dict(os.environ, {"FF_DATA_RELEASE": "old", "FF_DATASET_ID": "old"}):
                receipt = gates.prepare_data(
                    root, source, object(), "bucket", root / "log", run=run, publish=publish
                )
            self.assertFalse(publish.call_args.kwargs["promote"])
            self.assertFalse(receipt["promoted"])
            self.assertEqual(receipt["data_release"], "c" * 64)

    def test_prepare_failure_never_stages(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            publish = Mock()
            with self.assertRaisesRegex(RuntimeError, "upstream"):
                gates.prepare_data(
                    root,
                    "a" * 40,
                    object(),
                    "bucket",
                    root / "log",
                    run=Mock(side_effect=RuntimeError("upstream")),
                    publish=publish,
                )
            publish.assert_not_called()

    def test_test_credentials_cannot_reach_production(self):
        with tempfile.TemporaryDirectory() as directory:
            env = gates.test_environment(
                {
                    "AWS_ACCESS_KEY_ID": "real",
                    "AWS_CONTAINER_CREDENTIALS_RELATIVE_URI": "/credential",
                    "FF_MODEL_S3_BUCKET": "production",
                    "S3_BUCKET": "production",
                },
                Path(directory),
                "cpu",
            )
            self.assertEqual(env["AWS_ACCESS_KEY_ID"], "testing")
            self.assertNotIn("AWS_CONTAINER_CREDENTIALS_RELATIVE_URI", env)
            self.assertEqual(env["FF_MODEL_S3_BUCKET"], "")
            self.assertNotIn("S3_BUCKET", env)

    def test_numeric_skips_and_empty_collection_fail(self):
        for counts in ({}, {"tests": 3, "skipped": 1}, {"tests": 2, "failure": 1}):
            self.assertFalse(gates.suite_passed("numerical_gpu", 0, counts))
        self.assertTrue(gates.suite_passed("unit", 0, {"tests": 4, "skipped": 1}))
        self.assertFalse(gates.suite_passed("unit", 0, {"tests": 1, "skipped": 1}))
        self.assertFalse(gates.suite_passed("unit", 2, {"tests": 1}))

    def test_numerical_cpu_excludes_only_unavailable_devices(self):
        command = gates.test_command("numerical_cpu")
        self.assertIn("not native_mps and not native_cuda", command)
        self.assertNotIn("not native_cuda", gates.test_command("numerical_gpu"))


if __name__ == "__main__":
    unittest.main()
