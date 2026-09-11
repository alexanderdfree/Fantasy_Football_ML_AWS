"""Required CI must run when a review stack targets another Codex branch."""

from fnmatch import fnmatchcase
from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.unit
WORKFLOWS = Path(__file__).resolve().parents[1] / ".github/workflows"


@pytest.mark.parametrize("workflow", ["tests.yml", "codeql.yml"])
def test_stack_bases_receive_checks_without_enabling_feature_branch_pushes(workflow):
    events = yaml.load((WORKFLOWS / workflow).read_text(), Loader=yaml.BaseLoader)["on"]
    for base in (
        "main",
        "codex/context-cleanup-followup",
        "codex/fix-expert-comparison-fairness",
        "codex/design-audit-foundations",
        "codex/fix-inheritance-reception",
    ):
        assert any(fnmatchcase(base, pattern) for pattern in events["pull_request"]["branches"])
    assert events["push"]["branches"] == ["main"]


@pytest.mark.parametrize("workflow", ["deploy.yml", "batch-image.yml"])
def test_production_push_triggers_still_target_only_main(workflow):
    events = yaml.load((WORKFLOWS / workflow).read_text(), Loader=yaml.BaseLoader)["on"]
    assert events["push"]["branches"] == ["main"]
