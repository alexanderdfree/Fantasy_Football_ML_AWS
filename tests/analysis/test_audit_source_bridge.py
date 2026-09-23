"""The source bridge permits only the reviewed observer identity correction."""

import pytest

from src.analysis.audit_source_bridge import IDENTITY_FUNCTION, verify_routing_only

pytestmark = pytest.mark.unit

PARENT = """def configure(config, *, arm):
    from src.shared.aggregate_targets import infer_position
    position = infer_position(config["targets"])
    return position
"""
CORRECTED = (
    IDENTITY_FUNCTION
    + """\ndef configure(config, *, arm):
    position = configured_position(config)
    return position
"""
)


def test_exact_observer_identity_replacement_is_allowed():
    verify_routing_only(PARENT, CORRECTED)


@pytest.mark.parametrize(
    "changed",
    [
        CORRECTED.replace("return position", 'config["loss_weights"] = {}\n    return position'),
        CORRECTED.replace('"TE", "K"', '"K"'),
        CORRECTED.replace('config.get("filter_fn")', 'config.get("other_fn")'),
    ],
)
def test_bridge_cannot_hide_numerical_or_identity_changes(changed):
    with pytest.raises(ValueError):
        verify_routing_only(PARENT, changed)
