import re
import shlex

import pytest

from src.tuning import feature_selection, launch_ab
from src.tuning.ab_harness import build_cells, resolve_spec
from src.tuning.tune_nn_storage import stacked_default_seed_list

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("flags", [[], ["--stacked"], ["--no-stacked"]])
@pytest.mark.parametrize("positions", [["RB"], ["QB", "RB", "WR", "TE"], ["K", "DST"]])
def test_stage1_commands_execute_the_advertised_seed_grid(capsys, flags, positions):
    assert feature_selection.main(["plan", "--positions", *positions, *flags]) == 0
    output = capsys.readouterr().out
    commands = re.findall(
        r"^\s*~(\d+) cells\s+(python -m src\.tuning\.launch_ab .+)$", output, re.M
    )
    assert len(commands) == 2
    for advertised, command in commands:
        args = launch_ab._build_parser().parse_args(shlex.split(command)[3:])
        spec = resolve_spec(args.spec, positions=args.positions, seeds=args.seeds, only=args.only)
        assert len(build_cells(spec)) == int(advertised)
        assert len(build_cells(spec)) <= args.max_cells
        stacked = flags != ["--no-stacked"] and positions[0] not in {"K", "DST"}
        assert args.stacked_seeds is stacked
        if stacked:
            assert args.seeds == spec.seeds == stacked_default_seed_list()
        else:
            assert args.seeds is None
            assert spec.seeds == [42, 123, 7]
