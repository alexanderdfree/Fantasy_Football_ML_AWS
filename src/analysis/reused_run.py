"""Read-only analysis adapter for exact whole-run reuse."""

from dataclasses import replace

from src.training.context import RunContext, current_context


def run_position(position, *, seed=42, frames=None, fresh=False):
    from src.shared.registry import get_runner
    from src.tuning._execution import isolated_outputs

    base = current_context() or RunContext.defaults(seed=seed)
    with isolated_outputs(str(base.data_root), seed=seed) as context:
        context = replace(
            context, raw_root=base.raw_root, reuse_results=not fresh, report_sink=None
        )
        runner = get_runner(position)
        kwargs = {"seed": seed, "context": context}
        if frames is not None and position not in {"K", "DST"}:
            return runner(*frames, **kwargs)
        return runner(**kwargs)
