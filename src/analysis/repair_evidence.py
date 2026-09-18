"""Pure checks shared by the development reporter and confirmation gate."""


def complete_sample_counts(left, right, expected_left, expected_right):
    """Reject model-specific row filtering, including equal but incomplete samples."""
    counts = (left.get("n"), right.get("n"), expected_left, expected_right)
    return all(type(n) is int and n > 0 for n in counts) and len(set(counts)) == 1


def execution_signature(record):
    """Compare recorded execution settings while allowing distinct run/job IDs."""
    execution = record.get("execution")
    if not isinstance(execution, dict):
        raise ValueError("missing execution settings")
    overrides = execution.get("overrides")
    if (
        execution.get("regime") != "eager"
        or not isinstance(execution.get("device"), str)
        or not execution["device"].startswith("cuda")
        or execution.get("seed") != record.get("seed")
        or not isinstance(overrides, dict)
        or not all(isinstance(k, str) and isinstance(v, str) for k, v in overrides.items())
        or type(record.get("tf32_matmul")) is not bool
    ):
        raise ValueError("incomplete or incompatible execution settings")
    trainers = []
    for trainer in record.get("trainers", []):
        device = trainer.get("device")
        if (
            not isinstance(device, str)
            or not device.startswith("cuda")
            or trainer.get("amp") is not False
            or type(trainer.get("graph")) is not bool
        ):
            raise ValueError("missing production FP32 trainer settings")
        trainers.append((trainer.get("family"), device, trainer["amp"], trainer["graph"]))
    if len(trainers) != 2 or {t[0] for t in trainers} != {"nn", "attn_nn"}:
        raise ValueError("missing neural trainer execution settings")
    return (
        execution["regime"],
        execution["device"],
        execution["seed"],
        tuple(sorted(overrides.items())),
        record["tf32_matmul"],
        tuple(sorted(trainers)),
    )
