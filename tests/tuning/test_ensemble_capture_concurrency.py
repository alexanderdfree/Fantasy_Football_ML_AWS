import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import numpy as np
import pytest

from src.shared.neural_net import MultiHeadNetWithHistory
from src.shared.training import MultiHeadTrainer
from src.tuning.ab_ensemble_seeds import capture_attention_construction, capture_seeds

pytestmark = pytest.mark.unit


@pytest.fixture
def ordinary_methods(monkeypatch):
    calls = []

    def train(self, *args):
        calls.append(("train", self.owner))
        return {"trained": self.owner}

    def predict(self, X_static, *args):
        calls.append(("predict", self.owner, len(X_static)))
        return {"yards": np.ones(len(X_static))}

    monkeypatch.setattr(MultiHeadTrainer, "train", train)
    monkeypatch.setattr(MultiHeadNetWithHistory, "predict_numpy", predict)
    return train, predict, calls


def _train(owner):
    return MultiHeadTrainer.train(SimpleNamespace(owner=owner), object(), object(), 1)


def _predict(owner):
    return MultiHeadNetWithHistory.predict_numpy(
        SimpleNamespace(owner=owner),
        np.ones((3, 1)),
        np.ones((3, 1, 1)),
        np.ones((3, 1), dtype=bool),
        None,
    )


def test_concurrent_seed_captures_belong_to_their_trial(monkeypatch, ordinary_methods):
    barrier = threading.Barrier(2)

    def runner(*args, **kwargs):
        barrier.wait(timeout=5)
        _train(threading.current_thread().name)
        _predict(threading.current_thread().name)
        barrier.wait(timeout=5)

    monkeypatch.setattr("src.shared.registry.get_runner", lambda pos: runner)

    def capture():
        owner = threading.current_thread().name
        captures, test = capture_seeds("QB", [42, 43], {})
        return owner, [row["trainer"].owner for row in captures], test

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(capture) for _ in range(2)]
        results = [future.result(timeout=10) for future in futures]
    for owner, owners, test in results:
        assert owners == [owner, owner]
        assert test["args"][0].shape == (3, 1)
    assert MultiHeadTrainer.train is ordinary_methods[0]
    assert MultiHeadNetWithHistory.predict_numpy is ordinary_methods[1]


def test_capture_does_not_intercept_unrelated_thread(ordinary_methods):
    captures, test = [], {}
    with capture_attention_construction(captures, test), ThreadPoolExecutor(max_workers=1) as pool:
        assert pool.submit(_train, "ordinary").result(timeout=5) == {"trained": "ordinary"}
        pred = pool.submit(_predict, "ordinary").result(timeout=5)
        np.testing.assert_array_equal(pred["yards"], np.ones(3))
    assert not captures and not test
    assert ordinary_methods[2] == [("train", "ordinary"), ("predict", "ordinary", 3)]


def test_one_failed_capture_does_not_restore_over_another(ordinary_methods):
    a_entered, b_entered, a_exited = (threading.Event() for _ in range(3))

    def failed_capture():
        try:
            with capture_attention_construction([], {}):
                a_entered.set()
                assert b_entered.wait(5)
                raise RuntimeError("injected capture failure")
        except RuntimeError:
            pass
        finally:
            a_exited.set()

    def healthy_capture():
        assert a_entered.wait(5)
        captured = []
        with capture_attention_construction(captured, {}):
            b_entered.set()
            assert a_exited.wait(5)
            _train("healthy")
        return captured

    with ThreadPoolExecutor(max_workers=2) as pool:
        failed = pool.submit(failed_capture)
        healthy = pool.submit(healthy_capture)
        failed.result(timeout=10)
        assert [row["trainer"].owner for row in healthy.result(timeout=10)] == ["healthy"]
    assert MultiHeadTrainer.train is ordinary_methods[0]
    assert MultiHeadNetWithHistory.predict_numpy is ordinary_methods[1]


def test_nested_failed_and_repeated_captures_restore_methods(ordinary_methods):
    for _ in range(3):
        outer, inner = [], []
        with capture_attention_construction(outer, {}):
            _train("outer-before")
            with pytest.raises(ValueError, match="inner failure"):
                with capture_attention_construction(inner, {}):
                    _train("inner")
                    raise ValueError("inner failure")
            _train("outer-after")
        assert [row["trainer"].owner for row in outer] == ["outer-before", "outer-after"]
        assert [row["trainer"].owner for row in inner] == ["inner"]
        assert MultiHeadTrainer.train is ordinary_methods[0]
        assert MultiHeadNetWithHistory.predict_numpy is ordinary_methods[1]
        assert _train("ordinary") == {"trained": "ordinary"}
