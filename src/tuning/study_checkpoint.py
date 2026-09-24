"""SQLite-safe study checkpoints and bounded campaign resume budgets."""

from __future__ import annotations

import contextlib
import os
import signal
import sqlite3
import tempfile
import threading
import time
from pathlib import Path


def sqlite_backup(path, timeout=120):
    if not Path(path).is_file():
        raise FileNotFoundError(path)
    fd, destination = tempfile.mkstemp(prefix="ff-study-", suffix=".db")
    os.close(fd)
    try:
        with (
            contextlib.closing(
                sqlite3.connect(
                    Path(path).resolve().as_uri() + "?mode=ro", uri=True, timeout=timeout
                )
            ) as source,
            contextlib.closing(sqlite3.connect(destination)) as target,
        ):
            source.backup(target)
        return destination
    except BaseException:
        Path(destination).unlink(missing_ok=True)
        raise


class StudyCheckpoint:
    """One writer, serialized snapshots, optional ETag protection for campaigns."""

    def __init__(
        self, bucket, db_path, key_prefix, *, s3=None, backup=sqlite_backup, conditional=False
    ):
        if s3 is None:
            import boto3

            s3 = boto3.client("s3")
        self.s3 = s3
        self.bucket = bucket
        self.db_path = str(db_path)
        self.key_prefix = key_prefix.rstrip("/")
        self.backup = backup
        self.conditional = conditional
        self.etag = None
        self.lock = threading.RLock()

    def _study_key(self):
        return f"{self.key_prefix}/study.db"

    def _results_key(self):
        return f"{self.key_prefix}/results.json"

    def download_study_db(self):
        from botocore.exceptions import ClientError

        path = Path(self.db_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + ".download")
        try:
            if self.conditional:
                response = self.s3.get_object(Bucket=self.bucket, Key=self._study_key())
                self.etag = response["ETag"]
                with temporary.open("wb") as target:
                    body = response["Body"]
                    try:
                        for chunk in iter(lambda: body.read(1024 * 1024), b""):
                            target.write(chunk)
                    finally:
                        body.close()
                with contextlib.closing(sqlite3.connect(temporary)) as db:
                    if db.execute("PRAGMA quick_check").fetchone()[0] != "ok":
                        raise ValueError("Invalid remote study checkpoint")
            else:
                self.s3.download_file(self.bucket, self._study_key(), self.db_path)
                return
            os.replace(temporary, path)
        except ClientError as exc:
            if exc.response["Error"]["Code"] not in {"404", "NoSuchKey", "NotFound"}:
                raise
        finally:
            temporary.unlink(missing_ok=True)

    def upload_study_db(self):
        if not Path(self.db_path).is_file():
            return
        with self.lock:
            snapshot = self.backup(self.db_path)
            try:
                if self.conditional:
                    with open(snapshot, "rb") as body:
                        response = self.s3.put_object(
                            Bucket=self.bucket,
                            Key=self._study_key(),
                            Body=body,
                            **({"IfMatch": self.etag} if self.etag else {"IfNoneMatch": "*"}),
                        )
                    self.etag = response["ETag"]
                else:
                    self.s3.upload_file(snapshot, self.bucket, self._study_key())
            finally:
                Path(snapshot).unlink(missing_ok=True)

    def upload_results(self, path):
        if Path(path).is_file():
            self.s3.upload_file(str(path), self.bucket, self._results_key())


def campaign_study_path(filename):
    directory = os.environ.get("FF_CAMPAIGN_STUDY_DIR")
    if not directory:
        return filename
    Path(directory).mkdir(parents=True, exist_ok=True)
    return str(Path(directory) / filename)


def campaign_checkpoint(db_path, suffix):
    prefix = os.environ.get("FF_CAMPAIGN_STUDY_PREFIX")
    bucket = os.environ.get("FF_CAMPAIGN_BUCKET")
    if not prefix or not bucket:
        return None
    return StudyCheckpoint(bucket, db_path, f"{prefix}/{suffix}", conditional=True)


def recover_trials(study):
    """Called only by a newly owned campaign step, never a live shared study."""
    from optuna.trial import TrialState

    for trial in study.get_trials(deepcopy=False, states=(TrialState.RUNNING,)):
        study.tell(trial.number, state=TrialState.FAIL, skip_if_finished=True)


def remaining_attempts(study, target):
    return max(0, target - len(study.trials))


def claim_trial(study, target, db_path):
    """Reserve one MPS attempt atomically across the allocation's processes."""
    from src.tuning.campaign_io import local_lock

    path = Path(db_path)
    with local_lock(path.parent / f".{path.name}.attempts", blocking=True):
        if remaining_attempts(study, target) == 0:
            return None
        # ask() persists RUNNING before releasing the allocation lock, so a
        # sibling cannot reserve the same last remaining budget slot.
        return study.ask()


def run_claimed_trial(study, trial, objective):
    """Keep Optuna's COMPLETE/PRUNED/FAIL semantics for a reserved attempt."""
    import optuna

    try:
        value = objective(trial)
    except optuna.TrialPruned:
        study.tell(trial, state=optuna.trial.TrialState.PRUNED)
    except Exception:
        study.tell(trial, state=optuna.trial.TrialState.FAIL)
        raise
    else:
        study.tell(trial, value)


class CampaignBudget:
    """Track active time in SQLite so queue time is not charged on resume."""

    def __init__(self, study, timeout, checkpoint=None):
        self.study = study
        self.checkpoint = checkpoint
        self.timeout = timeout
        self.previous = float(study.user_attrs.get("campaign_active_seconds", 0))
        self.started = time.monotonic()
        self.stop = threading.Event()
        self.thread = None
        self.error = None
        self.old_handler = None
        self.lock = threading.RLock()

    @property
    def remaining(self):
        return None if self.timeout is None else max(0.0, self.timeout - self.previous)

    def save(self):
        with self.lock:
            self.study.set_user_attr(
                "campaign_active_seconds", self.previous + time.monotonic() - self.started
            )
            if self.checkpoint is not None:
                self.checkpoint.upload_study_db()

    def callback(self, study, trial):
        if self.error is not None:
            raise RuntimeError("Campaign checkpoint failed") from self.error
        self.save()

    def _heartbeat(self):
        while not self.stop.wait(30):
            try:
                self.save()
            except Exception as exc:
                self.error = exc
                return

    def __enter__(self):
        self.old_handler = signal.getsignal(signal.SIGTERM)

        def terminate(signum, frame):
            self.save()
            raise SystemExit(143)

        signal.signal(signal.SIGTERM, terminate)
        self.thread = threading.Thread(target=self._heartbeat, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.stop.set()
        self.thread.join(timeout=60)
        signal.signal(signal.SIGTERM, self.old_handler)
        if exc is None:
            self.save()
            if self.error is not None:
                raise RuntimeError("Campaign checkpoint failed") from self.error
        else:
            with contextlib.suppress(Exception):
                self.save()
