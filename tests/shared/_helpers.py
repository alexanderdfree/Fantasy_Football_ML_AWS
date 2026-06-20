"""Shared test helpers for the ``tests/shared`` suite.

Small, dependency-light fixtures reused across more than one test module so the
same builder isn't maintained in two places (e.g. the model_sync sync vs refresh
suites both stage gzipped tarballs and stub S3 ``get_object`` bodies).
"""

from __future__ import annotations

import io
import tarfile


def make_tarball(members: dict[str, bytes]) -> bytes:
    """Build an in-memory ``.tar.gz`` from ``{member_name: bytes}``."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for name, data in members.items():
            info = tarfile.TarInfo(name=name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return buf.getvalue()


class FakeBody:
    """Minimal stand-in for a boto3 ``StreamingBody`` (``.read()`` returns bytes)."""

    def __init__(self, data: bytes):
        self._data = data

    def read(self) -> bytes:
        return self._data
