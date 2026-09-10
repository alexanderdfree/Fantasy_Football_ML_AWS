"""Exercise environment repair without modifying the test interpreter's libs."""

import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import fix_macos_openmp as openmp

pytestmark = pytest.mark.unit


@pytest.fixture
def environment(tmp_path, monkeypatch):
    prefix = tmp_path / "venv"
    bundled = prefix / "lib" / "python" / "torch" / "libomp.dylib"
    bundled.parent.mkdir(parents=True)
    bundled.write_bytes(b"original runtime")
    library = tmp_path / "homebrew" / "libomp.dylib"
    library.parent.mkdir()
    library.write_bytes(b"canonical runtime")
    monkeypatch.setattr(openmp.sys, "prefix", str(prefix))
    monkeypatch.setattr(openmp, "vendored_runtimes", lambda: [bundled])
    monkeypatch.setattr(openmp, "runtime_paths", lambda: {bundled.resolve()})
    return bundled, library


def test_repair_preserves_hardlinked_package_cache_and_is_idempotent(environment):
    bundled, library = environment
    cache = library.parent / "cached-original"
    os.link(bundled, cache)
    backup = openmp.repair(library)
    assert bundled.is_symlink()
    assert bundled.resolve() == library
    assert cache.read_bytes() == b"original runtime"
    assert list(backup.rglob("libomp.dylib"))[0].read_bytes() == b"original runtime"
    assert openmp.repair(library) is None


def test_failed_native_probe_restores_originals(environment, monkeypatch):
    bundled, library = environment
    second = bundled.parent / "sklearn" / "libomp.dylib"
    second.parent.mkdir()
    second.write_bytes(b"second runtime")
    monkeypatch.setattr(openmp, "vendored_runtimes", lambda: [bundled, second])
    monkeypatch.setattr(openmp, "runtime_paths", lambda: {library, Path("/conflicting/libomp")})
    with pytest.raises(RuntimeError, match="Expected one OpenMP runtime"):
        openmp.repair(library)
    assert not bundled.is_symlink()
    assert bundled.read_bytes() == b"original runtime"
    assert not second.is_symlink()
    assert second.read_bytes() == b"second runtime"


def test_reinstall_can_be_repaired_without_overwriting_old_backup(environment):
    bundled, library = environment
    first = openmp.repair(library)
    bundled.unlink()
    bundled.write_bytes(b"upgraded runtime")
    second = openmp.repair(library)
    assert first != second
    assert list(first.rglob("libomp.dylib"))[0].read_bytes() == b"original runtime"
    assert list(second.rglob("libomp.dylib"))[0].read_bytes() == b"upgraded runtime"


def test_cancelled_verification_restores_original(environment, monkeypatch):
    bundled, library = environment

    def interrupted_probe():
        raise KeyboardInterrupt

    monkeypatch.setattr(openmp, "runtime_paths", interrupted_probe)
    with pytest.raises(KeyboardInterrupt):
        openmp.repair(library)
    assert not bundled.is_symlink()
    assert bundled.read_bytes() == b"original runtime"


def test_cancelled_replacement_restores_original(environment, monkeypatch):
    bundled, library = environment
    replace = os.replace

    def replace_then_interrupt(source, destination):
        replace(source, destination)
        if Path(destination) == bundled and bundled.is_symlink():
            raise KeyboardInterrupt

    monkeypatch.setattr(openmp.os, "replace", replace_then_interrupt)
    with pytest.raises(KeyboardInterrupt):
        openmp.repair(library)
    assert not bundled.is_symlink()
    assert bundled.read_bytes() == b"original runtime"


def test_discovery_excludes_inherited_packages_but_keeps_repaired_links(tmp_path, monkeypatch):
    prefix = tmp_path / "venv"
    prefix.mkdir()
    external = tmp_path / "external" / "libomp.dylib"
    external.parent.mkdir()
    external.touch()
    owned = prefix / "libomp.dylib"
    owned.symlink_to(external)
    dist = SimpleNamespace(
        files=[Path("owned/libomp.dylib"), Path("external/libomp.dylib")],
        locate_file=lambda entry: owned if entry.parts[0] == "owned" else external,
    )
    alias = tmp_path / "venv-alias"
    alias.symlink_to(prefix, target_is_directory=True)
    monkeypatch.setattr(openmp.sys, "prefix", str(alias))
    monkeypatch.setattr(openmp.importlib.metadata, "distribution", lambda _: dist)
    assert openmp.vendored_runtimes() == [owned]


def test_check_only_does_not_repair(environment, monkeypatch):
    bundled, library = environment
    monkeypatch.setattr(openmp.sys, "platform", "darwin")
    assert openmp.main(["--library", str(library)]) == 1
    assert not bundled.is_symlink()
    assert not (Path(openmp.sys.prefix) / ".openmp-backups").exists()


@pytest.mark.parametrize("platform", ["linux", "win32"])
def test_other_platforms_do_not_discover_or_modify_libraries(platform, monkeypatch):
    monkeypatch.setattr(openmp.sys, "platform", platform)
    monkeypatch.setattr(openmp.subprocess, "run", lambda *a, **kw: pytest.fail("native probe"))
    assert openmp.main(["--apply"]) == 0
