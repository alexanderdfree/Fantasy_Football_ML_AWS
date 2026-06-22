"""Platform & hardware capability detection (reporting-only).

The single canonical "what hardware am I on?" helper for the project's target
environments (see the *Platform & hardware targets* section of CLAUDE.md):
Apple Silicon macOS, Windows 11 / WSL2 on an RTX 5080 (Blackwell ``sm_120``),
and AWS g4dn.xlarge (T4, ``sm_75``) / g5.xlarge (A10G, ``sm_86``) /
g6.xlarge (L4, ``sm_89``).

This module *reports* capabilities; it does **not** choose the run's device or
dtype. Device resolution stays in :mod:`src.shared.utils` (``requested_device`` /
``cuda_enabled`` / ``mps_enabled``), which layers the operator's ``FF_DEVICE``
override on top of what is available here. Any platform-specific optimization
should branch off :func:`detect_platform` rather than sniffing
``platform.system()`` ad-hoc, so the per-arch decision table lives in one place.
"""

from __future__ import annotations

import os
import platform
from dataclasses import dataclass

import torch

# Native BF16 Tensor Cores arrived with Ampere (sm_80). Comparing the compute
# capability tuple is version-stable, unlike ``torch.cuda.is_bf16_supported()``
# whose ``including_emulation`` default can report True on a T4 (sm_75) via a
# software path — exactly the case PR #293 hung on. (See auto-memory: GPU dtype
# × compute-capability.)
_BF16_MIN_CAPABILITY = (8, 0)

_OS_NAMES = {"Darwin": "macOS", "Windows": "Windows", "Linux": "Linux"}


def _normalize_os(system: str) -> str:
    return _OS_NAMES.get(system, system)


def _detect_is_wsl() -> bool:
    """True on WSL2 — a Linux userland whose kernel is built by Microsoft.

    Distinguishes the WSL2 target from native Windows: the native-Windows
    ``OPENBLAS_NUM_THREADS=1`` crash-guard (SETUP.md) is a correctness
    requirement that does *not* apply under WSL2 (it's Linux).
    """
    if platform.system() != "Linux":
        return False
    return "microsoft" in platform.uname().release.lower()


@dataclass(frozen=True)
class PlatformInfo:
    """Immutable snapshot of the host's compute-relevant hardware/OS facts.

    Fields map to the rows/columns of the CLAUDE.md platform matrix. The
    CUDA-only fields (``gpu_name`` / ``compute_capability`` / ``sm`` /
    ``recommended_cuda_wheel``) are ``None`` on MPS and CPU hosts.
    """

    os: str  # "macOS" | "Windows" | "Linux"
    is_wsl: bool
    arch: str  # platform.machine(), e.g. "arm64", "x86_64", "AMD64"
    backend: str  # best available accelerator: "cuda" | "mps" | "cpu"
    gpu_name: str | None
    compute_capability: tuple[int, int] | None  # CUDA only, e.g. (12, 0)
    sm: str | None  # CUDA only, e.g. "sm_120"
    supports_bf16: bool  # native BF16 Tensor Cores (CUDA sm_80+)
    cpu_count: int | None
    recommended_cuda_wheel: str | None  # "cu130" (sm_120) | "cu126" | None

    def summary(self) -> str:
        """One-line, log-friendly description of the host."""
        if self.backend == "cuda":
            accel = f"CUDA {self.gpu_name} {self.sm} bf16={self.supports_bf16}"
        elif self.backend == "mps":
            accel = "MPS (Apple Silicon)"
        else:
            accel = "CPU"
        wsl = "/WSL2" if self.is_wsl else ""
        return f"{self.os}{wsl} {self.arch} · {accel} · {self.cpu_count} CPUs"


def detect_platform() -> PlatformInfo:
    """Detect the host's compute backend, GPU architecture, and OS.

    Reporting-only — does not consult ``FF_DEVICE`` or change any run's
    device/dtype. ``backend`` is the *best available* accelerator (CUDA > MPS >
    CPU); the device a run actually uses is resolved by :mod:`src.shared.utils`
    with the operator override applied (and MPS is opt-in, not auto-selected).
    """
    if torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability(0)
        cc = (major, minor)
        return PlatformInfo(
            os=_normalize_os(platform.system()),
            is_wsl=_detect_is_wsl(),
            arch=platform.machine(),
            backend="cuda",
            gpu_name=torch.cuda.get_device_name(0),
            compute_capability=cc,
            sm=f"sm_{major}{minor}",
            supports_bf16=cc >= _BF16_MIN_CAPABILITY,
            cpu_count=os.cpu_count(),
            # Blackwell (sm_120) needs the cu130 wheel; T4/A10G/L4 run on cu126.
            recommended_cuda_wheel="cu130" if major >= 12 else "cu126",
        )

    backend = "mps" if torch.backends.mps.is_available() else "cpu"
    return PlatformInfo(
        os=_normalize_os(platform.system()),
        is_wsl=_detect_is_wsl(),
        arch=platform.machine(),
        backend=backend,
        gpu_name="Apple MPS" if backend == "mps" else None,
        compute_capability=None,
        sm=None,
        supports_bf16=False,
        cpu_count=os.cpu_count(),
        recommended_cuda_wheel=None,
    )
