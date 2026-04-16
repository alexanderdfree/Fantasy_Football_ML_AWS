"""Mock AWS SDK dependencies so sagemaker/launch.py can be imported locally."""
import sys
from unittest import mock

# boto3 is not installed locally — mock it before any test imports launch.py
if "boto3" not in sys.modules:
    sys.modules["boto3"] = mock.MagicMock()

# launch.py does `import sagemaker` (expecting AWS SDK) and
# `from sagemaker.pytorch import PyTorch`. Since the local sagemaker/ directory
# shadows the AWS package name, we add the expected sub-modules/attributes
# to the local namespace package so launch.py's imports succeed.
import sagemaker as _local_pkg

if not hasattr(_local_pkg, "Session"):
    _local_pkg.Session = mock.MagicMock()

if "sagemaker.pytorch" not in sys.modules:
    _mock_pytorch = mock.MagicMock()
    sys.modules["sagemaker.pytorch"] = _mock_pytorch
    _local_pkg.pytorch = _mock_pytorch
