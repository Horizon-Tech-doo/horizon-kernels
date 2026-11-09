# Copied as conftest.py next to upstream tests. TVM's tvm.testing probes CUDA/OpenCL/Vulkan/Metal through the
# fat wheel's stubs, and the CUDA stub abort()s on a ROCm-only host during collection. Answer "absent" for
# every non-ROCm device type without touching the stubs; ROCm (DLPack type 10) keeps the real check.
import sys
import tilelang  # noqa: F401  (puts the bundled tvm on sys.path)
import tvm.runtime.device  # noqa: F401
_Device = sys.modules["tvm.runtime.device"].Device
_orig_get = _Device._GetDeviceAttr


def _get(self, device_type, device_id, attr_id):
    if attr_id == 0 and int(device_type) != 10:
        return 0
    return _orig_get(self, device_type, device_id, attr_id)


_Device._GetDeviceAttr = _get
