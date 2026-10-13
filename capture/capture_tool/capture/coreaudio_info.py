"""Read-only questions to Core Audio that PortAudio does not answer (macOS only; everything is best effort and returns None elsewhere).

Why: PortAudio's `Stream.blocksize` only echoes what was asked, never what the driver granted. The page showed "buffer 64" while the
interface was still at the 512 of its own panel (found by Roman: 1083 samples of round trip at "64" on a Fireface UC). The real number is
the device's kAudioDevicePropertyBufferFrameSize, read here while the stream is open."""

import sys
from ctypes import CDLL, POINTER, Structure, byref, c_double, c_uint32, c_void_p, create_string_buffer, sizeof
from typing import Optional

_ca = None


def _fourcc(code: str) -> int:
    return int.from_bytes(code.encode("ascii"), "big")


class _Address(Structure):
    _fields_ = [("selector", c_uint32), ("scope", c_uint32), ("element", c_uint32)]


def _core_audio():
    global _ca
    if _ca is None:
        _ca = CDLL("/System/Library/Frameworks/CoreAudio.framework/CoreAudio")
        _ca.AudioObjectGetPropertyDataSize.argtypes = [c_uint32, POINTER(_Address), c_uint32, c_void_p, POINTER(c_uint32)]
        _ca.AudioObjectGetPropertyData.argtypes = [c_uint32, POINTER(_Address), c_uint32, c_void_p, POINTER(c_uint32), c_void_p]
    return _ca


def _device_ids(ca) -> list:
    address = _Address(_fourcc("dev#"), _fourcc("glob"), 0)
    size = c_uint32(0)
    if ca.AudioObjectGetPropertyDataSize(1, byref(address), 0, None, byref(size)) != 0:
        return []
    ids = (c_uint32 * (size.value // 4))()
    if ca.AudioObjectGetPropertyData(1, byref(address), 0, None, byref(size), ids) != 0:
        return []
    return list(ids)


def _device_name(ca, device_id: int) -> Optional[str]:
    address = _Address(_fourcc("name"), _fourcc("glob"), 0)           # the plain C-string name, the same one PortAudio lists
    buf = create_string_buffer(256)
    size = c_uint32(256)
    if ca.AudioObjectGetPropertyData(device_id, byref(address), 0, None, byref(size), buf) != 0:
        return None
    return buf.value.decode("utf-8", errors="ignore")


def buffer_frames(device_name: str) -> Optional[int]:
    """The I/O buffer (frames) the device has right now, or None if it cannot be read."""
    if sys.platform != "darwin" or not device_name:
        return None
    try:
        ca = _core_audio()
        for device_id in _device_ids(ca):
            if _device_name(ca, device_id) == device_name:
                value = c_uint32(0)
                size = c_uint32(4)
                address = _Address(_fourcc("fsiz"), _fourcc("glob"), 0)
                if ca.AudioObjectGetPropertyData(device_id, byref(address), 0, None, byref(size), byref(value)) == 0 and value.value > 0:
                    return int(value.value)
    except Exception:
        return None
    return None
