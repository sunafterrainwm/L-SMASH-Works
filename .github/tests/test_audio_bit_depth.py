#!/usr/bin/env python3
"""Check that the VapourSynth audio sources honour the source bit depth.

24bit PCM has to be reported as 24bit integer audio instead of being widened to
32bit. VapourSynth keeps such samples left aligned in a 32bit container, which
is what vspipe and the other audio sources expect, so the values themselves stay
at full 32bit scale.
"""

import array
import os
import struct
import sys
import tempfile

import vapoursynth as vs

core = vs.core
# Without an explicit path the plugin is expected to be autoloaded.
plugin = os.environ.get("LSMAS_PLUGIN")
if plugin:
    core.std.LoadPlugin(plugin)
elif not hasattr(core, "lsmas"):
    print("FAIL: the lsmas plugin is neither autoloaded nor given via LSMAS_PLUGIN")
    sys.exit(1)


def write_wav(path, bits, values, sample_type="int", rate=48000):
    bytes_per_sample = bits // 8
    if sample_type == "int":
        tag = 1
        data = b"".join((v & ((1 << bits) - 1)).to_bytes(bytes_per_sample, "little") for v in values)
    else:
        tag = 3
        data = b"".join(struct.pack("<f", v) for v in values)
    fmt = struct.pack("<HHIIHH", tag, 1, rate, rate * bytes_per_sample, bytes_per_sample, bits)
    body = b"fmt " + struct.pack("<I", len(fmt)) + fmt + b"data" + struct.pack("<I", len(data)) + data
    with open(path, "wb") as f:
        f.write(b"RIFF" + struct.pack("<I", 4 + len(body)) + b"WAVE" + body)


def read_samples(node, count):
    # VapourSynth hands out integer audio planes as unsigned, so reinterpret them.
    if node.sample_type == vs.FLOAT:
        fmt = "f"
    else:
        fmt = {1: "b", 2: "h", 4: "i"}[node.bytes_per_sample]
    out = array.array(fmt)
    for frame in node.frames():
        out.frombytes(bytes(memoryview(frame[0]).cast("B")))
        if len(out) >= count:
            break
    return list(out[:count])


def check(name, bits, values, expected_bits, expected_type, sample_type="int", shift=0):
    path = os.path.join(tempfile.mkdtemp(), name + ".wav")
    write_wav(path, bits, values, sample_type)
    expected = [v << shift for v in values] if shift else list(values)
    node = core.lsmas.LWLibavAudioSource(source=path, cache=0, indexingpr=0)
    errors = []
    if node.bits_per_sample != expected_bits:
        errors.append(f"bits_per_sample: got {node.bits_per_sample}, want {expected_bits}")
    if node.sample_type != expected_type:
        errors.append(f"sample_type: got {node.sample_type}, want {expected_type}")
    if node.num_samples < len(values):
        errors.append(f"num_samples: got {node.num_samples}, want at least {len(values)}")
    else:
        got = read_samples(node, len(expected))
        if got != expected:
            bad = next(i for i, (g, w) in enumerate(zip(got, expected)) if g != w)
            errors.append(f"sample {bad}: got {got[bad]}, want {expected[bad]}")
    if errors:
        print(f"FAIL {name}: " + "; ".join(errors))
        return False
    print(f"ok   {name}: {node.bits_per_sample}bit {node.sample_type.name}")
    return True


# Every value is exactly representable once scaled to float32, which is what the
# resampler uses internally, so the round trip has to be bit exact.
PATTERN_24 = [0, 1, -1, 0x7FFFFF, -0x800000, 0x123456, -0x123456, 0x7F, -0x80]
PATTERN_16 = [0, 1, -1, 0x7FFF, -0x8000, 0x1234, -0x1234, 0x7F, -0x80]
PATTERN_32 = [0, 1, -1, 0x7FFFFF00, -0x7FFFFF00, 0x12345600, -0x12345600, 0x7F, -0x80]

results = [
    check("pcm24", 24, PATTERN_24 * 64, 24, vs.INTEGER, shift=8),
    check("pcm16", 16, PATTERN_16 * 64, 16, vs.INTEGER),
    check("pcm32", 32, PATTERN_32 * 64, 32, vs.INTEGER),
    check("flt32", 32, [0.0, 0.5, -0.5, 1.0, -1.0, 0.25] * 64, 32, vs.FLOAT, "float"),
]
sys.exit(0 if all(results) else 1)
