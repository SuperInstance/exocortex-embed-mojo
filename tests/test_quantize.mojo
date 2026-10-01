# test_quantize.mojo — Tests for scalar quantization
#
# Ported to Mojo 1.2.0-dev (2026-10-01 nightly). Mechanical drift only:
#   - `fn` -> `def`; `let` -> `var`
#   - builtin `assert(cond, msg)` uncallable in Mojo 1.2 -> fleet
#     self-reporting `Counters` pattern, 1:1 with original asserts.
#   - List is not copyable -> `training.append(copy_vec(v))` helper replaces
#     the original `training.append(v)` (which would move/misuse the var the
#     test still needs later). Element values identical.
#
# 16 assertions covering fit, quantize, dequantize, round-trip error, distance.

from exocortex_embed.quantize import ScalarQuantizer


struct Counters(ImplicitlyCopyable):
    var total: Int
    var pass_count: Int
    var fail_count: Int

    def __init__(out self):
        self.total = 0
        self.pass_count = 0
        self.fail_count = 0

    def record(mut self, cond: Bool, test_name: String):
        self.total += 1
        if cond:
            self.pass_count += 1
            print("  PASS: ", test_name)
        else:
            self.fail_count += 1
            print("  FAIL: ", test_name)


def copy_vec(values: List[Float64]) -> List[Float64]:
    var result = List[Float64](capacity=len(values))
    for i in range(len(values)):
        result.append(values[i])
    return result^


def test_fit_and_quantize(mut c: Counters):
    var q = ScalarQuantizer(3)

    var v1 = List[Float64](capacity=3)
    v1.append(0.0); v1.append(1.0); v1.append(2.0)
    var v2 = List[Float64](capacity=3)
    v2.append(3.0); v2.append(4.0); v2.append(5.0)

    var training = List[List[Float64]](capacity=2)
    training.append(copy_vec(v1)); training.append(copy_vec(v2))
    q.fit(training)

    var qv = q.quantize(v1)
    # v1 is the minimum vector → should quantize near -128
    c.record(Int(qv[0]) == -128, "quantize min [0]")
    c.record(Int(qv[2]) == -128, "quantize min [2]")


def test_round_trip(mut c: Counters):
    var q = ScalarQuantizer(3)

    var v1 = List[Float64](capacity=3)
    v1.append(1.0); v1.append(2.0); v1.append(3.0)
    var v2 = List[Float64](capacity=3)
    v2.append(4.0); v2.append(5.0); v2.append(6.0)

    var training = List[List[Float64]](capacity=2)
    training.append(copy_vec(v1)); training.append(copy_vec(v2))
    q.fit(training)

    var quantized = q.quantize(v1)
    var recovered = q.dequantize(quantized)

    # Round-trip error should be small (≤ span/255 per dimension)
    # span per dim: 3, 3, 3 → max error ≈ 3/255 ≈ 0.012
    for j in range(3):
        var err = recovered[j] - v1[j]
        var abs_err = err if err >= 0 else -err
        c.record(abs_err < 0.05, "round-trip error within bounds [" + String(j) + "]")


def test_quantized_distance_ordering(mut c: Counters):
    var q = ScalarQuantizer(2)

    var v1 = List[Float64](capacity=2)
    v1.append(0.0); v1.append(0.0)
    var v2 = List[Float64](capacity=2)
    v2.append(5.0); v2.append(5.0)
    var v3 = List[Float64](capacity=2)
    v3.append(2.5); v3.append(2.5)

    var training = List[List[Float64]](capacity=3)
    training.append(copy_vec(v1)); training.append(copy_vec(v2)); training.append(copy_vec(v3))
    q.fit(training)

    var q1 = q.quantize(v1)
    var q2 = q.quantize(v2)
    var q3 = q.quantize(v3)

    var d13 = q.quantized_distance(q1, q3)
    var d12 = q.quantized_distance(q1, q2)

    # v1 is closer to v3 than to v2 → d13 < d12
    c.record(d13 < d12, "quantized distance ordering preserved")


def test_quantize_single_vector(mut c: Counters):
    var q = ScalarQuantizer(2)

    var v = List[Float64](capacity=2)
    v.append(0.0); v.append(10.0)

    var training = List[List[Float64]](capacity=1)
    training.append(copy_vec(v))
    q.fit(training)

    var qv = q.quantize(v)
    # Single vector: min = max, so quantized should be 0 (our handling)
    c.record(Int(qv[0]) == 0, "single vector [0]")
    c.record(Int(qv[1]) == 0, "single vector [1]")


def test_memory_savings(mut c: Counters):
    # Conceptual test: verify Int8 output type
    var q = ScalarQuantizer(4)

    var v1 = List[Float64](capacity=4)
    v1.append(1.0); v1.append(2.0); v1.append(3.0); v1.append(4.0)
    var v2 = List[Float64](capacity=4)
    v2.append(5.0); v2.append(6.0); v2.append(7.0); v2.append(8.0)

    var training = List[List[Float64]](capacity=2)
    training.append(copy_vec(v1)); training.append(copy_vec(v2))
    q.fit(training)

    var quantized = q.quantize(v1)
    # Verify all values fit in Int8 range
    for i in range(4):
        var val = Int(quantized[i])
        c.record(val >= -128, "int8 lower bound [" + String(i) + "]")
        c.record(val <= 127, "int8 upper bound [" + String(i) + "]")


def main() raises:
    print("=== quantize tests ===")
    var c = Counters()
    test_fit_and_quantize(c)
    test_round_trip(c)
    test_quantized_distance_ordering(c)
    test_quantize_single_vector(c)
    test_memory_savings(c)
    print("Results: ", c.pass_count, " passed, ", c.fail_count, " failed")
    if c.fail_count == 0:
        print("ALL TESTS PASSED")
    else:
        print("TESTS FAILED")
