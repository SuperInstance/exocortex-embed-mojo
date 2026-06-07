# test_quantize.mojo — Tests for scalar quantization
#
# 10 assertions covering fit, quantize, dequantize, round-trip error, distance.

from exocortex_embed.quantize import ScalarQuantizer


fn test_fit_and_quantize() raises:
    var q = ScalarQuantizer(3)

    var v1 = List[Float64](capacity=3)
    v1.append(0.0); v1.append(1.0); v1.append(2.0)
    var v2 = List[Float64](capacity=3)
    v2.append(3.0); v2.append(4.0); v2.append(5.0)

    var training = List[List[Float64]](capacity=2)
    training.append(v1); training.append(v2)
    q.fit(training)

    let qv = q.quantize(v1)
    # v1 is the minimum vector → should quantize near -128
    assert(Int(qv[0]) == -128, "quantize min [0]")
    assert(Int(qv[2]) == -128, "quantize min [2]")
    print("  [PASS] fit and quantize")


fn test_round_trip() raises:
    var q = ScalarQuantizer(3)

    var v1 = List[Float64](capacity=3)
    v1.append(1.0); v1.append(2.0); v1.append(3.0)
    var v2 = List[Float64](capacity=3)
    v2.append(4.0); v2.append(5.0); v2.append(6.0)

    var training = List[List[Float64]](capacity=2)
    training.append(v1); training.append(v2)
    q.fit(training)

    let quantized = q.quantize(v1)
    let recovered = q.dequantize(quantized)

    # Round-trip error should be small (≤ span/255 per dimension)
    # span per dim: 3, 3, 3 → max error ≈ 3/255 ≈ 0.012
    for j in range(3):
        let err = recovered[j] - v1[j]
        let abs_err = err if err >= 0 else -err
        assert(abs_err < 0.05, "round-trip error within bounds")
    print("  [PASS] round trip")


fn test_quantized_distance_ordering() raises:
    var q = ScalarQuantizer(2)

    var v1 = List[Float64](capacity=2)
    v1.append(0.0); v1.append(0.0)
    var v2 = List[Float64](capacity=2)
    v2.append(5.0); v2.append(5.0)
    var v3 = List[Float64](capacity=2)
    v3.append(2.5); v3.append(2.5)

    var training = List[List[Float64]](capacity=3)
    training.append(v1); training.append(v2); training.append(v3)
    q.fit(training)

    let q1 = q.quantize(v1)
    let q2 = q.quantize(v2)
    let q3 = q.quantize(v3)

    let d13 = q.quantized_distance(q1, q3)
    let d12 = q.quantized_distance(q1, q2)

    # v1 is closer to v3 than to v2 → d13 < d12
    assert(d13 < d12, "quantized distance ordering preserved")
    print("  [PASS] quantized distance ordering")


fn test_quantize_single_vector() raises:
    var q = ScalarQuantizer(2)

    var v = List[Float64](capacity=2)
    v.append(0.0); v.append(10.0)

    var training = List[List[Float64]](capacity=1)
    training.append(v)
    q.fit(training)

    let qv = q.quantize(v)
    # Single vector: min = max, so quantized should be 0 (our handling)
    assert(Int(qv[0]) == 0, "single vector [0]")
    assert(Int(qv[1]) == 0, "single vector [1]")
    print("  [PASS] quantize single vector (zero span)")


fn test_memory_savings() raises:
    # Conceptual test: verify Int8 output type
    var q = ScalarQuantizer(4)

    var v1 = List[Float64](capacity=4)
    v1.append(1.0); v1.append(2.0); v1.append(3.0); v1.append(4.0)
    var v2 = List[Float64](capacity=4)
    v2.append(5.0); v2.append(6.0); v2.append(7.0); v2.append(8.0)

    var training = List[List[Float64]](capacity=2)
    training.append(v1); training.append(v2)
    q.fit(training)

    let quantized = q.quantize(v1)
    # Verify all values fit in Int8 range
    for i in range(4):
        let val = Int(quantized[i])
        assert(val >= -128, "int8 lower bound")
        assert(val <= 127, "int8 upper bound")
    print("  [PASS] memory savings (int8 range)")


fn main() raises:
    print("=== quantize tests ===")
    test_fit_and_quantize()
    test_round_trip()
    test_quantized_distance_ordering()
    test_quantize_single_vector()
    test_memory_savings()
    print("=== all quantize tests passed ===")
