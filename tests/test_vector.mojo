# test_vector.mojo — Tests for vector operations
#
# 12 assertions covering dot, norm, cosine, euclidean, add, scale, normalize.

from exocortex_embed.vector import (
    dot, norm, cosine_similarity, euclidean_distance,
    add, scale, normalize,
)


fn test_dot_product() raises:
    var a = List[Float64](capacity=3)
    a.append(1.0); a.append(2.0); a.append(3.0)
    var b = List[Float64](capacity=3)
    b.append(4.0); b.append(5.0); b.append(6.0)

    let result = dot(a, b)
    # 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
    assert(result == 32.0, "dot product failed")
    print("  [PASS] dot product")


fn test_norm() raises:
    var v = List[Float64](capacity=3)
    v.append(3.0); v.append(4.0); v.append(0.0)

    let result = norm(v)
    # sqrt(9 + 16 + 0) = 5.0
    assert(result == 5.0, "norm failed")
    print("  [PASS] norm")


fn test_cosine_orthogonal() raises:
    var a = List[Float64](capacity=2)
    a.append(1.0); a.append(0.0)
    var b = List[Float64](capacity=2)
    b.append(0.0); b.append(1.0)

    let result = cosine_similarity(a, b)
    # Orthogonal → cosine = 0
    assert(result == 0.0, "cosine orthogonal failed")
    print("  [PASS] cosine orthogonal")


fn test_cosine_identical() raises:
    var a = List[Float64](capacity=3)
    a.append(1.0); a.append(2.0); a.append(3.0)

    let result = cosine_similarity(a, a)
    # Self-similarity = 1.0
    assert(result == 1.0, "cosine self failed")
    print("  [PASS] cosine identical")


fn test_cosine_opposite() raises:
    var a = List[Float64](capacity=2)
    a.append(1.0); a.append(0.0)
    var b = List[Float64](capacity=2)
    b.append(-1.0); b.append(0.0)

    let result = cosine_similarity(a, b)
    # Opposite → cosine = -1.0
    assert(result == -1.0, "cosine opposite failed")
    print("  [PASS] cosine opposite")


fn test_euclidean() raises:
    var a = List[Float64](capacity=2)
    a.append(0.0); a.append(0.0)
    var b = List[Float64](capacity=2)
    b.append(3.0); b.append(4.0)

    let result = euclidean_distance(a, b)
    assert(result == 5.0, "euclidean distance failed")
    print("  [PASS] euclidean distance")


fn test_add() raises:
    var a = List[Float64](capacity=3)
    a.append(1.0); a.append(2.0); a.append(3.0)
    var b = List[Float64](capacity=3)
    b.append(10.0); b.append(20.0); b.append(30.0)

    let result = add(a, b)
    assert(result[0] == 11.0, "add[0] failed")
    assert(result[1] == 22.0, "add[1] failed")
    assert(result[2] == 33.0, "add[2] failed")
    print("  [PASS] add")


fn test_scale() raises:
    var v = List[Float64](capacity=3)
    v.append(1.0); v.append(2.0); v.append(3.0)

    let result = scale(v, 2.0)
    assert(result[0] == 2.0, "scale[0] failed")
    assert(result[1] == 4.0, "scale[1] failed")
    assert(result[2] == 6.0, "scale[2] failed")
    print("  [PASS] scale")


fn test_normalize() raises:
    var v = List[Float64](capacity=2)
    v.append(3.0); v.append(4.0)

    let result = normalize(v)
    # [3/5, 4/5]
    assert(result[0] == 0.6, "normalize[0] failed")
    assert(result[1] == 0.8, "normalize[1] failed")

    # Norm of normalized vector should be 1.0
    let n = norm(result)
    assert(n == 1.0, "normalize: norm != 1")
    print("  [PASS] normalize")


fn main() raises:
    print("=== vector tests ===")
    test_dot_product()
    test_norm()
    test_cosine_orthogonal()
    test_cosine_identical()
    test_cosine_opposite()
    test_euclidean()
    test_add()
    test_scale()
    test_normalize()
    print("=== all vector tests passed ===")
