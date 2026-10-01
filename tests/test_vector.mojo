# test_vector.mojo — Tests for vector operations
#
# Ported to Mojo 1.2.0-dev (2026-10-01 nightly). Mechanical drift only:
#   - `fn` -> `def`; `let` -> `var`
#   - builtin `assert(cond, msg)` is uncallable in Mojo 1.2 (args coerce to a
#     single Tuple[Bool, String]) -> fleet self-reporting `Counters` pattern,
#     1:1 with the original asserts. Originals preserved in git history.
#
# 15 assertions covering dot, norm, cosine (3 cases), euclidean, add, scale, normalize.

from exocortex_embed.vector import (
    dot, norm, cosine_similarity, euclidean_distance,
    add, scale, normalize,
)


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


def test_dot_product(mut c: Counters):
    var a = List[Float64](capacity=3)
    a.append(1.0); a.append(2.0); a.append(3.0)
    var b = List[Float64](capacity=3)
    b.append(4.0); b.append(5.0); b.append(6.0)

    var result = dot(a, b)
    # 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
    c.record(result == 32.0, "dot product")


def test_norm(mut c: Counters):
    var v = List[Float64](capacity=3)
    v.append(3.0); v.append(4.0); v.append(0.0)

    var result = norm(v)
    # sqrt(9 + 16 + 0) = 5.0
    c.record(result == 5.0, "norm")


def test_cosine_orthogonal(mut c: Counters):
    var a = List[Float64](capacity=2)
    a.append(1.0); a.append(0.0)
    var b = List[Float64](capacity=2)
    b.append(0.0); b.append(1.0)

    var result = cosine_similarity(a, b)
    # Orthogonal → cosine = 0
    c.record(result == 0.0, "cosine orthogonal")


def test_cosine_identical(mut c: Counters):
    var a = List[Float64](capacity=3)
    a.append(1.0); a.append(2.0); a.append(3.0)

    var result = cosine_similarity(a, a)
    # Self-similarity = 1.0
    c.record(result == 1.0, "cosine identical")


def test_cosine_opposite(mut c: Counters):
    var a = List[Float64](capacity=2)
    a.append(1.0); a.append(0.0)
    var b = List[Float64](capacity=2)
    b.append(-1.0); b.append(0.0)

    var result = cosine_similarity(a, b)
    # Opposite → cosine = -1.0
    c.record(result == -1.0, "cosine opposite")


def test_euclidean(mut c: Counters):
    var a = List[Float64](capacity=2)
    a.append(0.0); a.append(0.0)
    var b = List[Float64](capacity=2)
    b.append(3.0); b.append(4.0)

    var result = euclidean_distance(a, b)
    c.record(result == 5.0, "euclidean distance")


def test_add(mut c: Counters):
    var a = List[Float64](capacity=3)
    a.append(1.0); a.append(2.0); a.append(3.0)
    var b = List[Float64](capacity=3)
    b.append(10.0); b.append(20.0); b.append(30.0)

    var result = add(a, b)
    c.record(result[0] == 11.0, "add[0]")
    c.record(result[1] == 22.0, "add[1]")
    c.record(result[2] == 33.0, "add[2]")


def test_scale(mut c: Counters):
    var v = List[Float64](capacity=3)
    v.append(1.0); v.append(2.0); v.append(3.0)

    var result = scale(v, 2.0)
    c.record(result[0] == 2.0, "scale[0]")
    c.record(result[1] == 4.0, "scale[1]")
    c.record(result[2] == 6.0, "scale[2]")


def test_normalize(mut c: Counters):
    var v = List[Float64](capacity=2)
    v.append(3.0); v.append(4.0)

    var result = normalize(v)
    # [3/5, 4/5]
    c.record(result[0] == 0.6, "normalize[0]")
    c.record(result[1] == 0.8, "normalize[1]")

    # Norm of normalized vector should be 1.0
    var n = norm(result)
    c.record(n == 1.0, "normalize: norm == 1")


def main() raises:
    print("=== vector tests ===")
    var c = Counters()
    test_dot_product(c)
    test_norm(c)
    test_cosine_orthogonal(c)
    test_cosine_identical(c)
    test_cosine_opposite(c)
    test_euclidean(c)
    test_add(c)
    test_scale(c)
    test_normalize(c)
    print("Results: ", c.pass_count, " passed, ", c.fail_count, " failed")
    if c.fail_count == 0:
        print("ALL TESTS PASSED")
    else:
        print("TESTS FAILED")
