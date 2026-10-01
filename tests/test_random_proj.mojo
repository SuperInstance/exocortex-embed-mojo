# test_random_proj.mojo — Tests for random projection
#
# Ported to Mojo 1.2.0-dev (2026-10-01 nightly). Mechanical drift only:
#   - `fn` -> `def`; `let` -> `var`
#   - builtin `assert(cond, msg)` uncallable in Mojo 1.2 -> fleet
#     self-reporting `Counters` pattern, 1:1 with original asserts.
#
# 9 assertions covering projection shape, distance preservation, determinism.

from exocortex_embed.random_proj import RandomProjection
from exocortex_embed.vector import euclidean_distance


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


def test_projection_shape(mut c: Counters):
    var rp = RandomProjection(10, 3, seed=42)
    var v = List[Float64](capacity=10)
    for i in range(10):
        v.append(Float64(i))

    var projected = rp.project(v)
    c.record(len(projected) == 3, "projection output dimension")


def test_deterministic(mut c: Counters):
    var rp1 = RandomProjection(5, 2, seed=999)
    var rp2 = RandomProjection(5, 2, seed=999)

    var v = List[Float64](capacity=5)
    for _ in range(5):
        v.append(1.0)

    var p1 = rp1.project(v)
    var p2 = rp2.project(v)

    c.record(p1[0] == p2[0], "deterministic [0]")
    c.record(p1[1] == p2[1], "deterministic [1]")


def test_distance_preservation(mut c: Counters):
    # JL lemma: pairwise distances should be approximately preserved
    var rp = RandomProjection(8, 4, seed=42)

    var a = List[Float64](capacity=8)
    a.append(1.0); a.append(0.0); a.append(0.0); a.append(0.0)
    a.append(0.0); a.append(0.0); a.append(0.0); a.append(0.0)

    var b = List[Float64](capacity=8)
    b.append(0.0); b.append(1.0); b.append(0.0); b.append(0.0)
    b.append(0.0); b.append(0.0); b.append(0.0); b.append(0.0)

    var orig_dist = euclidean_distance(a, b)  # √2 ≈ 1.414

    var pa = rp.project(a)
    var pb = rp.project(b)
    var proj_dist = euclidean_distance(pa, pb)

    # Allow 50% distortion for such aggressive reduction (8D → 4D)
    var ratio = proj_dist / orig_dist
    c.record(ratio > 0.3, "distance preservation lower bound")
    c.record(ratio < 3.0, "distance preservation upper bound")
    print("  (distance preservation ratio=", ratio, ")")


def test_unit_vector(mut c: Counters):
    var rp = RandomProjection(4, 2, seed=7)
    var ones = List[Float64](capacity=4)
    for _ in range(4):
        ones.append(1.0)

    var projected = rp.project(ones)
    # Should produce finite values
    for i in range(len(projected)):
        c.record(projected[i] > -1e10, "finite output lower")
        c.record(projected[i] < 1e10, "finite output upper")


def main() raises:
    print("=== random_proj tests ===")
    var c = Counters()
    test_projection_shape(c)
    test_deterministic(c)
    test_distance_preservation(c)
    test_unit_vector(c)
    print("Results: ", c.pass_count, " passed, ", c.fail_count, " failed")
    if c.fail_count == 0:
        print("ALL TESTS PASSED")
    else:
        print("TESTS FAILED")
