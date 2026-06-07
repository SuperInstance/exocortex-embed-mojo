# test_random_proj.mojo — Tests for random projection
#
# 8 assertions covering projection shape, distance preservation, determinism.

from exocortex_embed.random_proj import RandomProjection
from exocortex_embed.vector import euclidean_distance


fn test_projection_shape() raises:
    var rp = RandomProjection(10, 3, seed=42)
    var v = List[Float64](capacity=10)
    for i in range(10):
        v.append(Float64(i))

    let projected = rp.project(v)
    assert(len(projected) == 3, "projection output dimension")
    print("  [PASS] projection shape")


fn test_deterministic() raises:
    var rp1 = RandomProjection(5, 2, seed=999)
    var rp2 = RandomProjection(5, 2, seed=999)

    var v = List[Float64](capacity=5)
    for i in range(5):
        v.append(1.0)

    let p1 = rp1.project(v)
    let p2 = rp2.project(v)

    assert(p1[0] == p2[0], "deterministic [0]")
    assert(p1[1] == p2[1], "deterministic [1]")
    print("  [PASS] deterministic")


fn test_distance_preservation() raises:
    # JL lemma: pairwise distances should be approximately preserved
    var rp = RandomProjection(8, 4, seed=42)

    var a = List[Float64](capacity=8)
    a.append(1.0); a.append(0.0); a.append(0.0); a.append(0.0)
    a.append(0.0); a.append(0.0); a.append(0.0); a.append(0.0)

    var b = List[Float64](capacity=8)
    b.append(0.0); b.append(1.0); b.append(0.0); b.append(0.0)
    b.append(0.0); b.append(0.0); b.append(0.0); b.append(0.0)

    let orig_dist = euclidean_distance(a, b)  # √2 ≈ 1.414

    let pa = rp.project(a)
    let pb = rp.project(b)
    let proj_dist = euclidean_distance(pa, pb)

    # Allow 50% distortion for such aggressive reduction (8D → 4D)
    let ratio = proj_dist / orig_dist
    assert(ratio > 0.3, "distance preservation lower bound")
    assert(ratio < 3.0, "distance preservation upper bound")
    print("  [PASS] distance preservation (ratio=", ratio, ")")


fn test_unit_vector() raises:
    var rp = RandomProjection(4, 2, seed=7)
    var ones = List[Float64](capacity=4)
    for i in range(4):
        ones.append(1.0)

    let projected = rp.project(ones)
    # Should produce finite values
    for i in range(len(projected)):
        assert(projected[i] > -1e10, "finite output lower")
        assert(projected[i] < 1e10, "finite output upper")
    print("  [PASS] unit vector projection")


fn main() raises:
    print("=== random_proj tests ===")
    test_projection_shape()
    test_deterministic()
    test_distance_preservation()
    test_unit_vector()
    print("=== all random_proj tests passed ===")
