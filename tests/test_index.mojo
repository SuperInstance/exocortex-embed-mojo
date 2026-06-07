# test_index.mojo — Tests for vector index
#
# 10 assertions covering add, search, remove, size.

from exocortex_embed.index import VectorIndex


fn make_vec(values: List[Float64]) -> List[Float64]:
    var result = List[Float64](capacity=len(values))
    for i in range(len(values)):
        result.append(values[i])
    return result


fn test_add_and_size() raises:
    var index = VectorIndex(3)
    assert(index.size() == 0, "initial size")

    var v = List[Float64](capacity=3)
    v.append(1.0); v.append(0.0); v.append(0.0)
    index.add("a", v)
    assert(index.size() == 1, "size after add")

    var v2 = List[Float64](capacity=3)
    v2.append(0.0); v2.append(1.0); v2.append(0.0)
    index.add("b", v2)
    assert(index.size() == 2, "size after second add")
    print("  [PASS] add and size")


fn test_search_top_k() raises:
    var index = VectorIndex(3)

    var v1 = List[Float64](capacity=3)
    v1.append(1.0); v1.append(0.0); v1.append(0.0)
    var v2 = List[Float64](capacity=3)
    v2.append(0.0); v2.append(1.0); v2.append(0.0)
    var v3 = List[Float64](capacity=3)
    v3.append(0.0); v3.append(0.0); v3.append(1.0)

    index.add("x", v1)
    index.add("y", v2)
    index.add("z", v3)

    var query = List[Float64](capacity=3)
    query.append(0.9); query.append(0.1); query.append(0.0)

    let results = index.search(query, 2)
    assert(len(results) == 2, "search returns k results")
    # "x" should be top result (closest to query direction)
    assert(results[0].get[0]() == "x", "top result is x")
    print("  [PASS] search top-k")


fn test_search_empty() raises:
    var index = VectorIndex(3)
    var query = List[Float64](capacity=3)
    query.append(1.0); query.append(0.0); query.append(0.0)

    let results = index.search(query, 5)
    assert(len(results) == 0, "empty index returns empty")
    print("  [PASS] search empty")


fn test_remove() raises:
    var index = VectorIndex(2)

    var v1 = List[Float64](capacity=2)
    v1.append(1.0); v1.append(0.0)
    var v2 = List[Float64](capacity=2)
    v2.append(0.0); v1.append(1.0)

    index.add("a", v1)
    index.add("b", v2)
    assert(index.size() == 2, "size before remove")

    index.remove("a")
    assert(index.size() == 1, "size after remove")

    # Search should only find "b"
    var query = List[Float64](capacity=2)
    query.append(1.0); query.append(0.0)
    let results = index.search(query, 5)
    assert(len(results) == 1, "search after remove")
    assert(results[0].get[0]() == "b", "remaining is b")
    print("  [PASS] remove")


fn test_search_ordering() raises:
    var index = VectorIndex(2)

    var v1 = List[Float64](capacity=2)
    v1.append(1.0); v1.append(0.0)
    var v2 = List[Float64](capacity=2)
    v2.append(0.8); v2.append(0.6)
    var v3 = List[Float64](capacity=2)
    v3.append(0.0); v3.append(1.0)

    index.add("right", v1)
    index.add("diag", v2)
    index.add("up", v3)

    var query = List[Float64](capacity=2)
    query.append(1.0); query.append(0.0)

    let results = index.search(query, 3)
    # Should be ordered: right > diag > up
    assert(results[0].get[0]() == "right", "first is right")
    assert(results[1].get[0]() == "diag", "second is diag")
    assert(results[2].get[0]() == "up", "third is up")
    print("  [PASS] search ordering")


fn main() raises:
    print("=== index tests ===")
    test_add_and_size()
    test_search_top_k()
    test_search_empty()
    test_remove()
    test_search_ordering()
    print("=== all index tests passed ===")
