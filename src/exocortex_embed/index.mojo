# index.mojo — Brute-force vector index with top-k cosine similarity search
#
# Honest about complexity: O(n×d) per query for n vectors of dimension d.
# For the exocortex's small-to-medium collections (hundreds to low thousands),
# brute force is faster than ANN indexing due to:
#   1. No index build time
#   2. No hyperparameter tuning (ef, M, num_trees, etc.)
#   3. SIMD-friendly linear scan (cache-coherent, branch-predictable)
#   4. Exact results — no approximation errors
#
# When the collection grows past ~10K vectors, switch to HNSW or IVF.
# "If brute force is fast enough, it's the best algorithm." — practical wisdom

from .vector import cosine_similarity


## Internal representation: id + vector pair
struct Entry:
    var id: String
    var vector: List[Float64]

    fn __init__(inout self, owned id: String, owned vector: List[Float64]):
        self.id = id^
        self.vector = vector^


## ---------------------------------------------------------------------------
## VectorIndex — Brute-force cosine similarity index
## ---------------------------------------------------------------------------
struct VectorIndex:
    var entries: List[Entry]
    var dim: Int

    fn __init__(inout self, dim: Int):
        self.dim = dim
        self.entries = List[Entry]()

    ## Add a vector to the index.
    fn add(inout self, owned id: String, inout vector: List[Float64]):
        # Copy the vector into the index
        var v = List[Float64](capacity=len(vector))
        for i in range(len(vector)):
            v.append(vector[i])
        self.entries.append(Entry(id^, v^))

    ## Search for top-k vectors by cosine similarity.
    ## Returns List of (id, score) tuples sorted descending by similarity.
    fn search(inout self, inout query: List[Float64], k: Int) -> List[Tuple[String, Float64]]:
        var scores = List[Tuple[String, Float64]](capacity=len(self.entries))

        for i in range(len(self.entries)):
            let sim = cosine_similarity(query, self.entries[i].vector)
            scores.append(Tuple(self.entries[i].id, sim))

        # Simple selection sort for top-k (fine for small k)
        var result = List[Tuple[String, Float64]](capacity=k)
        var used = List[Bool](capacity=len(scores))
        for i in range(len(scores)):
            used.append(False)

        for _ in range(min(k, len(scores))):
            var best_idx: Int = -1
            var best_score: Float64 = -2.0
            for i in range(len(scores)):
                if not used[i] and scores[i].get[1]() > best_score:
                    best_score = scores[i].get[1]()
                    best_idx = i
            if best_idx >= 0:
                used[best_idx] = True
                result.append(scores[best_idx])

        return result

    ## Remove a vector by id.
    fn remove(inout self, id: String):
        var new_entries = List[Entry](capacity=len(self.entries))
        for i in range(len(self.entries)):
            if self.entries[i].id != id:
                new_entries.append(self.entries[i])
        self.entries = new_entries^

    ## Return the number of vectors in the index.
    fn size(inout self) -> Int:
        return len(self.entries)
