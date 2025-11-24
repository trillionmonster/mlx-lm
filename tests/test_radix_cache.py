
import unittest
import mlx.core as mx
import time
import random
from mlx_lm.radix_cache import RadixCache

class TestRadixCache(unittest.TestCase):
    def setUp(self):
        self.cache = RadixCache()

    def _create_kv(self, length, dim=8):
        # Create a dummy KV cache of shape (1, 1, length, dim)
        k = mx.random.uniform(shape=(1, 1, length, dim))
        v = mx.random.uniform(shape=(1, 1, length, dim))
        return [(k, v)]

    def test_basic_insert_and_match(self):
        print("\nTesting Basic Insert and Match...")
        tokens = [1, 2, 3, 4]
        # Mock KV cache: List of (K, V) tuples. 
        # Shape: (Batch=1, Heads=1, SeqLen=4, Dim=8)
        kv = [(mx.zeros((1, 1, 4, 8)), mx.zeros((1, 1, 4, 8)))]
        
        # Use _insert_impl for synchronous execution in tests
        self.cache._insert_impl(tokens, kv)
        
        # Test Full Match
        matched_tokens, cached_kv, node = self.cache.match_prefix(tokens)
        self.assertEqual(matched_tokens, tokens)
        self.assertIsNotNone(cached_kv)
        print("Full match passed.")

        # Test Partial Match
        query_tokens = [1, 2, 5, 6]
        matched_tokens, cached_kv, node = self.cache.match_prefix(query_tokens)
        
        # Now we support partial matching via slicing!
        self.assertEqual(matched_tokens, [1, 2])
        self.assertIsNotNone(cached_kv)
        
        # Verify sliced cache shape
        # Original cache was length 4. Matched length 2.
        # cached_kv[0][0] should have shape (..., 2, ...)
        k, v = cached_kv[0]
        self.assertEqual(k.shape[-2], 2)
        
        print("Partial match (with slicing) passed.")

    def test_split_node(self):
        print("\nTesting Node Splitting...")
        # Insert A: [1, 2, 3, 4]
        tokens_a = [1, 2, 3, 4]
        kv_a = [(mx.full((1, 1, 4, 8), 1.0), mx.full((1, 1, 4, 8), 1.0))]
        self.cache._insert_impl(tokens_a, kv_a)
        
        # Insert B: [1, 2, 5, 6] -> Should split at [1, 2]
        tokens_b = [1, 2, 5, 6]
        kv_b = [(mx.full((1, 1, 4, 8), 2.0), mx.full((1, 1, 4, 8), 2.0))]
        self.cache._insert_impl(tokens_b, kv_b)
        
        # Now structure should be:
        # Root -> Node([1, 2]) -> Node([3, 4])
        #                      -> Node([5, 6])
        
        # Verify Match [1, 2]
        matched_tokens, cached_kv, node = self.cache.match_prefix([1, 2])
        self.assertEqual(matched_tokens, [1, 2])
        self.assertIsNotNone(cached_kv)
        # Check KV content size. Should be length 2.
        k, v = cached_kv[0]
        self.assertEqual(k.shape[-2], 2)
        print("Split verification: Match [1, 2] passed.")
        
        # Verify Match [1, 2, 3, 4]
        matched_tokens, _, _ = self.cache.match_prefix([1, 2, 3, 4])
        self.assertEqual(matched_tokens, [1, 2, 3, 4])
        print("Split verification: Match [1, 2, 3, 4] passed.")

        # Verify Match [1, 2, 5, 6]
        matched_tokens, _, _ = self.cache.match_prefix([1, 2, 5, 6])
        self.assertEqual(matched_tokens, [1, 2, 5, 6])
        print("Split verification: Match [1, 2, 5, 6] passed.")

    def test_eviction(self):
        print("\nTesting Eviction...")
        # Insert 3 sequences
        # 1. [1, 1]
        self.cache._insert_impl([1, 1], [(mx.zeros((1, 2, 8)), mx.zeros((1, 2, 8)))])
        # 2. [2, 2]
        self.cache._insert_impl([2, 2], [(mx.zeros((1, 2, 8)), mx.zeros((1, 2, 8)))])
        # 3. [3, 3]
        self.cache._insert_impl([3, 3], [(mx.zeros((1, 2, 8)), mx.zeros((1, 2, 8)))])
        
        # Access [1, 1] and [3, 3] to make [2, 2] the LRU
        self.cache.match_prefix([1, 1])
        self.cache.match_prefix([3, 3])
    
        # Evict 1 token (should remove a whole node if it's a leaf)
        # [2, 2] is LRU.
        self.cache.evict(1)
    
        # Check if [2, 2] is gone
        matched, _, _ = self.cache.match_prefix([2, 2])
        self.assertEqual(matched, [])
    
        # Check if [1, 1] and [3, 3] are still there
        matched, _, _ = self.cache.match_prefix([1, 1])
        self.assertEqual(matched, [1, 1])
        matched, _, _ = self.cache.match_prefix([3, 3])
        self.assertEqual(matched, [3, 3])
        print("Eviction passed.")

    def test_complex_splitting(self):
        print("\n=== Test: Complex Splitting ===")
        # 1. Insert [1, 2, 3, 4, 5]
        tokens1 = [1, 2, 3, 4, 5]
        kv1 = self._create_kv(5)
        self.cache._insert_impl(tokens1, kv1)
        
        # 2. Insert [1, 2, 3, 6, 7] -> Split at 3
        tokens2 = [1, 2, 3, 6, 7]
        kv2 = self._create_kv(5)
        self.cache._insert_impl(tokens2, kv2)
        
        # Verify structure implicitly by matching
        # Match [1, 2, 3] -> Should return full match from the split node
        m, kv, node = self.cache.match_prefix([1, 2, 3])
        self.assertEqual(m, [1, 2, 3])
        self.assertEqual(kv[0][0].shape[-2], 3)
        
        # Match [1, 2, 3, 4] -> Should match child 1
        m, kv, node = self.cache.match_prefix([1, 2, 3, 4])
        self.assertEqual(m, [1, 2, 3, 4])
        
        # Match [1, 2, 3, 6] -> Should match child 2
        m, kv, node = self.cache.match_prefix([1, 2, 3, 6])
        self.assertEqual(m, [1, 2, 3, 6])

        # 3. Insert [1, 2, 8, 9] -> Split at 2
        tokens3 = [1, 2, 8, 9]
        kv3 = self._create_kv(4)
        self.cache._insert_impl(tokens3, kv3)
        
        # Now structure:
        # Root -> [1, 2] -> [3] -> [4, 5]
        #                       -> [6, 7]
        #                -> [8, 9]
        
        # Verify [1, 2]
        m, kv, node = self.cache.match_prefix([1, 2])
        self.assertEqual(m, [1, 2])
        
        # Verify [1, 2, 8]
        m, kv, node = self.cache.match_prefix([1, 2, 8])
        self.assertEqual(m, [1, 2, 8])
        print("Complex splitting passed.")

    def test_large_dataset_stress(self):
        print("\n=== Test: Large Dataset Stress ===")
        # Generate 100 sequences with shared prefixes
        # Prefix A: [100, 101, ..., 119] (Length 20)
        # Prefix B: [200, 201, ..., 219] (Length 20)
        
        prefix_a = list(range(100, 120))
        prefix_b = list(range(200, 220))
        
        data = []
        
        # 50 sequences starting with A
        for i in range(50):
            suffix = [i, i+1, i+2]
            tokens = prefix_a + suffix
            kv = self._create_kv(len(tokens))
            data.append((tokens, kv))
            
        # 50 sequences starting with B
        for i in range(50):
            suffix = [i, i+1, i+2]
            tokens = prefix_b + suffix
            kv = self._create_kv(len(tokens))
            data.append((tokens, kv))
            
        # Shuffle and insert
        random.shuffle(data)
        start_time = time.time()
        for tokens, kv in data:
            self.cache._insert_impl(tokens, kv)
        print(f"Inserted 100 sequences in {time.time() - start_time:.4f}s")
        
        # Verify all
        for tokens, _ in data:
            m, kv, _ = self.cache.match_prefix(tokens)
            self.assertEqual(m, tokens)
            self.assertEqual(kv[0][0].shape[-2], len(tokens))
            
        print("Verification of 100 sequences passed.")
        
        # Test Eviction
        # Total tokens approx: 100 * 23 = 2300.
        # Let's evict 1000 tokens.
        initial_tokens = self.cache.total_tokens
        print(f"Total tokens before eviction: {initial_tokens}")
        
        self.cache.evict(1000)
        
        remaining_tokens = self.cache.total_tokens
        print(f"Total tokens after eviction: {remaining_tokens}")
        self.assertLess(remaining_tokens, initial_tokens)
        
        # Verify that we can still match something
        lost_count = 0
        for tokens, _ in data:
            m, _, _ = self.cache.match_prefix(tokens)
            if len(m) < len(tokens):
                lost_count += 1
        print(f"Lost {lost_count} sequences after eviction.")
        self.assertGreater(lost_count, 0)

    def test_partial_match_slicing_correctness(self):
        print("\n=== Test: Partial Match Slicing Correctness ===")
        # Insert [10, 20, 30, 40]
        tokens = [10, 20, 30, 40]
        # KV values: 0.0, 1.0, 2.0, 3.0
        k = mx.array([[[[0.0], [1.0], [2.0], [3.0]]]]) # (1, 1, 4, 1)
        v = mx.array([[[[0.0], [1.0], [2.0], [3.0]]]])
        kv = [(k, v)]
        
        self.cache._insert_impl(tokens, kv)
        
        # Match [10, 20]
        m, cached_kv, _ = self.cache.match_prefix([10, 20])
        self.assertEqual(m, [10, 20])
        
        # Check values
        k_slice, v_slice = cached_kv[0]
        self.assertEqual(k_slice.shape[-2], 2)
        
        # Verify content: should be 0.0 and 1.0
        # We need to evaluate to check values
        k_data = k_slice.flatten().tolist()
        self.assertEqual(k_data, [0.0, 1.0])
        print("Slicing values verified.")

    def test_capacity_limit_strictness(self):
        print("\n=== Test: Capacity Limit Strictness (1MB) ===")
        # 1MB capacity
        capacity = 1024 * 1024 
        self.cache = RadixCache(capacity_bytes=capacity)
        
        # Create a KV cache to calculate bytes per token
        # dim=128, float32 (4 bytes) -> 2 * 128 * 4 = 1024 bytes per token per layer (approx)
        # Let's use a larger dim to fill memory faster and have fewer tokens
        dim = 128
        
        # Insert one small sequence to initialize bytes_per_token
        tokens_init = [1, 2, 3]
        kv_init = self._create_kv(3, dim=dim)
        self.cache._insert_impl(tokens_init, kv_init)
        
        bytes_per_token = self.cache.bytes_per_token
        print(f"Bytes per token: {bytes_per_token}")
        self.assertIsNotNone(bytes_per_token)
        
        # Calculate how many tokens fit in 1MB
        max_tokens = int(capacity / bytes_per_token)
        print(f"Max tokens capacity: {max_tokens}")
        
        # Sequence A: takes 40% capacity
        len_a = int(max_tokens * 0.4)
        tokens_a = list(range(1000, 1000 + len_a))
        kv_a = self._create_kv(len_a, dim=dim)
        self.cache._insert_impl(tokens_a, kv_a)
        
        # Sequence B: takes 40% capacity
        len_b = int(max_tokens * 0.4)
        tokens_b = list(range(2000, 2000 + len_b))
        kv_b = self._create_kv(len_b, dim=dim)
        self.cache._insert_impl(tokens_b, kv_b)
        
        # Current usage: ~80% + init
        
        # Sequence C: takes 40% capacity -> Should trigger eviction
        len_c = int(max_tokens * 0.4)
        tokens_c = list(range(3000, 3000 + len_c))
        kv_c = self._create_kv(len_c, dim=dim)
        self.cache._insert_impl(tokens_c, kv_c)
        
        # Verify that we didn't exceed capacity significantly (soft limit)
        current_usage = self.cache.current_memory_usage
        print(f"Current usage: {current_usage} bytes, Capacity: {capacity} bytes")
        self.assertLessEqual(current_usage, capacity * 1.1) # Allow small margin
        
        # Verify that something was evicted
        # We inserted A, B, C (total 120%). At least one should be gone or partial.
        # Since we accessed A, B, C in order of insertion, A is oldest?
        # No, insert updates last_accessed?
        # In insert: node.last_accessed = time.time()
        # So C is most recent. A is oldest. A should be evicted.
        
        m_a, _, _ = self.cache.match_prefix(tokens_a)
        m_b, _, _ = self.cache.match_prefix(tokens_b)
        m_c, _, _ = self.cache.match_prefix(tokens_c)
        
        print(f"Match lengths: A={len(m_a)}/{len(tokens_a)}, B={len(m_b)}/{len(tokens_b)}, C={len(m_c)}/{len(tokens_c)}")
        
        # At least one should be incomplete
        is_evicted = (len(m_a) < len(tokens_a)) or (len(m_b) < len(tokens_b)) or (len(m_c) < len(tokens_c))
        self.assertTrue(is_evicted, "Eviction should have occurred")

    def test_s1_s2_s3_scenario(self):
        print("\n=== Starting Test: S1, S2(Longer), S3(Shorter) ===")
        
        # Setup
        self.cache = RadixCache(capacity_bytes=100 * 1024 * 1024) # 100MB
        
        # Define tokens
        # S1: 20 tokens
        s1_tokens = list(range(100, 120))
        # S2: 25 tokens (S1 + 5 more)
        s2_tokens = s1_tokens + list(range(120, 125))
        # S3: 10 tokens (First 10 of S1)
        s3_tokens = s1_tokens[:10]
        
        print(f"S1 (Base): {len(s1_tokens)} tokens")
        print(f"S2 (Longer): {len(s2_tokens)} tokens")
        print(f"S3 (Shorter): {len(s3_tokens)} tokens")
        
        # Create dummy KV cache for S1
        # Shape: (1, 1, 20, 1) - (Batch, Heads, SeqLen, Dim)
        # Values: 100..119
        kv_s1 = [mx.arange(20).reshape(1, 1, 20, 1) + 100]
        
        # --- Step 1: Insert S1 ---
        print("\n--- Step 1: Inserting S1 ---")
        self.cache._insert_impl(s1_tokens, kv_s1)
        
        print(f"Cache stats: {self.cache.get_stats()}")
        
        # --- Step 2: Match S2 (S1 + S1c) ---
        print("\n--- Step 2: Matching S2 (S1 + S1c) ---")
        matched_tokens, cached_kv, node = self.cache.match_prefix(s2_tokens)
        
        print(f"Matched length: {len(matched_tokens)}")
        self.assertEqual(len(matched_tokens), 20)
        self.assertEqual(matched_tokens, s1_tokens)
        self.assertIsNotNone(cached_kv)
        self.assertEqual(cached_kv[0].shape[2], 20)
        print(">> S2 Result: SUCCESS (Full S1 reused)")
        
        # --- Step 3: Match S3 (S1a) ---
        print("\n--- Step 3: Matching S3 (S1a) ---")
        matched_tokens_3, cached_kv_3, node_3 = self.cache.match_prefix(s3_tokens)
        
        print(f"Matched length: {len(matched_tokens_3)}")
        self.assertEqual(len(matched_tokens_3), 10)
        self.assertEqual(matched_tokens_3, s3_tokens)
        self.assertIsNotNone(cached_kv_3)
        
        # Verify sliced cache
        sliced_len = cached_kv_3[0].shape[2]
        self.assertEqual(sliced_len, 10)
        
        # Verify values
        # Sliced should be 100..109. Last value 109.
        last_val = cached_kv_3[0][0, 0, -1, 0].item()
        self.assertEqual(last_val, 109, f"Last value of slice should be 109, got {last_val}")
        
        print(">> S3 Result: SUCCESS (Sliced S1 reused)")

if __name__ == '__main__':
    unittest.main()
