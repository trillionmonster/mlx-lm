import time
import logging
import itertools
import mlx.core as mx
from typing import List, Any, Tuple, Optional

from .generate import BatchGenerator, Batch
from .radix_cache import RadixCache, NonSliceable
from .models.cache import (
    BatchKVCache, 
    KVCache, 
    ArraysCache, 
    MambaCache, 
    RotatingKVCache, 
    BatchRotatingKVCache,
    CacheList
)

logger = logging.getLogger(__name__)

def _make_cache(model, left_padding):
    """
    Convert a list of regular caches into their corresponding
    batch-aware caches.
    """
    def to_batch_cache(c):
        if isinstance(c, KVCache):
            return BatchKVCache(left_padding)
        elif isinstance(c, ArraysCache):
            c.left_padding = mx.array(left_padding)
            return c
        elif isinstance(c, RotatingKVCache):
            if c.keep > 0:
                raise ValueError("RotatingKVCache with keep tokens is not supported.")
            return BatchRotatingKVCache(c.max_size, left_padding)
        elif isinstance(c, CacheList):
            return CacheList(*(to_batch_cache(sub_c) for sub_c in c.caches))
        else:
            raise ValueError(f"{type(c)} does not yet support batching")

    if hasattr(model, "make_cache"):
        cache = model.make_cache()
        return [to_batch_cache(c) for c in cache]
    else:
        return [BatchKVCache(left_padding) for _ in model.layers]

def _left_pad_prompts(prompts, max_length=None):
    if max_length is None:
        max_length = max(len(p) for p in prompts)
    reversed_prompts = [p[::-1] for p in prompts]
    padded = list(itertools.zip_longest(*reversed_prompts, fillvalue=0))
    return mx.array(padded, dtype=mx.int32).T[:, ::-1]

class PromptCacheBatchGenerator(BatchGenerator):
    def __init__(
        self,
        model,
        radix_cache_size: int = 2 * 1024 * 1024 * 1024, # 2GB default
        **kwargs
    ):
        super().__init__(model, **kwargs)
        # 创建 RadixCache 时传入快照创建回调
        self.radix_cache = RadixCache(
            capacity_bytes=radix_cache_size,
            snapshot_callback=self._create_snapshot
        )
        self.request_tokens = {}  # Tracks tokens for each request (uid -> list of tokens)
        self.prompt_tokens = {}   # Stores original prompt tokens (uid -> list of tokens)
    
    def _create_snapshot(self, tokens: List[int]) -> Optional[List[Any]]:
        """
        按需创建指定 token 序列的状态快照（用于 Mamba 层）
        
        Args:
            tokens: 需要计算快照的 token 序列
            
        Returns:
            快照状态 (List[cache_per_layer]) 或 None
        """
        print(f"[SnapshotCallback] Creating snapshot for {len(tokens)} tokens")
        
        try:
            # 1. 创建临时 cache
            temp_cache = self.model.make_cache() if hasattr(self.model, "make_cache") else None
            if temp_cache is None:
                return None
            
            # 2. 将 tokens 转为 mx.array
            tokens_array = mx.array([tokens], dtype=mx.int32)  # Shape: [1, seq_len]
            
            # 3. 运行模型获取状态
            _ = self.model(tokens_array, cache=temp_cache)
            mx.eval(_)  # 确保计算完成
            
            # 4. 提取快照
            snapshot = []
            for layer_cache in temp_cache:
                if isinstance(layer_cache, KVCache):
                    # Transformer cache: 提取 K, V
                    snapshot.append((layer_cache.keys, layer_cache.values))
                elif isinstance(layer_cache, ArraysCache):
                    # Mamba cache: 包装为 NonSliceable
                    snapshot.append(NonSliceable(layer_cache.cache))
                else:
                    snapshot.append(None)
            
            print(f"[SnapshotCallback] Successfully created snapshot")
            return snapshot
            
        except Exception as e:
            print(f"[SnapshotCallback] Failed to create snapshot: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _process_prompts(self, prompts):
        """
        Process a batch of new prompts, leveraging the RadixCache.
        """
        uids, inputs, max_tokens = zip(*prompts)
        
        # Convert inputs to lists for easier handling
        input_lists = [inp.tolist() if isinstance(inp, mx.array) else list(inp) for inp in inputs]
        
        # Store tokens for tracking
        for uid, inp_list in zip(uids, input_lists):
            self.request_tokens[uid] = inp_list.copy()
            self.prompt_tokens[uid] = inp_list # Keep reference for cache insertion
        
        # 1. Match prefixes in RadixCache
        # matches is a list of (matched_tokens, cached_kv, last_node)
        matches = []
        for inp in input_lists:
            logger.debug(f"[PromptCache] Matching prefix for input (len={len(inp)}). First 10: {inp[:10]}")
            m = self.radix_cache.match_prefix(inp)
            matches.append(m)
        
        # 2. Calculate lengths and padding
        lengths = [len(p) for p in input_lists]
        max_length = max(lengths)
        left_padding = [max_length - l for l in lengths]
        
        # 3. Determine effective match lengths and batch start index
        effective_lens = [len(m[0]) for m in matches]
        
        # Check for NonSliceable (Mamba) caches
        has_nonsliceable = False
        for m in matches:
            if m[1] is not None:
                # Check the first layer's cache
                first_layer = m[1][0]
                if isinstance(first_layer, NonSliceable):
                    has_nonsliceable = True
                    break
        
        # Iteratively determine batch_start_index
        while True:
            physical_starts = [lp + l for lp, l in zip(left_padding, effective_lens)]
            batch_start_index = min(physical_starts)
            
            if not has_nonsliceable:
                # For sliceable caches (Transformers), we can always truncate to batch_start_index
                break
            
            changed = False
            for i in range(len(input_lists)):
                physical_match_end = left_padding[i] + effective_lens[i]
                if physical_match_end > batch_start_index:
                    # We would need to truncate this match to align with the batch.
                    # Since it's NonSliceable, we can't. We must discard it.
                    if effective_lens[i] > 0:
                        effective_lens[i] = 0
                        changed = True
            
            if not changed:
                break

        # 4. Construct the Batch Cache
        prompt_cache = None
        
        # Only create cache if we actually have something to reuse
        if batch_start_index > 0:
            prompt_cache = _make_cache(self.model, left_padding)
            
            # We need to populate prompt_cache with data from matches
            # Iterate over layers
            for layer_idx, layer_cache in enumerate(prompt_cache):
                
                if isinstance(layer_cache, BatchKVCache):
                    # Transformer Cache
                    # We need to construct keys/values tensors
                    # Find a reference tensor to get shapes/dtypes
                    ref_k, ref_v = None, None
                    for m in matches:
                        if m[1] is not None:
                            c = m[1][layer_idx]
                            if isinstance(c, tuple): # (k, v)
                                ref_k, ref_v = c
                                break
                    
                    if ref_k is not None:
                        B = len(input_lists)
                        if ref_k.ndim == 4:
                            n_kv_heads = ref_k.shape[1]
                            head_dim = ref_k.shape[3]
                            v_dim = ref_v.shape[3]
                        else:
                            n_kv_heads = ref_k.shape[0]
                            head_dim = ref_k.shape[2]
                            v_dim = ref_v.shape[2]
                        dtype = ref_k.dtype
                        
                        # Allocate batch tensors
                        keys = mx.zeros((B, n_kv_heads, batch_start_index, head_dim), dtype=dtype)
                        values = mx.zeros((B, n_kv_heads, batch_start_index, v_dim), dtype=dtype)
                        
                        for i, m in enumerate(matches):
                            # We only use the cache if effective_lens[i] allows it
                            # effective_lens[i] is the logical length of the match
                            # valid_len is how much of that match fits into the batch_start_index
                            valid_len = batch_start_index - left_padding[i]
                            
                            if valid_len > 0 and m[1] is not None:
                                c = m[1][layer_idx]
                                if isinstance(c, tuple):
                                    k, v = c
                                    # Slice k, v to valid_len
                                    # k shape: (H, S, D) or (B, H, S, D)
                                    if k.ndim == 4:
                                        k_slice = k[0, :, :valid_len, :]
                                        v_slice = v[0, :, :valid_len, :]
                                    else:
                                        k_slice = k[:, :valid_len, :]
                                        v_slice = v[:, :valid_len, :]
                                    
                                    # Insert into batch tensors
                                    # Destination: [i, :, left_padding[i]:batch_start_index, :]
                                    # The length of destination slice is batch_start_index - left_padding[i] == valid_len
                                    try:
                                        keys[i, :, left_padding[i]:batch_start_index, :] = k_slice
                                        values[i, :, left_padding[i]:batch_start_index, :] = v_slice
                                    except ValueError as e:
                                        print(f"Error in prompt cache assignment:")
                                        print(f"i={i}, valid_len={valid_len}")
                                        print(f"left_padding[i]={left_padding[i]}, batch_start_index={batch_start_index}")
                                        print(f"keys.shape={keys.shape}")
                                        print(f"k.shape={k.shape}, k.ndim={k.ndim}")
                                        print(f"k_slice.shape={k_slice.shape}")
                                        print(f"Target slice shape: {keys[i, :, left_padding[i]:batch_start_index, :].shape}")
                                        raise e
                        
                        layer_cache.state = (keys, values, layer_cache.offset + batch_start_index, layer_cache.left_padding)
                        
                elif isinstance(layer_cache, ArraysCache):
                    # Mamba Cache
                    # We need to find the structure of the state.
                    ref_states = None
                    for m in matches:
                        if m[1] is not None:
                            c = m[1][layer_idx]
                            if isinstance(c, NonSliceable):
                                c = c.data
                            if isinstance(c, list):
                                ref_states = c
                                break
                    
                    if ref_states is not None:
                        # ref_states is a list of arrays (e.g. SSM state, Conv state)
                        num_states = len(ref_states)
                        batched_states = []
                        
                        for s_idx in range(num_states):
                            # Collect state s_idx from all batch items
                            batch_items = []
                            for i, m in enumerate(matches):
                                state_to_use = None
                                valid_len = batch_start_index - left_padding[i]
                                
                                # For Mamba, valid_len must equal effective_lens[i] (no truncation allowed)
                                # And effective_lens[i] must be > 0
                                if valid_len > 0 and effective_lens[i] > 0 and m[1] is not None:
                                    c = m[1][layer_idx]
                                    if isinstance(c, NonSliceable):
                                        c = c.data
                                    if isinstance(c, list) and len(c) > s_idx:
                                        state_to_use = c[s_idx]
                                
                                if state_to_use is None:
                                    # Initialize zero state
                                    ref_s = ref_states[s_idx]
                                    # Shape of ref_s is usually (1, D, N) or similar?
                                    # We need a zero tensor of the same shape
                                    state_to_use = mx.zeros_like(ref_s)
                                
                                batch_items.append(state_to_use)
                            
                            # Stack
                            batched_s = mx.stack(batch_items, axis=0)
                            batched_states.append(batched_s)
                        
                        layer_cache.cache = batched_states

        if prompt_cache is None:
             prompt_cache = _make_cache(self.model, left_padding)

        # 5. Prepare inputs for the step
        padded_inputs = _left_pad_prompts(input_lists, max_length)
        step_inputs = padded_inputs[:, batch_start_index:]
        
        # Handle full cache hit case
        if step_inputs.shape[1] == 0:
            # Must run at least one token to get logits
            step_inputs = padded_inputs[:, -1:]
            # Adjust cache offset back by 1
            for cache in prompt_cache:
                if hasattr(cache, 'offset'):
                    if isinstance(cache.offset, mx.array):
                        cache.offset = cache.offset - 1
                    else:
                        cache.offset -= 1
        
        # 6. Run the model step (Prefill)
        # Note: For Mamba layers, we don't need to collect intermediate states
        # The final state at the end of prefill already contains all information
        # Intermediate snapshots are only created when the RadixTree actually splits
        y, logprobs = self._step(step_inputs, prompt_cache)
        mx.async_eval(y, logprobs)

        # 7. Insert the result into RadixCache
        # try:
        #     for i, uid in enumerate(uids):
        #         if uid in self.prompt_tokens:
        #             tokens = self.prompt_tokens[uid]
        #             cache_data = []
        #             
        #             # Determine left padding for this batch item
        #             lp = left_padding[i]
        #             
        #             # Extract cache for this item
        #             for layer_cache in prompt_cache:
        #                 if isinstance(layer_cache, BatchKVCache):
        #                     valid_len = len(tokens)
        #                     k = layer_cache.keys[i, :, lp : lp + valid_len, :]
        #                     v = layer_cache.values[i, :, lp : lp + valid_len, :]
        #                     cache_data.append((k, v))
        #                     
        #                 elif isinstance(layer_cache, ArraysCache):
        #                     states = []
        #                     for stacked_arr in layer_cache.cache:
        #                         s = stacked_arr[i]
        #                         states.append(s)
        #                     cache_data.append(NonSliceable(states))
        #                 else:
        #                     cache_data.append(None)
        #             
        #             # Pass intermediate states if available
        #             # Note: We no longer collect intermediate states during prefill
        #             # They will be created lazily when RadixTree splits occur
        #             self.radix_cache.insert(tokens, cache_data, intermediate_states=None)
        #             del self.prompt_tokens[uid]
        #     
        #     print(f"[RadixCache] Inserted {len(uids)} prompts into cache.")
        #     
        # except Exception as ex:
        #     print(f"[ERROR] Failed to insert into cache: {ex}")

        return Batch(
            list(uids),
            y,
            logprobs,
            list(max_tokens),
            [0] * len(uids),
            prompt_cache,
        )

    def _next(self):
        tic = time.perf_counter()

        # 1. Try to add new prompts
        batch = self.active_batch
        num_active = len(batch) if batch else 0

        if (
            len(self.unprocessed_prompts) > 0
            and (self.completion_batch_size - num_active) > 0
        ):
            n_to_take = min(
                self.completion_batch_size - num_active,
                self.prefill_batch_size,
            )
            prompts = self.unprocessed_prompts[:n_to_take]
            self.unprocessed_prompts = self.unprocessed_prompts[n_to_take:]

            new_batch = self._process_prompts(prompts)

            if self.active_batch is None:
                self.active_batch = new_batch
            else:
                self.active_batch.extend(new_batch)

            toc = time.perf_counter()
            self._stats.prompt_time += toc - tic

        # 2. Run decode for active batch
        batch = self.active_batch
        if batch is None:
            return []

        y, logprobs = batch.y, batch.logprobs
        batch.y, batch.logprobs = self._step(y[:, None], batch.cache)
        mx.async_eval(batch.y, batch.logprobs)

        y = y.tolist()
        toc = time.perf_counter()
        self._stats.generation_time += toc - tic

        keep_idx = []
        end_idx = []
        responses = []

        for e, (t, uid, num_tok, max_tok) in enumerate(
            zip(y, batch.uids, batch.num_tokens, batch.max_tokens)
        ):
            # Update tokens
            if uid in self.request_tokens:
                self.request_tokens[uid].append(t)
            
            num_tok += 1
            batch.num_tokens[e] = num_tok
            if t in self.stop_tokens:
                finish_reason = "stop"
                end_idx.append(e)
            elif num_tok >= max_tok:
                finish_reason = "length"
                end_idx.append(e)
            else:
                finish_reason = None
                keep_idx.append(e)
            responses.append(self.Response(uid, t, logprobs[e], finish_reason))
            
            if finish_reason:
                # if uid in self.request_tokens:
                #     del self.request_tokens[uid]
                pass

        # Remove any finished completions
        if len(end_idx):
            # Insert finished requests into cache
            # Note: We must cache BEFORE filtering the batch, because filtering modifies batch.uids
            self._cache_finished_requests(batch, end_idx)
            
            # Clean up request_tokens for finished requests
            # We also need to do this BEFORE filtering because we need the UIDs from the current batch state
            for e in end_idx:
                 uid = batch.uids[e]
                 if uid in self.request_tokens:
                     del self.request_tokens[uid]

            if len(keep_idx) > 0:
                batch.filter(keep_idx)
            else:
                self.active_batch = None
        
        self._stats.generation_tokens += len(responses)
        return responses

    def _cache_finished_requests(self, batch, end_idx):
        for idx in end_idx:
            uid = batch.uids[idx]
            if uid not in self.request_tokens:
                continue
            
            tokens = self.request_tokens[uid]
            
            # Extract cache
            cache_data = []
            for layer_cache in batch.cache:
                if isinstance(layer_cache, BatchKVCache):
                    # Transformer
                    # keys: (B, H, S, D)
                    # offset is the current length
                    # left_padding[idx] is the start
                    
                    lp = layer_cache.left_padding[idx]
                    if isinstance(lp, mx.array):
                        lp = lp.item()
                    
                    offset = layer_cache.offset
                    if isinstance(offset, mx.array):
                        if offset.size == 1:
                            offset = offset.item()
                        else:
                            # If offset is an array (per-batch offset), get the one for this index
                            # Note: BatchKVCache usually has a single offset for the whole batch if they are aligned,
                            # or it might be per-batch. Let's check how BatchKVCache is implemented.
                            # In mlx_lm, BatchKVCache.offset is usually an int or a scalar array.
                            # But if it's a vector, we need to pick the right one.
                            # However, standard BatchKVCache in mlx_lm uses a single offset for the batch end.
                            # If we are here, it means offset has > 1 elements.
                            # Let's try to take the first element or handle it if it's per-batch.
                            # Actually, for BatchKVCache, offset is usually the current generation step index common to all.
                            # If it is an array of shape (1,), .item() works.
                            # If it is shape (B,), we should take offset[idx]?
                            # Let's assume if it's an array and size > 1, it might be per-batch offsets (though unlikely for standard BatchKVCache).
                            # But wait, if we are in a batch generation, all sequences are at the same 'offset' in the cache buffer usually?
                            # Let's look at how offset is used.
                            try:
                                if offset.ndim > 0 and offset.shape[0] > idx:
                                     offset = offset[idx].item()
                                else:
                                     offset = offset[0].item()
                            except:
                                 # Fallback
                                 offset = int(offset)

                    # Valid length = offset - lp
                    # This should match len(tokens) approximately
                    
                    # offset in BatchKVCache tracks real tokens (excluding padding)
                    # keys includes padding.
                    # We want [lp : lp + offset] which is [lp : _idx]
                    end_idx = lp + offset
                    k = layer_cache.keys[idx, :, lp:end_idx, :]
                    v = layer_cache.values[idx, :, lp:end_idx, :]
                    cache_data.append((k, v))
                    
                elif isinstance(layer_cache, ArraysCache):
                    # Mamba
                    states = []
                    for stacked_arr in layer_cache.cache:
                        s = stacked_arr[idx]
                        states.append(s)
                    cache_data.append(NonSliceable(states))
                else:
                    cache_data.append(None)
            
            self.radix_cache.insert(tokens, cache_data)
            logger.debug(f"[RadixCache] Inserted finished request {uid} with {len(tokens)} tokens")
