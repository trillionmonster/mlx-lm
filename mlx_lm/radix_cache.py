# Copyright © 2024 Apple Inc.

import time
import heapq
import logging
import threading
import concurrent.futures
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional, Any, Union
import mlx.core as mx
import mlx.utils

logger = logging.getLogger(__name__)

class NonSliceable:
    """
    Wrapper for data that cannot be sliced (e.g. Mamba state).
    """
    def __init__(self, data):
        self.data = data

@dataclass
class RadixTreeNode:
    """
    Radix Tree Node
    """
    key: Tuple[int, ...] = field(default_factory=tuple)  # Token sequence fragment corresponding to this node
    value: Optional[List[Any]] = None  # KV Cache corresponding to this node (usually List[Tuple[mx.array, mx.array]])
    children: Dict[int, 'RadixTreeNode'] = field(default_factory=dict)  # Child node mapping {first_token: Node}
    parent: Optional['RadixTreeNode'] = None
    last_accessed: float = field(default_factory=time.time)
    lock_count: int = 0  # Reference count to prevent eviction of nodes in use
    needs_snapshot: bool = False  # Flag indicating if a snapshot is needed (for Mamba lazy snapshot)
    snapshot_created_count: int = 0  # Count of snapshots created for this node (for statistics)

    def is_leaf(self) -> bool:
        return len(self.children) == 0
    
    def get_full_path(self) -> Tuple[int, ...]:
        """Get the full token path from the root to the current node"""
        path = []
        node = self
        while node.parent is not None:
            path = list(node.key) + path
            node = node.parent
        return tuple(path)

def _slice_cache(cache: List[Any], length: int) -> List[Any]:
    """
    Helper function: Slice KV Cache to the specified length.
    Uses mlx.utils.tree_map to automatically handle arbitrary nested structures (List, Tuple, Dict, etc.).
    """
    if cache is None:
        return None

    def slice_fn(x):
        if isinstance(x, NonSliceable):
            raise ValueError("Cannot slice NonSliceable data")
        # Only operate on MLX arrays
        if isinstance(x, mx.array):
            # Automatically detect length dimension (usually -2: SeqLen)
            # K, V shape is usually (Batch, Heads, SeqLen, Dim)
            axis = -2
            if x.ndim >= 2:
                current_len = x.shape[axis]
                target_len = length
                if target_len < 0:
                    target_len = current_len + target_len
                
                if target_len <= current_len:
                    return x[..., :target_len, :]
        return x

    # tree_map recursively traverses all leaf nodes in cache and applies slice_fn
    return mlx.utils.tree_map(slice_fn, cache)

class RadixCache:
    """
    Radix Tree based KV Cache manager.
    Designed for Unified Memory Architecture (Apple Silicon), does not rely on PagedAttention.
    """
    def __init__(self, capacity_bytes: int = 2 * 1024**3, snapshot_callback=None):
        """
        Args:
            capacity_bytes: Cache capacity limit (bytes). Default 2GB.
            snapshot_callback: Optional snapshot creation callback function
                - Signature: callback(tokens: List[int]) -> cache_state
                - Used to create Mamba intermediate state snapshots on demand
        """
        self.root = RadixTreeNode(key=())
        self.root.last_accessed = float('inf') # Root node is never evicted
        self.nodes_cnt = 1
        self.total_tokens = 0
        self.capacity_bytes = capacity_bytes
        self.bytes_per_token = None # Will be calculated on first insertion
        self.snapshot_callback = snapshot_callback  # Snapshot creation callback
        self.lazy_snapshots_created = 0  # Statistic: number of lazily created snapshots
        
        self.lock = threading.RLock()
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="RadixCacheWorker")

    def _calculate_bytes_per_token(self, kv_cache: List[Any], num_tokens: int) -> float:
        if num_tokens == 0 or kv_cache is None:
            return 0.0
        
        total_bytes = 0
        def count_fn(x):
            nonlocal total_bytes
            if isinstance(x, mx.array):
                total_bytes += x.nbytes
            return x
        
        mlx.utils.tree_map(count_fn, kv_cache)
        return total_bytes / num_tokens

    @property
    def current_memory_usage(self) -> int:
        if self.bytes_per_token is None:
            return 0
        return int(self.total_tokens * self.bytes_per_token)
    
    def get_stats(self) -> dict:
        """Get cache statistics"""
        with self.lock:
            return {
                "nodes_count": self.nodes_cnt,
                "total_tokens": self.total_tokens,
                "memory_usage_bytes": self.current_memory_usage,
                "memory_usage_mb": self.current_memory_usage / (1024**2),
                "capacity_mb": self.capacity_bytes / (1024**2),
                "usage_percent": (self.current_memory_usage / self.capacity_bytes * 100) if self.capacity_bytes > 0 else 0,
                "lazy_snapshots_created": self.lazy_snapshots_created,
            }

    def match_prefix(self, tokens: List[int]) -> Tuple[List[int], Optional[List[Any]], 'RadixTreeNode']:
        """
        Find the longest matching prefix in the tree.
        
        Args:
            tokens: Full Token sequence
            
        Returns:
            (matched_tokens, cached_kv, last_node)
            - matched_tokens: List of matched Tokens
            - cached_kv: Corresponding KV Cache
            - last_node: The last matched node (used for subsequent insert)
        """
        with self.lock:
            node = self.root
            matched_len = 0
            tokens_tuple = tuple(tokens)
            
            # Record the last valid match (node containing valid Cache)
            # Root node usually has value=None, but logically treated as length 0 match
            last_valid_match = ([], None, node)
            
            logger.debug(f"[RadixCache] Searching for prefix in {len(tokens)} tokens. Tree has {self.nodes_cnt} nodes, {self.total_tokens} total tokens.")
            
            # Traverse the tree
            while True:
                node.last_accessed = time.time()
                
                # Check if remaining tokens can match a child node
                remaining_tokens = tokens_tuple[matched_len:]
                if not remaining_tokens:
                    break
                    
                first_token = remaining_tokens[0]
                if first_token not in node.children:
                    logger.debug(f"[RadixCache] No child for token {first_token} at depth {matched_len}")
                    break
                
                child = node.children[first_token]
                logger.debug(f"[RadixCache] Found child with key length {len(child.key)}, has_value={child.value is not None}")
                
                # Check if child.key is a prefix of remaining_tokens
                # Optimization: Python tuple comparison is very fast
                common_len = 0
                min_len = min(len(child.key), len(remaining_tokens))
                
                # Find common prefix length
                # Can use binary search or direct comparison, direct comparison is fast for short sequences
                if remaining_tokens[:len(child.key)] == child.key:
                    common_len = len(child.key)
                    logger.debug(f"[RadixCache] Quick match: common_len={common_len}")
                else:
                    # Partial match, need to compare one by one to find match length
                    for i in range(min_len):
                        if child.key[i] != remaining_tokens[i]:
                            break
                        common_len += 1
                    else:
                        # Loop finished normally, meaning min_len fully matched
                        common_len = min_len
                    logger.debug(f"[RadixCache] Slow match: common_len={common_len}, child.key_len={len(child.key)}, remaining_len={len(remaining_tokens)}")
                
                if common_len == len(child.key):
                    # Fully matched this child node, continue down
                    matched_len += common_len
                    node = child
                    logger.debug(f"[RadixCache] Fully matched child. New matched_len={matched_len}, node.value={node.value is not None}")
                    
                    # If current node has valid Cache, update last_valid_match
                    if node.value is not None:
                        last_valid_match = (list(tokens_tuple[:matched_len]), node.value, node)
                        logger.debug(f"[RadixCache] Updated last_valid_match to {matched_len} tokens")
                else:
                    # Partial match
                    # For NonSliceable (Mamba) data, cannot use partial cache
                    # But can still be part of the path, continue searching down for full match
                    logger.debug(f"[RadixCache] Partial match: common_len={common_len} < child.key_len={len(child.key)}")
                    
                    # Check if cache contains NonSliceable data
                    has_nonsliceable = False
                    if child.value is not None:
                        for layer_cache in child.value:
                            if isinstance(layer_cache, NonSliceable):
                                has_nonsliceable = True
                                break
                    
                    if has_nonsliceable:
                        # NonSliceable (Mamba) data: Check if split node has snapshot
                        if child.value is not None:
                            # Found split node with snapshot, can use!
                            total_match_len = matched_len + common_len
                            logger.debug(f"[RadixCache] Found NonSliceable snapshot at depth {total_match_len}")
                            return list(tokens_tuple[:total_match_len]), child.value, node
                        elif child.needs_snapshot and self.snapshot_callback is not None:
                            # Node needs snapshot but doesn't have one yet, try to create on demand
                            total_match_len = matched_len + common_len
                            full_path = node.get_full_path() + child.key  # Full path
                            
                            logger.debug(f"[RadixCache] Lazy creating snapshot for split node at depth {total_match_len}")
                            try:
                                # Call callback function to calculate snapshot
                                snapshot = self.snapshot_callback(list(full_path))
                                if snapshot is not None:
                                    child.value = snapshot
                                    child.needs_snapshot = False
                                    child.snapshot_created_count += 1
                                    self.lazy_snapshots_created += 1
                                    logger.debug(f"[RadixCache] Successfully created snapshot. Total lazy snapshots: {self.lazy_snapshots_created}")
                                    return list(tokens_tuple[:total_match_len]), snapshot, node
                                else:
                                    logger.debug(f"[RadixCache] Snapshot creation returned None")
                            except Exception as e:
                                logger.debug(f"[RadixCache] Failed to create snapshot: {e}")
                            
                            # Creation failed, fall back to last valid match
                            logger.debug(f"[RadixCache] Falling back to last valid match")
                            break
                        else:
                            # No snapshot and cannot create, cannot continue matching
                            # Return last valid match (last_valid_match)
                            logger.debug(f"[RadixCache] Partial match but no snapshot at split node. Using last valid match.")
                            break
                    elif child.value is not None:
                        # Normal KV Cache, can try to slice
                        total_match_len = matched_len + common_len
                        try:
                            cached_kv = _slice_cache(child.value, common_len)
                            logger.debug(f"[RadixCache] Sliced cache at depth {total_match_len}")
                            return list(tokens_tuple[:total_match_len]), cached_kv, node
                        except ValueError:
                            pass
                    
                    # Cannot use partial cache, stop searching
                    break
            
            # Return last valid match (could be root node, i.e., no match)
            logger.debug(f"[RadixCache] Returning last valid match: {len(last_valid_match[0])} tokens")
            return last_valid_match

    def insert(self, tokens: List[int], kv_cache: List[Any], intermediate_states: Optional[List[List[Any]]] = None):
        """
        Insert new Token sequence and KV Cache into the tree.
        Executes asynchronously, does not block the main thread.
        """
        self.executor.submit(self._insert_impl, tokens, kv_cache, intermediate_states)

    def _insert_impl(self, tokens: List[int], kv_cache: List[Any], intermediate_states: Optional[List[List[Any]]] = None):
        """
        Actual insertion logic, executed in a background thread.
        """
        with self.lock:
            if not tokens:
                return

            logger.debug(f"[RadixCache] Inserting {len(tokens)} tokens. First 10: {tokens[:10]}")
            logger.debug(f"[RadixCache] Has intermediate_states: {intermediate_states is not None}, count: {len(intermediate_states) if intermediate_states else 0}")

            # 1. Initialize bytes_per_token
            if self.bytes_per_token is None and kv_cache is not None:
                self.bytes_per_token = self._calculate_bytes_per_token(kv_cache, len(tokens))
                logger.debug(f"[RadixCache] Calculated bytes_per_token: {self.bytes_per_token:.2f} bytes")

            # 2. Check capacity and perform eviction
            tokens = tuple(tokens)
            node = self.root
            idx = 0
            
            while idx < len(tokens):
                node.last_accessed = time.time()
                remaining = tokens[idx:]
                if not remaining:
                    break
                    
                first_token = remaining[0]
                
                if first_token not in node.children:
                    # Case 1: No matching child node, create a new child node directly
                    
                    # --- Capacity Check ---
                    tokens_needed = len(remaining)
                    self._ensure_capacity(tokens_needed)
                    # ----------------
                    
                    new_node = RadixTreeNode(
                        key=remaining,
                        value=kv_cache, # Here stores the complete KV Cache
                        parent=node
                    )
                    node.children[first_token] = new_node
                    self.nodes_cnt += 1
                    self.total_tokens += len(remaining)
                    logger.debug(f"[RadixCache] Created new node for {len(remaining)} tokens.")
                    return
                
                child = node.children[first_token]
                
                # Calculate common prefix
                common_len = 0
                min_len = min(len(child.key), len(remaining))
                for i in range(min_len):
                    if child.key[i] != remaining[i]:
                        break
                    common_len += 1
                else:
                    common_len = min_len
                
                if common_len == len(child.key):
                    # Case 2: Fully matched current child node, continue down
                    node = child
                    idx += common_len
                    # If this is the end of the path, update value
                    if idx == len(tokens):
                        node.value = kv_cache
                        logger.debug(f"[RadixCache] Updated existing node value.")
                else:
                    # Case 3: Partial match, need to split child node (Split)
                    logger.debug(f"[RadixCache] Splitting node. Common len: {common_len}")
                    
                    # --- Capacity Check ---
                    remaining_new_key = remaining[common_len:]
                    if remaining_new_key:
                        self._ensure_capacity(len(remaining_new_key))
                    # ----------------

                    # 1. Create split point (Parent)
                    split_key = child.key[:common_len]     # [A, B]
                    remaining_child_key = child.key[common_len:] # [C, D]
                    
                    # Calculate split point position in full tokens sequence (used to locate intermediate_states)
                    split_position = idx + common_len
                    
                    # Check if contains NonSliceable data
                    has_nonsliceable = False
                    if child.value is not None:
                        for layer_cache in child.value:
                            if isinstance(layer_cache, NonSliceable):
                                has_nonsliceable = True
                                break
                    
                    # Try to create cache for split node
                    split_value = None
                    if has_nonsliceable:
                        # For NonSliceable (Mamba) data:
                        # Strategy: Lazy Snapshot Creation
                        # 1. If intermediate_states is provided, use it
                        # 2. Otherwise, check if it is worth creating a snapshot for this split point
                        #    - Current heuristic: Always create (because split means multi-path reuse)
                        if intermediate_states and split_position > 0 and split_position <= len(intermediate_states):
                            split_value = intermediate_states[split_position - 1]
                            logger.debug(f"[RadixCache] Split node uses provided intermediate snapshot at position {split_position}")
                        else:
                            # Mark as needing snapshot creation (lazy until really needed)
                            # Set to None for now, detect and create on demand in match_prefix
                            split_value = None
                            logger.debug(f"[RadixCache] Split node marked for lazy snapshot creation at position {split_position}")
                    elif child.value is not None:
                        # Sliceable data (Transformer KV Cache), slice directly
                        try:
                            split_value = _slice_cache(child.value, -len(remaining_child_key))
                            logger.debug(f"[RadixCache] Split node has sliced KV cache")
                        except Exception as e:
                            logger.debug(f"[RadixCache] Failed to slice cache: {e}")
                    
                    split_node = RadixTreeNode(
                        key=split_key,
                        value=split_value,
                        parent=node,
                        needs_snapshot=(has_nonsliceable and split_value is None)  # Mark as needing snapshot
                    )
                    logger.debug(f"[RadixCache] Created split node with {len(split_key)} tokens, has_value={split_node.value is not None}, needs_snapshot={split_node.needs_snapshot}")
                    
                    # 2. Adjust original child
                    del node.children[first_token] # Remove original reference
                    node.children[split_key[0]] = split_node # Add new parent node
                    
                    child.parent = split_node
                    child.key = remaining_child_key
                    split_node.children[remaining_child_key[0]] = child # Original child becomes child of split_node
                    
                    # 3. Create new branch (if there are remaining)
                    if remaining_new_key:
                        new_leaf = RadixTreeNode(
                            key=remaining_new_key,
                            value=kv_cache,
                            parent=split_node
                        )
                        split_node.children[remaining_new_key[0]] = new_leaf
                        self.nodes_cnt += 1
                        self.total_tokens += len(remaining_new_key)
                    else:
                        # Ended exactly at split point
                        split_node.value = kv_cache
                    
                    self.nodes_cnt += 1 # Added a split_node
                    return

    def _ensure_capacity(self, tokens_needed: int):
        """
        Ensure there is enough space to accommodate tokens_needed.
        If not enough, trigger eviction.
        """
        if self.bytes_per_token is None or self.bytes_per_token == 0:
            return

        bytes_needed = tokens_needed * self.bytes_per_token
        current_usage = self.total_tokens * self.bytes_per_token
        
        if (current_usage + bytes_needed) > self.capacity_bytes:
             logger.debug(f"[RadixCache] Capacity check: Usage {current_usage/1024**3:.2f}GB + Needed {bytes_needed/1024**3:.2f}GB > Limit {self.capacity_bytes/1024**3:.2f}GB")

        # If a single request exceeds total capacity, there is no way, just try best effort
        if bytes_needed > self.capacity_bytes:
            self.evict(self.total_tokens) 
            return

        while (self.total_tokens * self.bytes_per_token + bytes_needed) > self.capacity_bytes:
            current_usage = self.total_tokens * self.bytes_per_token
            bytes_to_free = (current_usage + bytes_needed) - self.capacity_bytes
            tokens_to_free = int(bytes_to_free / self.bytes_per_token) + 1
            
            freed = self.evict(tokens_to_free)
            if freed == 0:
                break

    def evict(self, num_tokens_to_free: int) -> int:
        """
        LRU Eviction Strategy.
        Returns:
            int: Number of tokens actually freed
        """
        # Collect all leaf nodes
        leaves = []
        stack = [self.root]
        while stack:
            node = stack.pop()
            if node.is_leaf() and node != self.root:
                leaves.append(node)
            for child in node.children.values():
                stack.append(child)
        
        # Sort by last access time (oldest first)
        leaves.sort(key=lambda x: x.last_accessed)
        
        freed_tokens = 0
        for leaf in leaves:
            if freed_tokens >= num_tokens_to_free:
                break
            
            if leaf.lock_count > 0:
                continue
                
            # Delete leaf node
            parent = leaf.parent
            if parent:
                del parent.children[leaf.key[0]]
                node_tokens = len(leaf.key)
                freed_tokens += node_tokens
                self.total_tokens -= node_tokens
                self.nodes_cnt -= 1
                
                # Explicitly release memory references
                leaf.value = None
                leaf.children = None
                
                # If parent becomes a leaf node and has no value (just an intermediate node), can also consider recursive deletion
                # But for simplicity, only delete one level here.
        
        if freed_tokens > 0:
            logger.debug(f"[RadixCache] Evicted {freed_tokens} tokens to free space. Current usage: {self.total_tokens} tokens.")
            
        return freed_tokens
