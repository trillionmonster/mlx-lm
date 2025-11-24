import argparse
import time
import threading
import sys
import uuid
import statistics
from queue import Queue

try:
    from openai import OpenAI
except ImportError:
    print("Error: 'openai' module not found. Please install it using 'pip install openai'")
    sys.exit(1)

def generate_prompts(thread_id):
    # Unique prefix to ensure no cross-thread cache hits for Round 1
    # This ensures we are testing the caching mechanism for a new session/request sequence
    unique_prefix = f"Thread-{thread_id}-{uuid.uuid4()} " 
    
    # Base content to reach ~1500 tokens. 
    # "The quick brown fox jumps over the lazy dog. " is 9 words.
    # 150 repetitions * 9 = 1350 words.
    # Depending on tokenizer, this should be well within 1000-2000 tokens.
    base_text = "The quick brown fox jumps over the lazy dog. " * 150
    
    prompt1 = unique_prefix + base_text
    
    # Round 2: prompt1 + suffix
    # Suffix ~ 200 tokens
    suffix1 = " This is the extension for round two. " * 20
    prompt2 = prompt1 + suffix1
    
    # Round 3: prompt1 + different suffix
    suffix2 = " However, this is the alternative ending for round three. " * 20
    prompt3 = prompt1 + suffix2
    
    return [prompt1, prompt2, prompt3]

def send_request(client, model, prompt):
    start_time = time.time()
    first_token_time = None
    token_count = 0
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=50, # Short generation, we care about TTFT
            stream=True
        )
        
        for chunk in response:
            if chunk.choices and chunk.choices[0].delta.content:
                if first_token_time is None:
                    first_token_time = time.time()
                token_count += 1
                
    except Exception as e:
        print(f"Request failed: {e}")
        return None

    end_time = time.time()
    
    if first_token_time:
        ttft = first_token_time - start_time
        gen_time = end_time - first_token_time
        tps = token_count / gen_time if gen_time > 0 else 0
    else:
        ttft = end_time - start_time # Fallback if no tokens
        tps = 0
        
    return {
        "ttft": ttft,
        "tps": tps,
        "total_time": end_time - start_time
    }

def worker(thread_id, base_url, api_key, model, results_queue):
    client = OpenAI(base_url=base_url, api_key=api_key)
    prompts = generate_prompts(thread_id)
    
    thread_results = []
    for i, prompt in enumerate(prompts):
        # print(f"Thread {thread_id} sending Round {i+1}...")
        res = send_request(client, model, prompt)
        if res:
            res["round"] = i + 1
            thread_results.append(res)
        # Small delay to ensure clean separation, though not strictly necessary
        time.sleep(0.1)
        
    results_queue.put(thread_results)

def main():
    parser = argparse.ArgumentParser(description="Test MLX Prompt Cache Acceleration")
    parser.add_argument("--base-url", type=str, default="http://localhost:8080/v1", help="Base URL for the API")
    parser.add_argument("--api-key", type=str, default="EMPTY", help="API Key")
    parser.add_argument("--model", type=str, default="default_model", help="Model name")
    parser.add_argument("--concurrency", type=int, default=4, help="Number of concurrent threads")
    args = parser.parse_args()

    # Sanitize base_url
    if args.base_url.endswith("/chat/completions"):
        args.base_url = args.base_url[:-17]
    if args.base_url.endswith("/"):
        args.base_url = args.base_url[:-1]

    print(f"Starting Prompt Cache Test against {args.base_url}")
    print(f"Concurrency: {args.concurrency}")
    print("Scenario: 3 Rounds per thread.")
    print("  Round 1: Unique prompt (~1500 tokens)")
    print("  Round 2: Round 1 + Suffix 1 (Should hit cache)")
    print("  Round 3: Round 1 + Suffix 2 (Should hit cache)")
    
    queue = Queue()
    threads = []
    
    start_time = time.time()
    
    for i in range(args.concurrency):
        t = threading.Thread(target=worker, args=(i, args.base_url, args.api_key, args.model, queue))
        t.start()
        threads.append(t)
        
    for t in threads:
        t.join()
        
    total_time = time.time() - start_time
    
    # Aggregate results
    all_results = []
    while not queue.empty():
        all_results.extend(queue.get())
        
    # Group by round
    rounds = {1: [], 2: [], 3: []}
    for res in all_results:
        r = res["round"]
        rounds[r].append(res)
        
    print("\nTest Results:")
    print(f"Total Wall Time: {total_time:.2f}s")
    
    for r in range(1, 4):
        data = rounds[r]
        if not data:
            print(f"Round {r}: No data")
            continue
            
        avg_ttft = statistics.mean([d["ttft"] for d in data])
        avg_tps = statistics.mean([d["tps"] for d in data])
        print(f"Round {r}: Avg TTFT = {avg_ttft:.4f}s, Avg TPS = {avg_tps:.2f}")
        
    # Validation
    if rounds[1] and rounds[2]:
        ttft1 = statistics.mean([d["ttft"] for d in rounds[1]])
        ttft2 = statistics.mean([d["ttft"] for d in rounds[2]])
        speedup = ttft1 / ttft2 if ttft2 > 0 else 0
        print(f"\nCache Speedup (Round 1 vs Round 2): {speedup:.2f}x")
        if speedup > 1.2:
            print("SUCCESS: Prompt caching acceleration detected.")
        else:
            print("WARNING: No significant acceleration detected. Check if prompt caching is enabled on server.")

if __name__ == "__main__":
    main()
