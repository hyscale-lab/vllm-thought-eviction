import subprocess
import time
import requests
import threading
import os
import sys
import json
import signal

# --- CONFIGURATION ---
SERVER_SCRIPT = os.path.expanduser("~/thought-clustering/scripts/start_vllm_server.sh")
MODEL_PATH = "/dataset/common/DeepSeek-R1-Distill-Qwen-1.5B"
PORT = 8000 # Your script searches for a port; we'll assume 8000 or check config
BASE_URL = f"http://localhost:{PORT}"
start_eviction_event = threading.Event()

def start_server():
    print(f"🚀 Launching vLLM server...")
    # Change: Connect stdout/stderr directly to your terminal's output
    process = subprocess.Popen(
        [SERVER_SCRIPT, "uv", "1", "0.3", MODEL_PATH],
        stdout=sys.stdout,  # Send server stdout directly to terminal
        stderr=sys.stderr,  # Send server stderr directly to terminal
        text=True,
        preexec_fn=os.setsid
    )
    return process

def wait_for_server():
    """Polls the /v1/models endpoint until the server is live."""
    print("⏳ Waiting for server to initialize (this can take a few minutes)...")
    while True:
        try:
            response = requests.get(f"{BASE_URL}/v1/models")
            if response.status_code == 200:
                print("✅ Server is UP and healthy!")
                break
        except requests.exceptions.ConnectionError:
            pass
        time.sleep(3)

def trigger_eviction(request_id="test", evictable_token_ranges=[(200, 250)]):
    """Hits the custom eviction endpoint mid-generation."""
    
    start_eviction_event.wait()
    time.sleep(2) # Wait for some tokens to be generated
    print("\n[Evictor] ⚡ Triggering /v1/kv_cache/evict...")
    internal_request_id = request_id
    if not request_id.startswith("chatcmpl-"):
        internal_request_id = f"chatcmpl-{request_id}"
    try:
        payload = {
            "request_id": internal_request_id,
            "evictable_token_ranges": evictable_token_ranges,
        }
        res = requests.post(f"{BASE_URL}/v1/kv_cache/evict", json=payload)
        print(f"[Evictor] Status: {res.status_code}, Response: {res.text}")
    except Exception as e:
        print(f"[Evictor] Error: {e}")

def run_test_generation():
    """Runs a generation with streaming to allow concurrent eviction."""
    print("[Client] Sending long-thought prompt (Streaming)...")
    prompt = """
    Provide a rigorous, measure-theoretic proof for the existence of a non-measurable set (Vitali set). 
    Then, explain the Banach-Tarski paradox as a consequence of this. 
    For every logical step you take, perform a self-critique to ensure there are no 
    contradictions with the Axiom of Choice. Think as deeply as possible and 
    provide at least 5000 words of internal reasoning.
    """
    
    payload = {
        "model": MODEL_PATH,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 3000,
        "temperature": 0.6,
        "stream": True,  # <--- CRITICAL: This prevents the call from blocking
        "request_id": "test",
    }
    
    try:
        response = requests.post(f"{BASE_URL}/v1/chat/completions", json=payload, stream=True)
        print("\n--- STARTING GENERATION STREAM ---")
        
        full_content = ""
        first_token_received = False
        for line in response.iter_lines():
            if line:
                if not first_token_received:
                    start_eviction_event.set()
                    first_token_received = True
                    
                decoded_line = line.decode('utf-8').lstrip("data: ")
                if decoded_line == "[DONE]":
                    break
                try:
                    chunk = json.loads(decoded_line)
                    delta = chunk['choices'][0]['delta'].get('content', '')
                    full_content += delta
                    # Print tokens in real-time so you can see when eviction hits
                    print(delta, end="", flush=True) 
                except json.JSONDecodeError:
                    pass
        
        print("\n--- STREAM FINISHED ---")
    except Exception as e:
        print(f"❌ Generation failed: {e}")

if __name__ == "__main__":
    server_proc = None
    try:
        # 1. Start Server
        server_proc = start_server()
        
        # 2. Wait for it to be ready
        wait_for_server()
        
        # 3. Start Eviction Thread
        evict_thread = threading.Thread(target=trigger_eviction)
        evict_thread.start()
        
        run_test_generation()
        
        evict_thread.join()

    except KeyboardInterrupt:
        print("\nStopping test...")
    finally:
        if server_proc:
            print("🛑 Shutting down vLLM server...")
            os.killpg(os.getpgid(server_proc.pid), signal.SIGTERM)
            print("Done.")