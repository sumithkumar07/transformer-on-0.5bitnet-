import numpy as np

def generate_sovereign_data(filepath, size_mb=100):
    print(f"[SYSTEM]: Generating {size_mb}MB Sovereign Dataset...")
    
    # 100MB = 100 * 1024 * 1024 bytes
    total_bytes = size_mb * 1024 * 1024
    
    # Create structured patterns to simulate "Technical/Code" logic
    # - Fibonacci sequences
    # - Prime-based noise
    # - Periodic delimiters (mimicking brackets/spaces)
    
    data = bytearray(total_bytes)
    
    # Fill with a base repeating pattern (Technical Structure)
    pattern = b"SOVEREIGN_V13_AUTONOMOUS_ENTITY_READY_" 
    p_len = len(pattern)
    for i in range(0, total_bytes, p_len):
        chunk = pattern[:min(p_len, total_bytes - i)]
        data[i:i+len(chunk)] = chunk
        
    # Inject "Mathematical Logic" every 1KB
    for i in range(0, total_bytes, 1024):
        # Insert a local "gradient" of numbers
        for j in range(16):
            if i + j < total_bytes:
                data[i + j] = (i // 1024 + j) % 256
                
    with open(filepath, "wb") as f:
        f.write(data)
    
    print(f"[SUCCESS]: Dataset saved to {filepath} ({total_bytes} bytes).")

if __name__ == "__main__":
    generate_sovereign_data("sovereign_100.bin")
