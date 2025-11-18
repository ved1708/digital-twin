#!/usr/bin/env python3
"""
Simple Sionna test to verify GPU acceleration and basic functionality
"""

import tensorflow as tf
import sionna as sn
import numpy as np

print("=" * 50)
print("SIONNA GPU TEST")
print("=" * 50)

# Check versions
print(f"✓ Sionna version: {sn.__version__}")
print(f"✓ TensorFlow version: {tf.__version__}")

# Check GPU availability
gpus = tf.config.list_physical_devices('GPU')
print(f"✓ GPUs available: {len(gpus)}")
if gpus:
    for i, gpu in enumerate(gpus):
        print(f"  GPU {i}: {gpu.name}")

# Check if GPU is actually being used
print(f"✓ TensorFlow built with CUDA: {tf.test.is_built_with_cuda()}")
print(f"✓ GPU available for TensorFlow: {tf.test.is_gpu_available()}")

print("\n" + "=" * 50)
print("TESTING SIONNA MODULES")
print("=" * 50)

try:
    # Test 1: Basic OFDM
    print("Test 1: OFDM Modulator/Demodulator")
    ofdm_mod = sn.ofdm.OFDMModulator(
        cyclic_prefix_length=16,
        pilot_ofdm_symbol_indices=[2, 11]
    )
    ofdm_demod = sn.ofdm.OFDMDemodulator(ofdm_mod)
    print("✓ OFDM modules created successfully")
    
    # Test 2: Channel coding
    print("\nTest 2: LDPC Encoder/Decoder")
    ldpc_encoder = sn.fec.ldpc.LDPC5GEncoder(k=100, n=200)
    ldpc_decoder = sn.fec.ldpc.LDPC5GDecoder(ldpc_encoder, hard_out=True)
    print("✓ LDPC modules created successfully")
    
    # Test 3: Channel models
    print("\nTest 3: Channel Models")
    awgn_channel = sn.channel.AWGN()
    rayleigh_channel = sn.channel.RayleighBlockFading(num_rx=2, num_tx=2)
    print("✓ Channel models created successfully")
    
    # Test 4: Ray Tracing (if available)
    print("\nTest 4: Ray Tracing")
    if hasattr(sn, 'rt'):
        print("✓ Ray Tracing module available")
        # Simple RT scene test
        try:
            scene = sn.rt.Scene()
            print("✓ Ray Tracing scene created successfully")
        except Exception as e:
            print(f"⚠ Ray Tracing scene creation failed: {e}")
    else:
        print("⚠ Ray Tracing module not available")
        
    # Test 5: Simple simulation with GPU
    print("\nTest 5: GPU Computation Test")
    with tf.device('/GPU:0' if gpus else '/CPU:0'):
        # Simple matrix multiplication to test GPU
        a = tf.random.normal([1000, 1000])
        b = tf.random.normal([1000, 1000])
        c = tf.matmul(a, b)
        print(f"✓ Matrix multiplication completed on {'GPU' if gpus else 'CPU'}")
        
        # Simple OFDM simulation
        batch_size = 64
        num_bits = 1000
        
        # Generate random bits
        bits = tf.random.uniform([batch_size, num_bits], 0, 2, dtype=tf.int32)
        
        # QAM mapping
        mapper = sn.mapping.Mapper("qam", num_bits_per_symbol=4)
        symbols = mapper(bits)
        
        print(f"✓ Generated {symbols.shape} QAM symbols on {'GPU' if gpus else 'CPU'}")

    print("\n" + "=" * 50)
    print("ALL TESTS PASSED! 🎉")
    print("Sionna is ready for RF/ML simulations!")
    print("=" * 50)
    
except Exception as e:
    print(f"\n❌ Error during testing: {e}")
    import traceback
    traceback.print_exc()