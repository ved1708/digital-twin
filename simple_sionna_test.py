#!/usr/bin/env python3
"""
Simple working Sionna test
"""

import tensorflow as tf
import sionna as sn
import numpy as np

print("🚀 SIONNA WORKING TEST")
print("=" * 40)

# Check setup
print(f"✓ Sionna: {sn.__version__}")
print(f"✓ TensorFlow: {tf.__version__}")
print(f"✓ GPUs: {len(tf.config.list_physical_devices('GPU'))}")

# Simple working example
print("\n📡 Testing Sionna Components:")

try:
    # 1. QAM Mapper
    mapper = sn.mapping.Mapper("qam", num_bits_per_symbol=4)
    demapper = sn.mapping.Demapper("app", "qam", num_bits_per_symbol=4)
    print("✓ QAM Mapper/Demapper created")
    
    # 2. AWGN Channel
    channel = sn.channel.AWGN()
    print("✓ AWGN Channel created")
    
    # 3. Simple simulation
    batch_size = 100
    num_bits = 1000
    
    # Generate random bits
    bits = tf.random.uniform([batch_size, num_bits], 0, 2, dtype=tf.int32)
    
    # Map to symbols
    symbols = mapper(bits)
    
    # Add noise
    noisy_symbols = channel([symbols, 10.0])  # 10 dB SNR
    
    print(f"✓ Processed {symbols.shape} symbols")
    print(f"✓ Symbol power: {tf.reduce_mean(tf.abs(symbols)**2):.3f}")
    
    print("\n🎉 Sionna is ready for RF simulations!")
    
except Exception as e:
    print(f"❌ Error: {e}")