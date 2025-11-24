import numpy as np
from attention import scaled_dot_product_attention



def demonstrate_attention():
    
    print("=" * 80)
    print("SCALED DOT-PRODUCT ATTENTION DEMONSTRATION")
    print("=" * 80)
    
    # Example 1: Simple case with small matrices
    print("\n" + "=" * 80)
    print("EXAMPLE 1: Basic Attention (3 queries, 4 keys, dimension 2)")
    print("=" * 80)
    
    # Create simple Q, K, V matrices
    Q = np.array([
        [1.0, 0.0],  # Query 1
        [0.0, 1.0],  # Query 2
        [1.0, 1.0]   # Query 3
    ])
    
    K = np.array([
        [1.0, 0.0],  # Key 1
        [0.0, 1.0],  # Key 2
        [1.0, 1.0],  # Key 3
        [0.5, 0.5]   # Key 4
    ])
    
    V = np.array([
        [10.0, 0.0],  # Value 1
        [0.0, 10.0],  # Value 2
        [5.0, 5.0],   # Value 3
        [7.0, 3.0]    # Value 4
    ])
    
    print("\nQuery Matrix Q (3x2):")
    print(Q)
    print("\nKey Matrix K (4x2):")
    print(K)
    print("\nValue Matrix V (4x2):")
    print(V)
    
    output, weights = scaled_dot_product_attention(Q, K, V)
    
    print("\nAttention Weights (3x4):")
    print("(Each row sums to 1.0, showing probability distribution)")
    print(weights)
    print("\nRow sums (should all be 1.0):", weights.sum(axis=-1))
    
    print("\nAttention Output (3x2):")
    print("(Weighted combination of values based on query-key similarity)")
    print(output)
    
    # Example 2: With masking (causal/autoregressive attention)
    print("\n" + "=" * 80)
    print("EXAMPLE 2: Causal (Autoregressive) Masking")
    print("=" * 80)
    print("Prevent attending to future tokens")
    
    seq_len = 4
    d_k = 3
    
    # Random Q, K, V for this example
    np.random.seed(42)
    Q = np.random.randn(seq_len, d_k)
    K = np.random.randn(seq_len, d_k)
    V = np.random.randn(seq_len, d_k)
    
    # Create causal mask (lower triangular matrix)
    causal_mask = np.tril(np.ones((seq_len, seq_len)))
    
    print("\nCausal Mask (4x4):")
    print("(1 = allowed to attend, 0 = masked out)")
    print(causal_mask.astype(int))
    
    output_masked, weights_masked = scaled_dot_product_attention(Q, K, V, mask=causal_mask)
    
    print("\nAttention Weights WITH Causal Mask:")
    print("(Notice upper triangle is ~0, can't attend to future)")
    print(np.round(weights_masked, 4))
    
    # Compare with unmasked
    output_unmasked, weights_unmasked = scaled_dot_product_attention(Q, K, V)
    
    print("\nAttention Weights WITHOUT Mask:")
    print("(Can attend to all positions, including future)")
    print(np.round(weights_unmasked, 4))
    
if __name__ == "__main__":
    demonstrate_attention()