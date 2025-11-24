import numpy as np

def scaled_dot_product_attention(Q, K, V, mask=None):
    """    
    Formula: Attention(Q, K, V) = softmax(Q·K^T / sqrt(d_k))·V

    mask : numpy.ndarray, optional
        Mask of shape (..., seq_len_q, seq_len_k) or broadcastable to that shape.
        Elements where mask == 0 will be set to -inf before softmax.
    """
    #dimension of single keys/query
    d_k = K.shape[-1]
    
    # Step 2: Compute attention scores (QK^T)/ sqrt(d_k)
    # Swaps the last two dims of k; to match with the dimension of Q for matrix multiplication
    attention_scores = np.matmul(Q, K.transpose(*range(K.ndim - 2), -1, -2))
    attention_scores = attention_scores / np.sqrt(d_k)
    
    #Apply mask (if provided)
    if mask is not None:
        attention_scores = np.where(mask == 0, -1e9, attention_scores)
    
    # Subtract max for stability(prevents overflow in exp)
    attention_scores_max = np.max(attention_scores, axis=-1, keepdims=True)
    exp_scores = np.exp(attention_scores - attention_scores_max)
    # Compute softmax probabilities
    attention_weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)
    
    output = np.matmul(attention_weights, V)
    
    return output, attention_weights
