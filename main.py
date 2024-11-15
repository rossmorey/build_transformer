import numpy as np
import torch

def compute_attention_scores(query, key, value, mask=None):
    """
    Compute scaled dot-product attention scores.


    Parameters:
    - query: Tensor of shape (batch_size, num_heads, seq_length, d_k)
    - key: Tensor of shape (batch_size, num_heads, seq_length, d_k)
    - value: Tensor of shape (batch_size, num_heads, seq_length, d_v)
    - mask: Optional mask tensor


    Returns:
    - attention_output: Weighted sum of values
    - attention_weights: Attention probability distribution
    """
    # Compute the dot product of query and key:
    scores = np.dot(query, key)


    # Scale the scores:
    d_k = key.shape[-1]
    scaled_scores = scores / np.sqrt(d_k)


    # Apply mask if provided
    if mask is not None:
        scaled_scores += (mask * -np.inf)  # Assuming mask is 0 for valid positions and 1 for masked positions


    # Apply softmax to get attention weights
    attention_weights = np.exp(scaled_scores) / np.sum(np.exp(scaled_scores), axis=-1, keepdims=True)


    # Compute weighted sum of values
    attention_output = np.matmul(attention_weights, value)

    return attention_output, attention_weights

def positional_encoding(seq_length, d_model):
    """
    Creates a pattern of numbers that encodes position information.


    Parameters:
    - seq_length: How many items are in your input sequence
    - d_model: The size of your model's working space


    Returns:
    - pos_encoding: A matrix of size (seq_length × d_model) containing the position patterns
    """
    # Step 1: Create an empty matrix to store the position patterns
    empty_matrix = np.arange(seq_length)


    # Step 2: Calculate the position numbers and division terms
    position = empty_matrix[:, np.newaxis]
    div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))


    # Step 3: Fill the matrix with sine and cosine patterns
    # Your code here
    pos_encoding = np.zeros((seq_length, d_model))
    pos_encoding[:, 0::2] = np.sin(position * div_term)
    pos_encoding[:, 1::2] = np.cos(position * div_term)

    return torch.FloatTensor(pos_encoding)