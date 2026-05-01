# Tiny Recursion Models (TRM): Technical Specification & Implementation Guide

[cite_start]This document provides a comprehensive technical overview of **Tiny Recursion Models (TRM)**, as introduced in *"Less is More: Recursive Reasoning with Tiny Networks"* by Alexia Jolicoeur-Martineau[cite: 1]. [cite_start]This guide is designed to provide a coding agent with the full context, mathematical logic, and pseudocode necessary to implement the TRM architecture, specifically the **Self-Attention variant** used for solving the ARC-AGI dataset[cite: 4, 171].

---

## 1. Core Philosophy: "Less is More"
[cite_start]TRM is a parameter-efficient alternative to Hierarchical Reasoning Models (HRM) and Large Language Models (LLMs)[cite: 4, 104]. [cite_start]While LLMs struggle with hard puzzles due to the autoregressive nature of token generation, TRM uses **recursive reasoning** and **deep supervision** to iteratively refine a predicted answer[cite: 7, 102].

### Key Advantages:
* [cite_start]**Efficiency**: Achieves state-of-the-art results on ARC-AGI with only ~7M parameters (less than 0.01% the size of modern LLMs)[cite: 5].
* [cite_start]**Depth Emulation**: By recursing a 2-layer network multiple times, it emulates a very deep network (hundreds of layers) without the memory cost of full backpropagation[cite: 16, 64].
* [cite_start]**Simplicity**: Unlike its predecessor (HRM), TRM requires no complex biological justifications or fixed-point theorems[cite: 103, 211].

---

## 2. Architecture Overview (Self-Attention Variant)

The TRM architecture consists of a single "tiny" network that processes three primary components:
1.  [cite_start]**Input ($x$)**: The embedded question/puzzle[cite: 13, 130].
2.  [cite_start]**Prediction ($y$)**: The current state of the embedded answer[cite: 13, 125].
3.  [cite_start]**Latent ($z$)**: A latent reasoning feature that acts as an internal "scratchpad" or chain-of-thought[cite: 13, 134].

### The Tiny Network Block
[cite_start]For ARC-AGI, the network uses a **2-layer Transformer-style architecture**[cite: 4, 157]:
* [cite_start]**Components**: Self-Attention, Multi-Layer Perceptron (MLP), and RMSNorm[cite: 40, 166].
* [cite_start]**Activation**: SwiGLU[cite: 40].
* [cite_start]**Embeddings**: Rotary Embeddings (RoPE); no bias terms in the layers[cite: 40].
* [cite_start]**Structure**: `Input -> Self-Attention -> Add & Norm -> MLP -> Add & Norm` (Repeated 2x)[cite: 12, 157].

---

## 3. Recursive Reasoning Algorithm

[cite_start]TRM operates through two nested loops: **Deep Supervision** (outer loop) and **Recursive Reasoning** (inner loop)[cite: 14, 15].

### 3.1 Latent Recursion (The Inner Loop)
[cite_start]This function updates the reasoning state ($z$) and the prediction ($y$) multiple times[cite: 100].

```python
def latent_recursion(x, y, z, n=6):
    """
    Performs n latent reasoning steps followed by an answer refinement step.
    """
    for i in range(n):
        # Update the reasoning latent z based on question, current answer, and previous latent
        z = net(x, y, z) 
    
    # Update the prediction y using the updated reasoning z
    y = net(y, z) 
    
    return y, z
```

### 3.2 Deep Recursion (Training Optimization)
[cite_start]To save memory, TRM runs $T-1$ recursions without tracking gradients, and only the final recursion is backpropagated[cite: 101, 115].

```python
def deep_recursion(x, y, z, n=6, T=3):
    """
    Emulates massive depth by running multiple recursion passes.
    """
    # T-1 passes: Improve y and z WITHOUT gradients to save memory
    with torch.no_grad():
        for j in range(T - 1):
            y, z = latent_recursion(x, y, z, n)
            
    # Final pass: One recursion WITH gradients for optimization
    y, z = latent_recursion(x, y, z, n)
    
    return (y.detach(), z.detach()), output_head(y), Q_head(y)
```

---

## 4. Training with Deep Supervision & ACT

[cite_start]TRM uses **Deep Supervision** to ensure the model improves at every step[cite: 114]. [cite_start]It also employs **Adaptive Computational Time (ACT)** to stop early if a solution is found[cite: 101].

### Implementation Details:
* [cite_start]**Max Supervision Steps ($N_{sup}$)**: 16[cite: 55].
* [cite_start]**Loss Function**: Softmax Cross-Entropy for the prediction + Binary Cross-Entropy for the halting probability ($q$)[cite: 101, 174].
* [cite_start]**Halting**: The model learns a halting probability $q$ based on whether the current $y$ matches the true label[cite: 101, 174].

### Main Training Loop:
```python
# Initialization
y, z = y_init, z_init 

for step in range(N_supervision):
    x = input_embedding(x_input)
    
    # Run recursive reasoning
    (y, z), y_hat, q_hat = deep_recursion(x, y, z)
    
    # Standard prediction loss
    loss = softmax_cross_entropy(y_hat, y_true)
    
    # ACT Halting loss: Learn to predict if y_hat == y_true
    loss += binary_cross_entropy(q_hat, (y_hat == y_true))
    
    # Backpropagation
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    
    # Early-stopping if the halting head suggests completion
    if q_hat > 0: 
        break
```

---

## 5. Implementation Specifications for ARC-AGI

[cite_start]To implement the specific variant used for the ARC-AGI dataset, follow these parameters[cite: 187, 269]:

| Parameter | Value |
| :--- | :--- |
| **Network Layers** | 2 layers |
| **Hidden Size ($D$)** | 512 |
| **Batch Size** | 768 |
| **Optimizer** | AdamW ($\beta_1=0.9, \beta_2=0.95$) |
| **Learning Rate** | 1e-4 (1e-2 for embeddings) |
| **Recursions ($n, T$)** | $n=6, T=3$ |
| **Max Supervision ($N_{sup}$)** | 16 |
| **Normalization** | RMSNorm |
| **EMA** | Exponential Moving Average of weights (0.999) |
| **Data Augmentation** | Color permutation, Dihedral-group, and translations |

### The Role of $y$ and $z$:
* [cite_start]**$y$ (formerly $z_H$)**: Represents the **embedded solution**[cite: 125]. [cite_start]Reversing this embedding (via output head + argmax) yields the actual tokens of the answer[cite: 126, 305].
* [cite_start]**$z$ (formerly $z_L$)**: Represents the **latent reasoning**[cite: 127]. [cite_start]It cannot be decoded into a sensible output directly; it serves as the "engine" that transforms the current $y$ into a better one[cite: 131, 305].

### Attention vs. MLP:
* [cite_start]For ARC-AGI (30x30 grids), the **Self-Attention** layer is mandatory to handle the larger, variable context[cite: 171, 204]. [cite_start]The MLP-Mixer variant is only recommended for small, fixed-length tasks like 9x9 Sudoku[cite: 170].

---

## 6. Summary for the Coding Agent
1.  [cite_start]**Single Model**: Use one 2-layer Transformer block for both updating $z$ and updating $y$[cite: 152, 153]. [cite_start]The difference in task is determined by whether the input $x$ is included[cite: 151].
2.  [cite_start]**No IFT**: Do not use Implicit Function Theorem or 1-step gradient approximations[cite: 118, 211]. [cite_start]Backpropagate through the entire final recursion cycle[cite: 112].
3.  [cite_start]**Memory Management**: Use `torch.no_grad()` for the first $T-1$ cycles to allow for higher effective depth without OOM (Out of Memory) errors[cite: 119, 183].
4.  [cite_start]**Stability**: Implement **EMA** on weights and use **stable-max loss** to prevent divergence during training on small datasets[cite: 178, 269].