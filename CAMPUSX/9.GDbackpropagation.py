"""
Gradient Descent in Backpropagation

Gradient Descent is the optimization algorithm used together with
Backpropagation to update the weights of a neural network so that the loss
reduces after every iteration.

-----------------------------------------------------------
1. WHY GRADIENT DESCENT?
-----------------------------------------------------------
Neural networks have millions of weights.
We need a systematic way to update these weights so that the model improves.

Gradient Descent provides:
    weight_new = weight_old - learning_rate * gradient

Where:
- gradient = derivative of loss w.r.t weight (computed using Backprop)
- learning_rate = how fast the model learns

The goal is to move in the direction of *steepest decrease* of the loss.

-----------------------------------------------------------
2. HOW GRADIENT DESCENT CONNECTS WITH BACKPROPAGATION
-----------------------------------------------------------
Backpropagation computes:
    ∂L/∂w1, ∂L/∂w2, ∂L/∂b1, ∂L/∂b2, ...

Gradient Descent uses these gradients to update the weights.

So the pipeline is:

Forward Pass → Compute Loss → Backward Pass (Gradients) → Gradient Descent Update

Backprop = compute gradients  
Gradient Descent = apply gradients to update weights  

-----------------------------------------------------------
3. GRADIENT DESCENT FORMULA
-----------------------------------------------------------
For any weight w:

    w_new = w_old - α * (∂L/∂w)

Where:
- α (alpha) = learning rate (e.g., 0.001, 0.01)
- ∂L/∂w = gradient given by Backpropagation

Same for bias:

    b_new = b_old - α * (∂L/∂b)

-----------------------------------------------------------
4. NUMERICAL EXAMPLE
-----------------------------------------------------------
Suppose:
    x = 2
    w = 3
    b = 1
    target t = 15
    learning_rate = 0.01

Forward pass:
    y = w*x + b = 3*2 + 1 = 7

Loss:
    L = (y - t)^2 = 64

Backprop (compute gradients):
    dL/dy = 2*(y - t) = 2*(-8) = -16
    dL/dw = dL/dy * x = -16 * 2 = -32
    dL/db = dL/dy = -16

Gradient Descent update:
    w_new = 3 - 0.01*(-32) = 3.32
    b_new = 1 - 0.01*(-16) = 1.16

Weights increased because the prediction was too low.

-----------------------------------------------------------
5. TYPES OF GRADIENT DESCENT
-----------------------------------------------------------
1. **Batch Gradient Descent**
   Uses the entire dataset to compute gradients.
   Stable but very slow.

2. **Stochastic Gradient Descent (SGD)**
   Uses 1 sample at a time.
   Fast but noisy updates.

3. **Mini-Batch Gradient Descent**
   Uses small batches (like 32/64 samples).
   Best trade-off → most commonly used.

-----------------------------------------------------------
6. MINI IMPLEMENTATION FROM SCRATCH
-----------------------------------------------------------

import numpy as np

# Data
x = np.array([2])
t = np.array([15])

# Initialize weight and bias
w = 3.0
b = 1.0
lr = 0.01

for epoch in range(10):
    # Forward pass
    y = w*x + b
    loss = (y - t)**2

    # Backpropagation (compute gradients)
    dL_dy = 2 * (y - t)
    dL_dw = dL_dy * x
    dL_db = dL_dy

    # Gradient Descent update
    w -= lr * dL_dw
    b -= lr * dL_db

    print(f"Epoch {epoch+1}: Loss = {loss[0]:.4f}, w = {w}, b = {b}")

-----------------------------------------------------------
7. SUMMARY
-----------------------------------------------------------
✓ Backpropagation computes gradients (∂L/∂w)  
✓ Gradient Descent updates weights using those gradients  
✓ Together they minimize the loss and help the model learn  
✓ Most deep learning optimizers (Adam, RMSProp, Adagrad) are improved versions of gradient descent  
✓ Without Gradient Descent, backpropagation alone cannot update weights  

"""
