# Debug AI Code Helper

## 1. Tensor print debugging

When debugging tensor code, print:

- shape
- dtype
- device
- min / max / mean
- NaN status

This helps catch shape mismatch, wrong dtype, wrong device, and exploding values.

---

## 2. NaN loss

Common causes:

- learning rate too high
- division by zero
- log of zero
- exploding gradients
- invalid labels

First checks:

- print loss
- check outputs for NaN
- check gradients for NaN
- reduce learning rate
- reduce batch size

---

## 3. Shape mismatch

Print input and output shapes at each layer.

Use forward hooks to see:

input shape -> output shape

This is useful for neural networks with many layers.

---

## 4. Logging

Use logging instead of only print when a run takes longer.

Logs should include:

- loss
- learning rate
- step
- warning messages
- error messages

---

## 5. Timing and profiling

Use a Timer for quick checks.

Use cProfile when you need function-level timing.

Common bottlenecks:

- data loading
- preprocessing
- forward pass
- backward pass
- saving checkpoints

---

## 6. Memory debugging

Use tracemalloc for CPU memory.

For GPU memory, check:

torch.cuda.memory_summary()

If out of memory:

- reduce batch size
- delete unused tensors
- empty CUDA cache
- use mixed precision
- use gradient checkpointing

---

## 7. Debugging workflow

Before training:

- check shapes
- check dtypes
- check devices

First few steps:

- print loss
- check NaN
- inspect gradients

During training:

- log loss and learning rate
- use TensorBoard if needed

When broken:

- use breakpoint()
- inspect tensors interactively
