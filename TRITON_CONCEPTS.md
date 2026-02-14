# Triton Concepts

Triton is an open-source Python-like programming language which enables researchers with no CUDA experience to write highly efficient GPU code.

## The Core Idea: Programs, Not Threads

In CUDA, you think about thousands of individual threads that each process one element. In Triton, you think about **programs** that each process a **block** of elements.

A Triton kernel is a function that gets launched many times in parallel. Each launch is a **program instance**, and each instance processes a different chunk of data.

## Terminology

| Term | Definition |
|---|---|
| Grid | The total number of program instances you launch. If your vector has 1024 elements and your block size is 256, your grid has 4 programs. |
| Block size (B0, B1, ...) | How many elements each program processes along a given axis. This is a compile-time constant (declared as `tl.constexpr`). |
| Program ID (pid) | A unique integer identifying which program instance is running. In a 1D grid, pid ranges from 0 to grid_size - 1. |
| Offsets | The actual indices into data that this program will access. Computed from the program ID and block size using `tl.arange`. |
| Mask | A boolean tensor that prevents out-of-bounds memory access when data length isn't perfectly divisible by block size. |

## The Three Key Questions

Every Triton kernel answers three questions:

1. **Which program am I?** → `tl.program_id(axis)`
2. **What data should I work on?** → Compute offsets from the program ID
3. **What should I do with that data?** → Load, compute, store

## Anatomy of a Kernel

```
                      THE GRID
    ┌──────────┬──────────┬──────────┬──────────┐
    │Program 0 │Program 1 │Program 2 │Program 3 │
    │processes │processes │processes │processes │
    │block 0   │block 1   │block 2   │block 3   │
    └──────────┴──────────┴──────────┴──────────┘

    Data: [████████|████████|████████|████████]
           block 0   block 1   block 2   block 3
```

Each program instance:

- Gets a unique ID via `tl.program_id(axis=0)` → returns 0, 1, 2, or 3
- Computes which chunk of data it owns
- Loads that chunk, does math, stores the result

## 1D Example: One Program Axis

Vector of length 8, block size 4 → grid of 2 programs:

```
Program 0 (pid=0):                Program 1 (pid=1):
  offsets = [0, 1, 2, 3]           offsets = [4, 5, 6, 7]
  loads data[0:4]                  loads data[4:8]
  computes on it                   computes on it
  stores result[0:4]               stores result[4:8]
```

In code:
```python
pid = tl.program_id(axis=0)           # 0 or 1
offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)  # [0,1,2,3] or [4,5,6,7]
data = tl.load(x_ptr + offsets)        # load 4 elements
result = data + 10                     # compute
tl.store(out_ptr + offsets, result)     # store 4 elements
```

## 2D Example: Two Program Axes

For 2D operations (like outer products or matrix multiply), you use two program axes:

```
                    N (columns)
              ┌────────┬────────┐
              │pid_n=0 │pid_n=1 │
    ┌─────────┼────────┼────────┤
    │pid_m=0  │Block   │Block   │
M   │         │(0,0)   │(0,1)   │
(rows)├─────────┼────────┼────────┤
    │pid_m=1  │Block   │Block   │
    │         │(1,0)   │(1,1)   │
    └─────────┴────────┴────────┘
```

Each program works on one tile of the output. Program IDs along two axes determine which tile.

## The Data Flow

Every Triton program follows the same pattern:

```
    GPU Global Memory (HBM)
    ════════════════════════
            │ tl.load()
            ▼
    ┌─────────────────┐
    │  Registers/SRAM  │  ← Fast! Compute happens here
    │  (per program)   │
    └─────────────────┘
            │ tl.store()
            ▼
    GPU Global Memory (HBM)
    ════════════════════════
```

The key performance insight: `tl.load` and `tl.store` are *expensive* (they access slow global memory). The computation between them is *cheap*. So the goal is to load once, do as much work as possible, then store once. This is what "kernel fusion" means.

## tl.arange

`tl.arange(0, BLOCK_SIZE)` creates a tensor of consecutive integers: `[0, 1, 2, ..., BLOCK_SIZE-1]`. This is how you generate the offsets for loading a block of data.

When combined with broadcasting (adding a column vector to a row vector), you can create 2D offset grids — which is how you handle 2D data.

```python
# 1D offsets
offsets = tl.arange(0, B0)                          # shape: [B0]

# 2D offsets (for a 2D block)
row_offsets = tl.arange(0, B0)[:, None]              # shape: [B0, 1]
col_offsets = tl.arange(0, B1)[None, :]              # shape: [1, B1]
offsets_2d = row_offsets * stride + col_offsets       # shape: [B0, B1]
```

## Operations

For a beginner, it's useful to know the category of operations that Triton offers.

1. Creation
2. Shape Manipulation
3. Linear Algebra
4. Memory/Pointer
5. Indexing
6. Math
7. Reduction
8. Scan/Sort
9. Atomic
10. Random Number Generation
11. Iterators
12. Compiler Hint
13. Debug

## Code Examples

Don't worry about the specific syntax or decorators here. Get a sense of the flow of work here.
```
# ============================================================
# Example 1: Vector Add (the "hello world" of Triton)
# ============================================================

@triton.jit
def vector_add_kernel(
    x_ptr, y_ptr, output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Each program adds one block of x and y."""
    # Which program am I?
    pid = tl.program_id(axis=0)

    # What elements do I process?
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # Don't go out of bounds
    mask = offsets < n_elements

    # Load, compute, store
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)
```

## Memory and Performance

When you call `tl.load`, data travels from HBM → through the cache hierarchy → into registers. When you call `tl.store`, it goes back. Everything in between (your computation) happens in registers, which is extremely fast.

**The takeaway**: Minimize loads and stores. Maximize computation between them. This is the entire motivation for kernel fusion.

### Memory-Bound vs Compute-Bound

An operation is **memory-bound** when it spends more time loading/storing data than actually computing. Most element-wise operations (ReLU, add, dropout) are memory-bound because they do ~1 operation per element loaded.

An operation is **compute-bound** when the math takes longer than the data transfer. Matrix multiplication is compute-bound because for an `N×N` matmul, you load `O(N²)` data but do `O(N³)` operations.

```
A100 GPU:
  Compute: 312 TFLOPS (FP16)
  Memory:  2 TB/s

  To be compute-bound, you need:
  312 TFLOPS / 2 TB/s = 156 FLOPs per byte loaded

  Element-wise op:  ~1 FLOP/byte  → Memory-bound (way below 156)
  Matrix multiply:  ~N FLOPs/byte → Compute-bound for large N
```

### Why Fusion Matters

Consider computing `relu(x + y)` in PyTorch:

```
Without fusion (2 separate kernels):
  Kernel 1: Load x, Load y → Compute x+y → Store z     (2 loads, 1 store)
  Kernel 2: Load z → Compute relu(z) → Store result     (1 load, 1 store)
  Total: 3 loads + 2 stores from/to HBM

With fusion (1 Triton kernel):
  Load x, Load y → Compute relu(x+y) → Store result     (2 loads, 1 store)
  Total: 2 loads + 1 store from/to HBM
```

That's a 40% reduction in memory traffic for this simple example. For longer chains of operations, the savings compound dramatically.

### Tiling / Blocking

The idea of processing data in blocks (tiles) appears throughout GPU programming. Instead of processing one element at a time or the entire dataset at once, you process a **block** that fits in fast memory.

```
Full matrix (too big for SRAM):
┌──────────────────────┐
│                      │
│                      │
│                      │
│                      │
└──────────────────────┘

Tiled processing (block fits in SRAM):
┌─────┬─────┬─────┬───┐
│Tile │Tile │Tile │...│  ← Process one tile at a time
│ 0,0 │ 0,1 │ 0,2 │   │
├─────┼─────┼─────┼───┤
│Tile │Tile │Tile │...│
│ 1,0 │ 1,1 │ 1,2 │   │
└─────┴─────┴─────┴───┘
```

In Triton, your block size IS your tile size. Each program instance processes one tile.

## References

- https://triton-lang.org/main/python-api/triton.language.html#triton-language
