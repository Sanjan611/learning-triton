# GPU Fundamentals

Learning Triton becomes easier once you're able to comprehend what's happening underneath the hood on the hardware. It's also useful to know how the GPU works.

We will go into enough depth that better facilitates learning about Triton.

## CPU vs GPU: The Core Difference

A CPU has a few powerful cores optimized for sequential tasks. A GPU has thousands of simpler cores optimized for doing the same operation across many data elements simultaneously.

```
CPU (e.g. 16 cores):                GPU (e.g. thousands of cores):
┌──────┐ ┌──────┐                   ┌─┐┌─┐┌─┐┌─┐┌─┐┌─┐┌─┐┌─┐┌─┐┌─┐┌─┐┌─┐
│ Big  │ │ Big  │  ...              │·││·││·││·││·││·││·││·││·││·││·││·│ ...
│ Core │ │ Core │                   └─┘└─┘└─┘└─┘└─┘└─┘└─┘└─┘└─┘└─┘└─┘└─┘
└──────┘ └──────┘                   ┌─┐┌─┐┌─┐┌─┐┌─┐┌─┐┌─┐┌─┐┌─┐┌─┐┌─┐┌─┐
Few cores, each very fast           │·││·││·││·││·││·││·││·││·││·││·││·│ ...
                                    └─┘└─┘└─┘└─┘└─┘└─┘└─┘└─┘└─┘└─┘└─┘└─┘
                                    Many cores, each simpler
```

This is why GPUs excel at tasks like "add 10 to every element in a million-element vector" — each core handles a few elements, and they all work at the same time.

## GPU Architecture

### CUDA Cores

A **CUDA core** (NVIDIA's term) is the basic processing unit. Each core can execute one floating-point or integer operation per clock cycle. A modern GPU has thousands of these.

Think of CUDA cores as the workers on an assembly line — individually simple, but powerful in aggregate.

### Streaming Multiprocessors (SMs)

CUDA cores are grouped into **Streaming Multiprocessors (SMs)**. An SM is a self-contained processing unit that has:

- A set of CUDA cores (e.g. 64 FP32 cores on an A100)
- Its own shared memory (fast, on-chip)
- Its own registers
- A warp scheduler

```
GPU
├── SM 0
│   ├── CUDA Cores (64)
│   ├── Shared Memory (164 KB on A100)
│   ├── Registers (65536 per SM)
│   └── Warp Schedulers
├── SM 1
│   ├── ...
├── ...
└── SM 107 (A100 has 108 SMs)
```

### Warps

A **warp** is a group of 32 threads that execute the same instruction at the same time (SIMT — Single Instruction, Multiple Threads). This is the fundamental unit of execution on a GPU.

When an SM runs a program, it breaks the work into warps of 32 threads. All 32 threads in a warp execute in lockstep — they run the same instruction on different data.

```
One Warp (32 threads):
┌──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┐
│T0│T1│T2│T3│T4│T5│T6│T7│T8│T9│..│..│..│..│..│T31│
└──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴───┘
         All execute the SAME instruction
         on DIFFERENT data elements
```

Warps explain why block sizes should be multiples of 32 — it maps cleanly to the hardware, avoiding wasted threads in partially filled warps.

### Thread Blocks (CUDA Blocks)

A **thread block** is a group of threads (one or more warps) that are scheduled together onto a single SM. Threads within a block can cooperate via shared memory and synchronization.

For example, a thread block of 256 threads contains 256 / 32 = 8 warps.

## Memory Hierarchy

The GPU has several levels of memory, each with different speed and size tradeoffs. Understanding this hierarchy is key to writing fast GPU code.

### Global Memory (HBM)

- **Size**: Large (40–80 GB on A100)
- **Speed**: Slow relative to on-chip memory (~2 TB/s bandwidth on A100)
- **Scope**: Accessible by all SMs

This is where your data (tensors, arrays) lives. When a kernel reads or writes data, it accesses global memory.

### Shared Memory (SMEM)

- **Size**: Small (up to 164 KB per SM on A100)
- **Speed**: Fast (~19 TB/s on A100)
- **Scope**: Shared among all threads within one SM

Shared memory is on-chip and much faster than global memory. It's used for communication between threads in the same block and for caching data that will be reused.

### Registers

- **Size**: Tiny (per thread, ~255 registers on modern GPUs)
- **Speed**: Fastest
- **Scope**: Private to each thread

Registers hold the values you're actively computing with. When data is loaded, it ends up in registers (possibly via shared memory). All arithmetic happens on register values.

### L1 / L2 Cache

- **L1**: On each SM, often unified with shared memory
- **L2**: Shared across all SMs (e.g. 40 MB on A100)
- **Managed by hardware** — you don't control these directly

```
Speed:    Registers > Shared Memory > L1/L2 Cache > Global Memory (HBM)
Size:     Registers < Shared Memory < L1/L2 Cache < Global Memory (HBM)
```

## Execution Model: How Work Gets Scheduled

When you launch a kernel with a grid of N thread blocks:

1. The GPU receives N **thread blocks**
2. The hardware scheduler assigns thread blocks to SMs
3. Each SM can run multiple thread blocks concurrently (if it has enough resources)
4. Within each SM, thread blocks are split into warps of 32 threads
5. The warp scheduler interleaves warps to hide memory latency

```
Kernel launch (grid = 8 thread blocks)
    │
    ▼
┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐
│  SM 0    │  │  SM 1    │  │  SM 2    │  │  SM 3    │
│          │  │          │  │          │  │          │
│ Block 0  │  │ Block 2  │  │ Block 4  │  │ Block 6  │
│ Block 1  │  │ Block 3  │  │ Block 5  │  │ Block 7  │
└──────────┘  └──────────┘  └──────────┘  └──────────┘
```

**Latency hiding**: When a warp is waiting for a slow memory load, the SM switches to another warp that's ready to compute. This is why GPUs need many threads — to keep the cores busy while others wait for data.

## Summary

| Concept | What It Is |
|---|---|
| CUDA Core | Basic processing unit; executes one operation per clock |
| SM | Group of cores + shared memory + registers + warp scheduler |
| Warp | 32 threads executing the same instruction in lockstep |
| Thread Block | Group of warps scheduled onto a single SM |
| Global Memory (HBM) | Large, slow; where data lives |
| Shared Memory | Small, fast; on-chip, shared within an SM |
| Registers | Tiny, fastest; private to each thread |

The key insight: GPU performance is about keeping data in fast memory (registers, shared memory) and minimizing trips to slow global memory. Everything else — tiling, fusion, block sizing — follows from this principle.

## Resources

- https://medium.com/ai-insights-cobet/understanding-gpu-architecture-basics-and-key-concepts-40412432812b
- https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html — NVIDIA CUDA Programming Guide (Sections on hardware implementation, memory hierarchy)
- https://developer.nvidia.com/blog/cuda-refresher-cuda-programming-model/ — NVIDIA CUDA Refresher: The CUDA Programming Model
- https://www.nvidia.com/en-us/data-center/a100/ — NVIDIA A100 specifications (SM count, memory bandwidth, SRAM size)
