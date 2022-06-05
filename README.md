 ## Links
 
 - https://cs.nyu.edu/manycores/cuda_many_cores.pdf
 - https://www.youtube.com/watch?v=SrAMBi_8tIk&list=PLFtDTZgdIZy5n8LWmhTic07zTN3hJQVfy
 - https://www.youtube.com/watch?v=lGefnd7Fmmo&list=PLFtDTZgdIZy5n8LWmhTic07zTN3hJQVfy&index=4
 - https://nyu-cds.github.io/python-gpu/02-cuda/
 - https://en.wikipedia.org/wiki/Thread_block_(CUDA_programming)
 - https://developer.nvidia.com/blog/cuda-refresher-cuda-programming-model/
 - https://stackoverflow.com/a/10467342

## Terminology

- *CUDA* is Nvidia's proprietary API to execute code on their GPUs.
- *kernels* are CUDA programs that run on Nvidia GPUs.
- CPUs are *hosts*, GPUs are *devices*.
- Blocks contain threads. Threads in a block can have 1d, 2d, or 3d indexes.
- Grids contain blocks. Blocks in a grid can have 1d, 2d, or 3d indexes.
- Streaming multiprocessors (SM) are made up of cores.
- Warps always have 32 threads. A block is executed as several warps.
- A warp consists of lanes.
- The GPU is responsible for allocating thread blocks to SMs.

## Example
- I have an Nvidia GeForce GTX 1660 Ti.
- I learned this by running `nvidia-smi --query-gpu=name --format=csv`.
- Article on this particular GPU: https://www.nvidia.com/en-us/geforce/news/geforce-gtx-1660-ti-advanced-shaders-streaming-multiprocessor/
  - Turing architecture.
  - This GPU has 24 streaming multiprocessors and 1536 cores total (so 64 cores per SM).
