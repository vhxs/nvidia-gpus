 ## Links
 
 - https://cs.nyu.edu/manycores/cuda_many_cores.pdf
 - https://www.youtube.com/watch?v=SrAMBi_8tIk&list=PLFtDTZgdIZy5n8LWmhTic07zTN3hJQVfy
 - https://www.youtube.com/watch?v=lGefnd7Fmmo&list=PLFtDTZgdIZy5n8LWmhTic07zTN3hJQVfy&index=4
 - https://nyu-cds.github.io/python-gpu/02-cuda/
 - https://en.wikipedia.org/wiki/Thread_block_(CUDA_programming)

## Terminology

- *CUDA* is Nvidia's proprietary API to execute code on their GPUs.
- *kernels* are CUDA programs that run on Nvidia GPUs.
- Blocks contain threads.
- Grids contain blocks.
- Streaming multiprocessors (SM) are made up of cores.
- Warps always have 32 threads. A block is executed as several warps.
- A thread block is assigned to run on an SM.
