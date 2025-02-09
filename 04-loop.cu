// Exercise: Accelerating a For Loop with a Single Block of Threads
#include <stdio.h>

/*
 * Refactor `loop` to be a CUDA Kernel. The new kernel should
 * only do the work of 1 iteration of the original loop.
 */

__global__ void loop()
{
    // piggback on the built-in threadIdx.x
    printf("This is iteration number %d\n", threadIdx.x);
}

int main()
{
  /*
   * When refactoring `loop` to launch as a kernel, be sure
   * to use the execution configuration to control how many
   * "iterations" to perform.
   *
   * For this exercise, only use 1 block of threads.
   */
   loop<<<1, 10>>>();
   cudaDeviceSynchronize();
}
//up 1*10 vs down 2* 5
//each thread must be mapped to work on element in the vector.
// Exercise: Accelerating a For Loop with Multiple Blocks of Threads

#include <stdio.h>

__global__ void loop()
{
  /*
   * This idiomatic expression gives each thread
   * a unique index within the entire grid.
   */
  printf("blockIdx.x: %d, threadIdx.x: %d, Calculated Thread Index: %d\n", blockIdx.x, threadIdx.x, blockIdx.x * blockDim.x + threadIdx.x);
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  printf("%d\n", i);
}

int main()
{
  /*
   * Additional execution configurations that would
   * work and meet the exercises contraints are:
   *
   * <<<5, 2>>>
   * <<<10, 1>>>
   */

  loop<<<2, 5>>>();
  cudaDeviceSynchronize();
}
