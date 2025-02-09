// CUDA thread hierarchy
// BLOCK and THREAD <<<2, 3>>> 2 blocks and 3 threads per block
// GPU work in done in threads, which are organized into blocks.
// then blocks are organized into a grid.
// CUDA function is called kernels
// kernal launch with execution configuration <<<...>>> syntax


#include <stdio.h>

/*
 * Refactor firstParallel so that it can run on the GPU.
 */

__global__ void firstParallel()
{
  printf("This should be running in parallel.\n");
}

int main()
{
  /*
   * Refactor this call to firstParallel to execute in parallel
   * on the GPU.
   */

  firstParallel<<<5, 5>>>();
  
  /*
   * Some code is needed below so that the CPU will wait
   * for the GPU kernels to complete before proceeding.
   */
  cudaDeviceSynchronize();

}
