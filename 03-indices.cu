// CUDA-Provided Thread Hierarchy Variables
// gridDim.x is the number of blocks in the grid.
// CUDA does not support 4D grids or blocks. 
// CUDA grids and blocks are limited to three dimensions. 
// The built-in variables for grid dimensions are gridDim.x, gridDim.y, and gridDim.z,
// *****
// CUDA provide threadIdx.x and blockIdx.x variables to access the thread and block indices.
// blockIdx.x is the number of threads in a block.
// threadIdx.x is the index of the thread in its block.
// to access the thread and block indices.

#include <stdio.h>

__global__ void printSuccessForCorrectExecutionConfiguration()
{

  if(threadIdx.x == 1023 && blockIdx.x == 255)
  {
    printf("Success!\n");
  }
}

int main()
{
  /*
   * This is one possible execution context that will make
   * the kernel launch print its success message.
   */

  printSuccessForCorrectExecutionConfiguration<<<256, 1024>>>();

  /*
   * Don't forget kernel execution is asynchronous and you must
   * sync on its completion.
   */

  cudaDeviceSynchronize();
}
// !nvcc -o indices 03-indices.cu -run