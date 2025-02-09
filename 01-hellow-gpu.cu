// !nvcc -o hello-gpu 01-hello/01-hello-gpu.cu -run
#include <stdio.h>

void helloCPU()
{
  printf("Hello from the CPU.\n");
}

/*
 * The addition of `__global__` signifies that this function
 * should be launced on the GPU.
 * no _global_: error: a host function call cannot be configured
 */

__global__ void helloGPU()
{
  printf("Hello from the GPU longwei.\n");
}

__global__ void helloGPU2()
{
  printf("Hello from the GPU longwei2.\n");
}

int main()
{
  helloCPU();


  /*
   * Add an execution configuration with the <<<...>>> syntax
   * will launch this function as a kernel on the GPU. 
   * <<<2,2>>> means 2 blocks and 2 threads per block, 4 print output
   */

  helloGPU<<<1, 1>>>();

  /*
   * `cudaDeviceSynchronize` will block the CPU stream until
   * all GPU kernels have completed.  
   * It wait for GPU work. If removed, only CPU output will be printed.
   */

  cudaDeviceSynchronize();
  
  
  helloGPU2<<<1, 1>>>();
  cudaDeviceSynchronize();
}
