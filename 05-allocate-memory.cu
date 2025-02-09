// Allocate shared memory for host and device
// TOREAD: https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#memory-optimizations
// the easiest way to allocate memory on the device is to use the cudaMalloc function.
//  Exercise: Array Manipulation on both the Host and Device
#include <stdio.h>

void init(int *a, int N)
{
  int i;
  for (i = 0; i < N; ++i)
  {
    a[i] = i;
  }
}

__global__
void doubleElements(int *a, int N)
{
  int i;
  i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N)
  {
    a[i] *= 2;
  }
}

bool checkElementsAreDoubled(int *a, int N)
{
  int i;
  for (i = 0; i < N; ++i)
  {
    if (a[i] != i*2) return false;
  }
  return true;
}

int main()
{
  int N = 1000;
  int *a;

  size_t size = N * sizeof(int);

  /*
   * Use `cudaMallocManaged` to allocate pointer `a` available
   * on both the host and the device.
   */

  cudaMallocManaged(&a, size);

  init(a, N);

  size_t threads_per_block = 256;
  size_t number_of_blocks = (N + threads_per_block - 1) / threads_per_block;

  doubleElements<<<number_of_blocks, threads_per_block>>>(a, N);
  cudaDeviceSynchronize();

  bool areDoubled = checkElementsAreDoubled(a, N);
  printf("All elements were doubled? %s\n", areDoubled ? "TRUE" : "FALSE");

  /*
   * Use `cudaFree` to free memory allocated
   * with `cudaMallocManaged`.
   */

  cudaFree(a);
}

// what about the left over? more thread than work?
// Handling Block Configuration Mismatches to Number of Needed Threads
// how to know the number of workn in kernal function?
#include <stdio.h>


__global__ void initializeElementsTo(int initialValue, int *a, int N)
{
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i < N)
  {
    a[i] = initialValue;
  }
}

int main()
{
  /*
   * Do not modify `N`.
   */

  int N = 1000;

  int *a;
  size_t size = N * sizeof(int);

  cudaMallocManaged(&a, size);

  /*
   * Assume we have reason to want the number of threads
   * fixed at `256`: do not modify `threads_per_block`.
   */

  size_t threads_per_block = 256;

  /*
   * The following is idiomatic CUDA to make sure there are at
   * least as many threads in the grid as there are `N` elements.
   * same as celing(N/thread_per_block)
   */

  size_t number_of_blocks = (N + threads_per_block - 1) / threads_per_block;

  int initialValue = 6;

  initializeElementsTo<<<number_of_blocks, threads_per_block>>>(initialValue, a, N);
  cudaDeviceSynchronize();

  /*
   * Check to make sure all values in `a`, were initialized.
   */

  for (int i = 0; i < N; ++i)
  {
    if(a[i] != initialValue)
    {
      printf("FAILURE: target value: %d\t a[%d]: %d\n", initialValue, i, a[i]);
      cudaFree(a);
      exit(1);
    }
  }
  printf("SUCCESS!\n");

  cudaFree(a);
}

// grid stride loop, what if the data set is larger than the grid?
// gridStride = gridDim.x * blockDim.x
// assuming that all the data is already on the GPU
// Exercise: Use a Grid-Stride Loop to Manipulate an Array Larger than the Grid








