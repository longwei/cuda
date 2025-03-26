// When UM is allocated, it is not physically allocated until it is accessed. 
// A Page fault occurs in the CPU and trigger the migration
// cudaMemPrefetchAsync() can be used to prefetch the data to the GPU
// why not plan the data migration ahead of time?
// because data show sparse access pattern, and it is hard to predict it.
// the lazy loading and on-demand migration is more efficient

// Exercise: Explore UM Migration and Page Faulting

__global__
void deviceKernel(int *a, int N)
{
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  int stride = blockDim.x * gridDim.x;

  for (int i = idx; i < N; i += stride)
  {
    a[i] = 1;
  }
}

void hostFunction(int *a, int N)
{
  for (int i = 0; i < N; ++i)
  {
    a[i] = 1;
  }
}

int main()
{

  int N = 2<<24;
  size_t size = N * sizeof(int);
  int *a;
  cudaMallocManaged(&a, size);

  /*
   * Conduct experiments to learn more about the behavior of
   * `cudaMallocManaged`.
   *
   * What happens when unified memory is accessed only by the GPU?
   * What happens when unified memory is accessed only by the CPU?
   * What happens when unified memory is accessed first by the GPU then the CPU?
   * What happens when unified memory is accessed first by the CPU then the GPU?
   *
   * Hypothesize about UM behavior, page faulting specificially, before each
   * experiment, and then verify by running `nsys`.
   */

  cudaFree(a);
}
// 
// Yes, there is evidence of memory migration and/or page faulting when unified memory is accessed only by the CPU. 
// The report shows significant time spent in the cudaMallocManaged function, which indicates that memory allocation and possibly migration are occurring. 
// Additionally, the osrt_sum stats report shows a high percentage of time spent in system calls like poll and sem_timedwait, 
// which can be indicative of page fault handling and memory migration activities.

__global__
void deviceKernel(int *a, int N)
{
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  int stride = blockDim.x * gridDim.x;

  for (int i = idx; i < N; i += stride)
  {
    a[i] = 1;
  }
}

void hostFunction(int *a, int N)
{
  for (int i = 0; i < N; ++i)
  {
    a[i] = 1;
  }
}

int main()
{
  int N = 2<<24;
  size_t size = N * sizeof(int);
  int *a;
  cudaMallocManaged(&a, size); // <<--
  hostFunction(a, N);
  cudaFree(a);
}
