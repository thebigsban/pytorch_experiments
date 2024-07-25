# PyTorch Experiments


## Environment Setup

Conda environment because setting up Docker with Jupyter notebook and remote file access takes too long. 
```bash
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
conda install -c conda-forge jupyter matplotlib 
pip3 install einops
pip3 install tqdm
```

```bash
conda activate torchnb
```

## Docker Setup
For including custom CUDA kernels/functions in PyTorch. Use Docker here because developing C++ on Windows is hell. Uses the `cuda-pytorch` docker image. 

## Rotary Positional Encoding
Not sure about terminology, but the `rotate_half` implementation gives the correct results. Tried to implement one that was more in-line with the original Roformer paper, but that actually didn't end up working (the dot product after rotation depended on the index itself as well as the difference in indices). 


## Embedding
Basically just trying to figure out how `nn.Embedding` works. All it does is translate `int` to a `torch.tensor`. Also seems like the resulting `torch.tensor` can be updated via backprop. 

## HuggingFace Tokenizer
[Github](https://github.com/huggingface/tokenizers) and [Documentation](https://huggingface.co/docs/tokenizers/index)
```
pip install tokenizers
```

Did a test training on UniprotKB dataset, took a few minutes for 1k merges on 500k ish sequences. Also code for saving and loading models. 


## CUDA from PyTorch

Vaguely following the [YouTube lectures from CUDA MODE](https://www.youtube.com/@CUDAMODE). 

### Profiling
* CUDA is asynchronous, if you just use the `time` module in python, you measure instead the time that it takes the kernel to startup. 
* Autograd Profiler  
    * ```python
        with torch.autograd.profiler.profile(use_cuda = True) as prof
            func(arg)
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
      ```
    * Did one to profile implementations of multi-headed attention
* PyTorch Profiler
  * [Basic recipes here](https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html)
  * Export chrome trace and view at [chrome://tracing](chrome://tracing)

### Including CPP in PyTorch
* Requires C++ and Ninja so will use a Docker container to run most of this code 
* ```from torch.utils.cpp_extension import load_inline```


## CUDA Basics
[One tutorial here](https://www.nvidia.com/docs/io/116711/sc11-cuda-c-basics.pdf)

[and then the documentation here](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#kernels)

### Basics
* host:  CPU
* device:  gpu
* General steps:
  1. copy data from CPU to GPU
  2. load GPU code and execute, cache data on chip
  3. copy results from GPU to CPU 

### Memory Management
* separate host and device memory
  * device pointers point to GPU, host pointers point to CPU memory
    * can be passed to/from host/device code
    * cannot be dereferenced from outside of device (i.e. can't access values stored in GPU from CPU and vice versa)
* `cudaMemcpy()`:: blocks cpu until copy complete, begins when all previous cuda calls completed
* `cudaMemcpyAsync()`:: asnychronous, does not block CPU
* `cudaDeviceSynchronize()`:: blocks device until all previous cuda calls completed
  
### Parallel Computing
* **Blocks**
  * each parallel invocation of a function is called a **block**
  * sets of blocks referred to as a **grid**
  * `blockIdx.x` refers to the block index, which can be used to index into an array
* **Threads**
  * multiple threads per block (up to 1024)
  * `threadIdx.x` indexes threads within a block
  * `func<<<numBlocks, numThreads>>>(args)` calls the function `func` with `args` with `numBlocks` blocks and `numThreads` threads. 
  * Can do double indexing with `threadIdx` and `blockIdx`. 
  * Number of threads per block is `blockDim.x`
  * Common is `dim3 threadsPerBlock(16, 16);`
* Cooperating Threads
  * shared memory
    * declared with `__shared__`
    * within a block, threads can share data via shared memory
    * extremely fast
    * as opposed to global memory (device memory)
    * not visible to threads in other blocks
  * thread synchronization
    * `__syncthreads()`
    * prevent data hazards
* Blocks and threads are 3d (`blockIdx.x`, `blockIdx.y`, `blockIdx.z`)
**Example: Hello World**
```c++
global__ void mykernel(void) {   // function that you call on cpu, but runs on gpu
} 
int main(void) {                 // host function, compiled and run on cpu (gcc, cl)
    mykernel<<<1,1>>>();         // 'kernel launch'
    printf("Hello World!\n"); 
    return 0; 
}
```

**Example: Addition**
```c++
__global__ void add(int *a, int *b, int *c){
    *c = *a + *b
}

int main(void){
    int a,b,c;   // initialize variables on host
    int *d_a, *d_b, *d_c; //device copies of abc
    int size = sizeof(int) 

    //allocate space on device for abc
    cudaMalloc((void **)&d_a, size);
    cudaMalloc((void **)&d_b, size);
    cudaMalloc((void **)&d_c, size);

    //input values
    a = 2;
    b = 1;

    cudaMemcpy(d_a, &a, size, cudaMemcpyHostToDevice); cudaMemcpy(d_b, &b, size, cudaMemcpyHostToDevice); 
    // Launch add() kernel on GPU 
    add<<<1,1>>>(d_a, d_b, d_c); 
    
    // Copy result back to host 
    cudaMemcpy(&c, d_c, size, cudaMemcpyDeviceToHost); 
    // Cleanup 
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c); return 0; 
}
```
* use pointers for the variables because `add` runs on device and `a` `b` and `c` have to point to device memory. 
* to use `add` in parallel, we change the call from `add<<<1,1>>>` to `add<<<N, 1>>>`. 

**Example: Parallel Vector Addition**
```c++
__global__ void add(int *a, int *b, int *c){
    c[blockIdx.x] = a[blockIdx.x] + b[blockIdx.x]
}

#define N 512 
int main(void){
    int *a, *b, *c;
    int *d_a, *d_b, *d_c;
    int size = N*sizeof(int);

    // Alloc space for device copies of a, b, c 
    cudaMalloc((void **)&d_a, size); cudaMalloc((void **)&d_b, size); cudaMalloc((void **)&d_c, size); 

    // Alloc space for host copies of a, b, c and setup input values 
    a = (int *)malloc(size); 
    random_ints(a, N); 
    b = (int *)malloc(size); 
    random_ints(b, N); 
    c = (int *)malloc(size);

    // Copy inputs to device 
    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice); 
    cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice); 
    
    // Launch add() kernel on GPU with N blocks 
    add<<<N,1>>>(d_a, d_b, d_c); 
    
    // Copy result back to host 
    cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost); 
    
    // Cleanup f
    free(a); free(b); free(c); cudaFree(d_a); cudaFree(d_b); cudaFree(d_c); return 0; 
}
```

**Example: 1D Stencil (conv)**

Problem: sum elements within a certain radius. For example, if radius = 3, then need to sum 7 elements together to get the final output. 

Solution: 
* within each block, each thread is responsible for generating one element of the output array. 
* this means each block needs to read `blockDim.x + 2*radius` elements