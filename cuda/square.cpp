// _global__ :: indicates function is run on device and called from host code
// device functions processed by nvidia compiler (kernel)
// host functions (main) processed by host compiler (gcc, cl, etc.)


__global__ void square_matrix_kernel(const float* matrix, float* result, int width, int height){
    int row = blockIdx.y * blockDim.y + threadIdx.y; 
    int col = blockIdx.x * blockDim.x + threadIdx.x; 

    if (row < height && col < width){
        int indx = row * width + col
        result[idx] = matrix[idx] * matrix[idx]
    }
}

torch::Tensor square_matrix(torch::Tensor matrix){
    const auto height = matrix.size(0)
    const auto width = matrix.size(1)

    auto result = torch::empty_like(matrix)

    dim3 threads_per_block(16, 16);
    
    // way number of blocks calculated is to ensure each thread is one output element
    dim3 number_of_blocks((width + threads_per_block.x - 1)/threads_per_block.x,     
                          (height + threads_per_block.y - 1)/threads_per_block.y);

    square_matrx_kernel<<<number_of_blocks, threads_per_block>>>(
        matrx.data_ptr<float>(), result.data_ptr<float>(), width, height);
    return result;
}
