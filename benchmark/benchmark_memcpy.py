import torch
import triton
import triton.language as tl

@triton.jit
def memcpy_kernel(src_ptr, dst_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Compute the index of the first element this program will handle
    pid = tl.program_id(axis=0)
    # Create a range of block indices this program will handle
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    # Check bounds to avoid out-of-bounds memory access
    mask = offsets < n_elements
    # Load data from src_ptr to dst_ptr
    src = tl.load(src_ptr + offsets, mask=mask)
    tl.store(dst_ptr + offsets, src, mask=mask)

def triton_memcpy(src, dst, BLOCK_SIZE=1024):
    # Ensure src and dst are torch tensors on the GPU
    assert src.is_cuda and dst.is_cuda
    assert src.dtype == dst.dtype
    assert src.numel() == dst.numel()

    n_elements = src.numel()
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)  # Compute the number of program instances needed

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    memcpy_kernel[grid](src, dst, n_elements, BLOCK_SIZE)
    end_event.record()

    torch.cuda.synchronize()
    
    elapsed_time_ms = start_event.elapsed_time(end_event)
    return elapsed_time_ms

def pytorch_memcpy(src, dst):
    assert src.is_cuda and dst.is_cuda
    assert src.dtype == dst.dtype
    assert src.numel() == dst.numel()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    start_event.record()
    dst.copy_(src)
    end_event.record()

    torch.cuda.synchronize()
    
    elapsed_time_ms = start_event.elapsed_time(end_event)
    return elapsed_time_ms

def main():
    print(torch.cuda.is_available())
    print(torch.cuda.device_count())

    src = torch.randn(1024 * 1024, device='cuda')  # 1M elements
    dst_triton = torch.empty_like(src, device='cuda')
    dst_pytorch = torch.empty_like(src, device='cuda')

    pytorch_memcpy(src, dst_pytorch)
    print("Warm-up PyTorch memcpy complete.")
    
    # Perform the triton memcpy operation
    # elapsed_time_triton_ms = triton_memcpy(src, dst_triton)
    # print(f"Triton memcpy operation completed in {elapsed_time_triton_ms:.3f} ms")

    # Perform the PyTorch memcpy operation and measure time
    elapsed_time_pytorch_ms = pytorch_memcpy(src, dst_pytorch)
    print(f"PyTorch memcpy operation completed in {elapsed_time_pytorch_ms:.3f} ms")

    # Perform the triton memcpy operation
    elapsed_time_triton_ms = triton_memcpy(src, dst_triton)
    print(f"Triton memcpy operation completed in {elapsed_time_triton_ms:.3f} ms")

    # Calculate and display memory bandwidth
    bytes_transferred = src.numel() * src.element_size() * 2  # each element is copied from src to dst

    memory_bandwidth_triton_gb_s = bytes_transferred / (elapsed_time_triton_ms * 1e-3) / 1e9
    print(f"Triton Memory Bandwidth: {memory_bandwidth_triton_gb_s:.3f} GB/s")

    # Calculate and display memory bandwidth for PyTorch
    memory_bandwidth_pytorch_gb_s = bytes_transferred / (elapsed_time_pytorch_ms * 1e-3) / 1e9
    print(f"PyTorch Memory Bandwidth: {memory_bandwidth_pytorch_gb_s:.3f} GB/s")

    # Verify the memcpy operation
    assert torch.allclose(src, dst_triton), "Triton memcpy failed"
    assert torch.allclose(src, dst_pytorch), "PyTorch memcpy failed"

if __name__ == '__main__':
    main()