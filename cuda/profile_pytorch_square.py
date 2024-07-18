import torch


def time_pytorch_function(f, input):
    start = torch.cuda.Event(enable_timing = True)
    end = torch.cuda.Event(enable_timing = True)

    # warmup
    for _ in range(5):
        f(input)

    start.record()
    f(input)
    end.record()

    torch.cuda.synchronize()
    return start.elapsed_time(end)

def square_self(a):
    return a*a

def square_pow(a):
    return a ** 2


b = torch.randn(10000,10000).cuda()

#print(time_pytorch_function(torch.square, b))
#print(time_pytorch_function(square_pow, b))
#print(time_pytorch_function(square_self, b))

with torch.autograd.profiler.profile(use_cuda = True) as prof:
    torch.square(b)
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))


with torch.autograd.profiler.profile(use_cuda = True) as prof:
    square_pow(b)
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

with torch.autograd.profiler.profile(use_cuda = True) as prof:
    square_self(b)
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
