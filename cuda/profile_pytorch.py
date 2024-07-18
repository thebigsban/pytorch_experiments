import torch
import torch.nn.functional as F


a = torch.randn(10,10,1000,1000).cuda()


def manual_dpa(q, k, v):
    attn_weight = q@k.transpose(-2,-1)
    attn_bias = torch.zeros(attn_weight.shape).cuda()
    attn_weight += attn_bias
    attn_weight = torch.softmax(attn_weight, dim = -1)
    return attn_weight @ v

#### CHANGE TO IF TRUE IF YOU WANT OUTPUTS FROM THIS
if False:
    with torch.autograd.profiler.profile(use_cuda = True) as prof:
        F.scaled_dot_product_attention(a,a,a)
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))


    with torch.autograd.profiler.profile(use_cuda = True) as prof:
        manual_dpa(a,a,a)
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))


##################
## OUTPUT
#################
# STAGE:2024-07-17 17:05:18 22608:19120 C:\cb\pytorch_1000000000000\work\third_party\kineto\libkineto\src\ActivityProfilerController.cpp:314] Completed Stage: Warm Up
# C:\Users\banst\Documents\dev\pytorch_experiments\cuda\profile_pytorch_mha.py:8: UserWarning: 1Torch was not compiled with flash attention. (Triggered internally at C:\cb\pytorch_1000000000000\work\aten\src\ATen\native\transformers\cuda\sdp_utils.cpp:455.)
#   F.scaled_dot_product_attention(a,a,a)
# STAGE:2024-07-17 17:05:18 22608:19120 C:\cb\pytorch_1000000000000\work\third_party\kineto\libkineto\src\ActivityProfilerController.cpp:320] Completed Stage: Collection
# STAGE:2024-07-17 17:05:18 22608:19120 C:\cb\pytorch_1000000000000\work\third_party\kineto\libkineto\src\ActivityProfilerController.cpp:324] Completed Stage: Post Processing
# -------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------   
#                                              Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg     Self CUDA   Self CUDA %    CUDA total  CUDA time avg    # of Calls  
# -------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------   
#                aten::scaled_dot_product_attention         0.63%     109.000us       100.00%      17.328ms      17.328ms       3.000us         0.01%      57.268ms      57.268ms             1   
#     aten::_scaled_dot_product_efficient_attention         0.55%      96.000us        99.37%      17.219ms      17.219ms      12.000us         0.02%      57.265ms      57.265ms             1   
#                aten::_efficient_attention_forward        92.09%      15.957ms        98.46%      17.062ms      17.062ms      57.231ms        99.94%      57.237ms      57.237ms             1   
#                                   aten::transpose         0.24%      41.000us         0.35%      61.000us      15.250us       8.000us         0.01%      16.000us       4.000us             4   
#                                  aten::as_strided         0.12%      20.000us         0.12%      20.000us       5.000us       8.000us         0.01%       8.000us       2.000us             4   
#                                       aten::empty         6.38%       1.105ms         6.38%       1.105ms     276.250us       6.000us         0.01%       6.000us       1.500us             4   
# -------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
# Self CPU time total: 17.328ms
# Self CUDA time total: 57.268ms

# STAGE:2024-07-17 17:05:18 22608:19120 C:\cb\pytorch_1000000000000\work\third_party\kineto\libkineto\src\ActivityProfilerController.cpp:314] Completed Stage: Warm Up
# STAGE:2024-07-17 17:05:19 22608:19120 C:\cb\pytorch_1000000000000\work\third_party\kineto\libkineto\src\ActivityProfilerController.cpp:320] Completed Stage: Collection
# STAGE:2024-07-17 17:05:19 22608:19120 C:\cb\pytorch_1000000000000\work\third_party\kineto\libkineto\src\ActivityProfilerController.cpp:324] Completed Stage: Post Processing
# ------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
#                     Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg     Self CUDA   Self CUDA %    CUDA total  CUDA time avg    # of Calls
# ------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
#                 aten::to         0.01%      12.000us        44.41%      62.895ms      62.895ms       2.000us         0.00%      76.635ms      76.635ms             1
#           aten::_to_copy         0.03%      47.000us        44.40%      62.883ms      62.883ms       5.000us         0.00%      76.633ms      76.633ms             1
#              aten::copy_        43.53%      61.651ms        43.53%      61.651ms      61.651ms      76.626ms        48.45%      76.626ms      76.626ms             1
#             aten::matmul         0.12%     176.000us        18.41%      26.072ms      13.036ms      20.000us         0.01%      62.198ms      31.099ms             2
#                aten::bmm        18.16%      25.722ms        18.16%      25.722ms      12.861ms      62.141ms        39.29%      62.141ms      31.070ms             2
#               aten::add_         7.65%      10.841ms         7.65%      10.841ms      10.841ms      17.036ms        10.77%      17.036ms      17.036ms             1
#            aten::softmax         0.02%      24.000us         8.57%      12.135ms      12.135ms       2.000us         0.00%       2.264ms       2.264ms             1
#           aten::_softmax         8.55%      12.111ms         8.55%      12.111ms      12.111ms       2.262ms         1.43%       2.262ms       2.262ms             1
#            aten::reshape         0.04%      51.000us         0.06%      91.000us      22.750us      11.000us         0.01%      18.000us       4.500us             4
#             aten::expand         0.04%      55.000us         0.05%      64.000us      16.000us      10.000us         0.01%      17.000us       4.250us             4
# ------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
# Self CPU time total: 141.622ms
# Self CUDA time total: 158.150ms

from torch.profiler import profile, record_function, ProfilerActivity

with profile(activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    F.scaled_dot_product_attention(a,a,a)

prof.export_chrome_trace('F_sdpa.json')

with profile(activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    manual_dpa(a,a,a)
prof.export_chrome_trace('manual_dpa.json')

