import torch
import os

def dist_init(port=23456):
    
    def init_parrots(host_addr, rank, local_rank, world_size, port):
        os.environ['MASTER_ADDR'] = str(host_addr)
        os.environ['MASTER_PORT'] = str(port)
        os.environ['WORLD_SIZE'] = str(world_size)
        os.environ['RANK'] = str(rank)
        torch.distributed.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
    
    def init(host_addr, rank, local_rank, world_size, port):
        host_addr_full = 'tcp://' + host_addr + ':' + str(port)
        torch.distributed.init_process_group("nccl", init_method=host_addr_full,
                                            rank=rank, world_size=world_size)
        torch.cuda.set_device(local_rank)
        assert torch.distributed.is_initialized()


    def parse_host_addr(s):
        if '[' in s:
            left_bracket = s.index('[')
            right_bracket = s.index(']')
            prefix = s[:left_bracket]
            first_number = s[left_bracket+1:right_bracket].split(',')[0].split('-')[0]
            return prefix + first_number
        else:
            return s

    rank = int(os.environ['SLURM_PROCID'])
    local_rank = int(os.environ['SLURM_LOCALID'])
    world_size = int(os.environ['SLURM_NTASKS'])
    
    ip = parse_host_addr(os.environ['SLURM_STEP_NODELIST'])

    if torch.__version__ == 'parrots':
        init_parrots(ip, rank, local_rank, world_size, port)
    else:
        init(ip, rank, local_rank, world_size, port)

    return rank, local_rank, world_size

