"""
多算法对比模块
包含 MPD-PPO, Vanilla PPO, A2C, DDPG, TD3, SAC 的统一接口实现
"""
from .base_algorithm import BaseAlgorithm, ReplayBuffer, QNetworkMLP, QNetwork, create_actor, create_critic
from .ppo_vanilla import VanillaPPO
from .a2c import A2C
from .ddpg import DDPG
from .td3 import TD3
from .sac import SAC
from .taa_ppo import TAAPPO

ALGO_REGISTRY = {
    'mpd_ppo': 'MPDPPO',
    'vanilla_ppo': 'VanillaPPO',
    'a2c': 'A2C',
    'ddpg': 'DDPG',
    'td3': 'TD3',
    'sac': 'SAC',
    'taa_ppo': 'TAAPPO',
}

def create_algorithm(algo_name, input_dim, num_pots, action_dim, device='cpu'):
    """工厂函数：根据名称创建算法实例"""
    from ppo import MPDPPO as MPD

    algo_name = algo_name.lower()
    if algo_name == 'mpd_ppo':
        return MPD(input_dim, num_pots, action_dim, device)
    elif algo_name == 'vanilla_ppo':
        return VanillaPPO(input_dim, num_pots, action_dim, device)
    elif algo_name == 'a2c':
        return A2C(input_dim, num_pots, action_dim, device)
    elif algo_name == 'ddpg':
        return DDPG(input_dim, num_pots, action_dim, device)
    elif algo_name == 'td3':
        return TD3(input_dim, num_pots, action_dim, device)
    elif algo_name == 'sac':
        return SAC(input_dim, num_pots, action_dim, device)
    elif algo_name == 'taa_ppo':
        return TAAPPO(input_dim, num_pots, action_dim, device)
    else:
        raise ValueError(f"Unknown algorithm: {algo_name}. Choose from: {list(ALGO_REGISTRY.keys())}")
