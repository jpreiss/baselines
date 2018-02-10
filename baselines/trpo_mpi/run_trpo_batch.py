#!/usr/bin/env python3
# noinspection PyUnresolvedReferences
from mpi4py import MPI
from baselines.common import set_global_seeds
import os.path as osp
import gym
import logging
from baselines import logger
from baselines.ppo1.mlp_policy import MlpPolicy
from baselines.common.mpi_fork import mpi_fork
from baselines import bench
from baselines.trpo_mpi import trpo_batch
import sys
#from sysid_batch_policy import SysIDPolicy, Dim
from sysid_direct_batch_policy import SysIDPolicy, Dim

def train(env_id, num_timesteps, seed):
    import baselines.common.tf_util as U
    sess = U.single_threaded_session()
    sess.__enter__()

    rank = MPI.COMM_WORLD.Get_rank()
    if rank != 0:
        logger.set_level(logger.DISABLED)
    workerseed = seed + 10000 * MPI.COMM_WORLD.Get_rank()
    set_global_seeds(workerseed)

    env = gym.make(env_id)
    sysid_dim = int(env.sysid_dim)

    def sysid_batch_policy_fn(name, ob_space, ac_space):
        for space in (ob_space, ac_space):
            assert isinstance(space, gym.spaces.Box)
            assert len(space.shape) == 1
        dim = Dim(
            ob = ob_space.shape[0] - sysid_dim,
            sysid = sysid_dim, # NOTE this is a closure
            ob_concat = ob_space.shape[0],
            ac = ac_space.shape[0],
            embed = 2,
            agents = env.N,
            window = 20,
        )
        return SysIDPolicy(name=name, dim=dim, hid_size=32, n_hid=2)

    #env = bench.Monitor(env, logger.get_dir() and
        #osp.join(logger.get_dir(), str(rank)))
    env.seed(workerseed)
    gym.logger.setLevel(logging.WARN)

    trpo_batch.learn(env, sysid_batch_policy_fn, timesteps_per_batch=256, max_kl=0.01, cg_iters=10, cg_damping=0.1,
        max_timesteps=num_timesteps, gamma=0.99, lam=0.98, vf_iters=2, vf_stepsize=1e-3)
    env.close()

def main():
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--env', help='environment ID', default='Hopper-v1')
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--num-timesteps', type=int, default=int(1e8))
    args = parser.parse_args()
    logger.configure()
    train(args.env, num_timesteps=args.num_timesteps, seed=args.seed)


if __name__ == '__main__':
    main()
