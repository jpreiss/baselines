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
from collections import namedtuple
from sysid_batch_policy import *
import baselines.common.tf_util as U

EnvHypers = namedtuple("EnvHyperparams",
    "alpha_sysid embed_dim")

env_hypers = {
    "Reacher-Batch-v1": EnvHypers(alpha_sysid = 0.1, embed_dim = 2),
    "PointMass-Batch-v0": EnvHypers(alpha_sysid = 0.1, embed_dim = 2),
}

def train(env_id, num_timesteps, seed, use_embed):
    assert env_id in env_hypers
    hypers = env_hypers[env_id]

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
            embed = hypers.embed_dim,
            agents = env.N,
            window = 20,
        )
        return SysIDPolicy(name=name, flavor=EXTRA, dim=dim, hid_size=32, n_hid=2, 
            alpha_sysid=hypers.alpha_sysid)

    #env = bench.Monitor(env, logger.get_dir() and
        #osp.join(logger.get_dir(), str(rank)))
    env.seed(workerseed)
    gym.logger.setLevel(logging.WARN)

    trpo_batch.learn(env, sysid_batch_policy_fn, timesteps_per_batch=256, max_kl=0.01, cg_iters=10, cg_damping=0.1,
        max_timesteps=num_timesteps, gamma=0.99, lam=0.98, vf_iters=2, vf_stepsize=1e-3)
    env.close()
    return pi

def boxprint(lines):
    maxlen = max(len(s) for s in lines)
    frame = lambda s, c: c + s + c[::-1]
    rpad = lambda s: s + " " * (maxlen - len(s))
    bar = frame("-" * (maxlen + 2), '  *')
    print(bar)
    for line in lines:
        print(frame(rpad(line), '  | '))
    print(bar)

def main():
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--env', help='environment ID', default='Hopper-v1')
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--num-timesteps', type=int, default=int(1e8))
    parser.add_argument('--embed', help='use embedding', type=bool, default=False)
    args = parser.parse_args()
    logger.configure()
    boxprint([
        "starting SysID batch TRPO:",
        "environment: " + args.env,
        "using embedding" if args.embed else "NOT using embedding",
    ])
    pi = train(args.env, num_timesteps=args.num_timesteps,
        seed=args.seed, use_embed=args.embed)




if __name__ == '__main__':
    main()
