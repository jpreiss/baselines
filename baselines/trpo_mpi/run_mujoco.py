#!/usr/bin/env python3
# noinspection PyUnresolvedReferences
from mpi4py import MPI
import multiprocessing
from baselines.common.cmd_util import make_mujoco_env, mujoco_arg_parser
from baselines import logger
from baselines.ppo1.mlp_policy import MlpPolicy
from baselines.trpo_mpi import trpo_mpi
import gym

def train(env_id, num_iters, seed, logdir='./trpo_progress'):
    import baselines.common.tf_util as U
    sess = U.single_threaded_session()
    sess.__enter__()

    rank = MPI.COMM_WORLD.Get_rank()
    if rank == 0:
        logger.configure()
    else:
        logger.configure(format_strs=[])
        logger.set_level(logger.DISABLED)
    workerseed = seed + 10000 * MPI.COMM_WORLD.Get_rank()
    def policy_fn(name, ob_space, ac_space):
        return MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
            hid_size=32, num_hid_layers=2)

    #env = make_mujoco_env(env_id, workerseed)
    env = gym.wrappers.BatchCycler(
        gym.make('Reacher-Batch-v1'))

    trpo_mpi.learn(env, policy_fn, timesteps_per_batch=1024, max_kl=0.01, cg_iters=10, cg_damping=0.1,
        max_iters=num_iters, gamma=0.99, lam=0.98, vf_iters=5, vf_stepsize=1e-3,
        logdir=logdir)
    env.close()

def main():
    ap = mujoco_arg_parser()
    ap.add_argument('--logdir', help='logging directory', default='')
    args = ap.parse_args()
    train(args.env, num_iters=1000, seed=args.seed, logdir=args.logdir)

if __name__ == '__main__':
    main()

