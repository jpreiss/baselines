#!/usr/bin/env python3
from baselines.common import set_global_seeds, tf_util as U
from baselines import bench
import gym, logging
from baselines import logger
import sysid_batch_policy
import sysid_direct_batch_policy

def train(env_id, policy_type, num_timesteps, seed):
    from baselines.ppo1 import mlp_policy, mlp_batch_policy, pposgd_sysid, pposgd_simple, pposgd_batch

    U.make_session(num_cpu=1).__enter__()
    set_global_seeds(seed)

    env = gym.make(env_id)
    sysid_dim = int(env.sysid_dim)

    def mlp_policy_fn(name, ob_space, ac_space):
        for space in (ob_space, ac_space):
            assert isinstance(space, gym.spaces.Box)
            assert len(space.shape) == 1
        dim = sysid_batch_policy.Dim(
            ob = ob_space.shape[0] - sysid_dim,
            sysid = sysid_dim, # NOTE this is a closure
            ob_concat = ob_space.shape[0],
            ac = ac_space.shape[0],
            embed = 3,
            agents = env.N,
            window = 20,
        )
        print("dim:", dim)

        return sysid_direct_batch_policy.SysIDDirectPolicy(
            name=name, dim=dim, hid_size=48, n_hid=2)

    def sysid_batch_policy_fn(name, ob_space, ac_space):
        for space in (ob_space, ac_space):
            assert isinstance(space, gym.spaces.Box)
            assert len(space.shape) == 1
        dim = sysid_batch_policy.Dim(
            ob = ob_space.shape[0] - sysid_dim,
            sysid = sysid_dim, # NOTE this is a closure
            ob_concat = ob_space.shape[0],
            ac = ac_space.shape[0],
            embed = 3,
            agents = env.N,
            window = 64,
        )
        print("dim:", dim)

        return sysid_batch_policy.SysIDPolicy(
            name=name, dim=dim, hid_size=64, n_hid=2)

    if policy_type == "mlp":
        policy_fn = mlp_policy_fn
    elif policy_type == "sysid":
        policy_fn = sysid_batch_policy_fn
    else:
        assert False, "unsupported policy type {} - should be one of ('mlp', 'sysid')".format(policy_type)

    #env = bench.Monitor(env, logger.get_dir())
    env.seed(seed)
    gym.logger.setLevel(logging.WARN)
    pposgd_batch.learn(env, policy_fn,
            max_timesteps=num_timesteps,
            timesteps_per_actorbatch=256,
            clip_param=0.2, entcoeff=4.0e-2,
            optim_epochs=5, optim_stepsize=6e-4, optim_batchsize=512,
            gamma=0.99, lam=0.95, schedule='linear',
        )
    env.close()

def main():
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--env', help='environment ID', default='Hopper-v1')
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--num-timesteps', type=int, default=int(1e7))
    parser.add_argument('--policy', choices=['mlp', 'sysid'])
    args = parser.parse_args()
    logger.configure()
    train(args.env, args.policy, num_timesteps=args.num_timesteps, seed=args.seed)


if __name__ == '__main__':
    main()
