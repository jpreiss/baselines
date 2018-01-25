#!/usr/bin/env python3
from baselines.common import set_global_seeds, tf_util as U
from baselines import bench
import gym, logging
from baselines import logger
import sysid_batch_policy

def train(env_id, num_timesteps, seed):
    from baselines.ppo1 import mlp_policy, mlp_batch_policy, pposgd_sysid, pposgd_simple, pposgd_batch

    U.make_session(num_cpu=1).__enter__()
    set_global_seeds(seed)

    env = gym.make(env_id).env
    sysid_dim = int(env.sysid_dim)

    def sysid_policy_fn(name, ob_space, ac_space):

        # user chosen hyperparameters
        latent_dim = 3
        n_history = 20
        alpha_sysid = 0.0

        assert len(ob_space.high.shape) == 1
        assert len(ac_space.high.shape) == 1
        total_dim = ob_space.high.shape[0]
        obs_dim = total_dim - sysid_dim

        return sysid_policy.SysID(name,
            obs_dim, ac_space.high.shape[0],
            latent_dim, sysid_dim, n_history, alpha_sysid)

    def mlp_policy_fn(name, ob_space, ac_space):
        return mlp_batch_policy.MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
            hid_size=64, num_hid_layers=2)

    def sysid_batch_policy_fn(name, ob_space, ac_space):
        traj_len = 100
        return sysid_batch_policy.SysIDPolicy(name=name, 
            ob_space=ob_space, ac_space=ac_space, sysid_dim=sysid_dim, latent_dim=3,
            traj_len=traj_len, hid_size=64, n_hid=2)

    #env = bench.Monitor(env, logger.get_dir())
    env.seed(seed)
    gym.logger.setLevel(logging.WARN)
    pposgd_batch.learn(env, sysid_batch_policy_fn,
            max_timesteps=num_timesteps,
            timesteps_per_actorbatch=512,
            clip_param=0.1, entcoeff=0.0,
            optim_epochs=10, optim_stepsize=3e-4, optim_batchsize=2048,
            gamma=0.99, lam=0.95, schedule='linear',
        )
    env.close()

def main():
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--env', help='environment ID', default='Hopper-v1')
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--num-timesteps', type=int, default=int(1e7))
    args = parser.parse_args()
    logger.configure()
    train(args.env, num_timesteps=args.num_timesteps, seed=args.seed)


if __name__ == '__main__':
    main()
