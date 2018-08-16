from baselines.common import Dataset, explained_variance, fmt_row, zipsame
import baselines.common.batch_util2 as batch2
from baselines import logger
import baselines.common.tf_util as U
from baselines.common.mpi_adam import MpiAdam
from baselines.common.mpi_moments import mpi_moments

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

import time
import itertools as it


def flatten_lists(listoflists):
    return [el for list_ in listoflists for el in list_]

def printstats(var, name):
    print("{}: mean={:3f}, std={:3f}, min={:3f}, max={:3f}".format(
        name, np.mean(var), np.std(var), np.min(var), np.max(var)))

# TODO:
# scale by entropy coefficient
# learning rate args
# SysID
# MLP layer param inputs


# minimal shim to satisfy batch2's act() interface
class TempPolicy(object):
    def __init__(self, sess, ob_in, ac_out):
        self.sess = sess
        self.ob_in = ob_in
        self.ac_out = ac_out
        self.flavor = "plain"

    def act(self, stochastic, obs):
        ac = self.sess.run(self.ac_out, feed_dict = {
            self.ob_in : obs,
        })
        return ac, None, None

    def set_is_train(self, _):
        pass


class UniformPolicy(object):
    def __init__(self, low, high):
        self.low = low
        self.high = high
        self.flavor = "plain"

    def act(self, stochastic, obs):
        dim = len(self.low)
        N = obs.shape[0]
        acs = np.random.uniform(self.low, self.high, size=(N, dim))
        return acs, None, None

    def set_is_train(self, _):
        pass


def learn(np_random, env, policy_func, *,
        #entcoeff, # clipping parameter epsilon, entropy coeff
        optim_epochs, optim_batchsize,# optimization hypers
        gamma,
        max_iters=0,
        callback=None, # you can do anything in the callback, since it takes locals(), globals()
        schedule='constant', # annealing for stepsize parameters (epsilon and adam)
        logdir
        ):

    # this is just to grab Dim
    with tf.variable_scope("ignore"):
        ob_space = env.observation_space
        ac_space = env.action_space
        pi = policy_func("DO_NOT_USE", ob_space, ac_space)
        dim = pi.dim

    N = env.N

    # TODO param
    #lr = 1e-3
    lr = 3e-4
    scale_reward = 5.0
    tau = 0.005
    init_explore_steps = 1e4
    n_train_repeat = int(np.log2(2*N)) # made-up heuristic
    buf_len = int(1e6)
    mlp_size = (128, 128)


    # placeholders
    obs_ph = tf.placeholder(tf.float32, (None, dim.ob_concat), "ob")
    nextob_ph = tf.placeholder(tf.float32, (None, dim.ob_concat), "next_ob")
    ac_ph = tf.placeholder(tf.float32, (None, dim.ac), "ac")
    rew_ph = tf.placeholder(tf.float32, (None, ), "rew")

    # value function
    vf = batch2.MLP("myvf", obs_ph, mlp_size, 1, tf.nn.relu)

    # policy
    reg = tf.contrib.layers.l2_regularizer(1e-3)
    policy = batch2.SquashedGaussianPolicy("sgpolicy",
        obs_ph, mlp_size, dim.ac, tf.nn.relu, reg=reg)
    log_pi = policy.logp(policy.raw_ac)

    # double q functions - these ones are used "on-policy" in the vf loss
    q_in = tf.concat([obs_ph, policy.ac], axis=1)
    qf1 = batch2.MLP("qf1", q_in, mlp_size, 1, tf.nn.relu)
    qf2 = batch2.MLP("qf2", q_in, mlp_size, 1, tf.nn.relu)
    qf_min = tf.minimum(qf1.out, qf2.out)

    # policy loss
    policy_kl_loss = tf.reduce_mean(log_pi - qf_min)
    pi_reg_losses = tf.get_collection(
        tf.GraphKeys.REGULARIZATION_LOSSES, scope=policy.name)
    pi_reg_losses += [policy.reg_loss]
    policy_loss = policy_kl_loss + tf.reduce_sum(pi_reg_losses)

    # value function loss
    vf_loss = 0.5 * tf.reduce_mean((vf.out - tf.stop_gradient(qf_min - log_pi))**2)

    # same q functions, but for the off-policy TD training
    qtrain_in = tf.concat([obs_ph, ac_ph], axis=1)
    qf1_t = batch2.MLP("qf1", qtrain_in, mlp_size, 1, tf.nn.relu, reuse=True)
    qf2_t = batch2.MLP("qf2", qtrain_in, mlp_size, 1, tf.nn.relu, reuse=True)

    # target (slow-moving) vf, used to update Q functions
    with tf.variable_scope('target'):
        vf_TDtarget = batch2.MLP("vf_target", nextob_ph, mlp_size, 1, tf.nn.relu)

    # q fn TD-target & losses
    ys = tf.stop_gradient(scale_reward * rew_ph + gamma * vf_TDtarget.out)
    TD_loss1 = 0.5 * tf.reduce_mean((ys - qf1_t.out)**2)
    TD_loss2 = 0.5 * tf.reduce_mean((ys - qf2_t.out)**2)


    # training ops
    policy_opt_op = tf.train.AdamOptimizer(lr).minimize(
        policy_loss, var_list=policy.get_params_internal())

    vf_opt_op = tf.train.AdamOptimizer(lr).minimize(
        vf_loss, var_list=vf.vars)

    qf1_opt_op = tf.train.AdamOptimizer(lr).minimize(
        TD_loss1, var_list=qf1.vars)

    qf2_opt_op = tf.train.AdamOptimizer(lr).minimize(
        TD_loss2, var_list=qf2.vars)

    train_ops = [policy_opt_op, vf_opt_op, qf1_opt_op, qf2_opt_op]

    # ops to update slow-moving target vf
    vf_target_moving_avg_ops = [
        tf.assign(target, (1 - tau) * target + tau * source)
        for target, source in zip(vf_TDtarget.vars, vf.vars)
    ]



    buf_dims = (dim.ob_concat, dim.ac, 1, dim.ob_concat)
    replay_buf = batch2.ReplayBuffer(buf_len, buf_dims)


    with tf.Session() as sess:

        writer = tf.summary.FileWriter('./board', sess.graph)

        # init tf
        sess.run(tf.global_variables_initializer())
        sess.run(vf_target_moving_avg_ops)
        #sess.run([tf.assign(vtarg, v) for vtarg, v in zip(V_target.vars, V.vars)])

        do_train = False

        # update the policy every time step for high sample efficiency
        # note: closure around do_train
        def per_step_callback(locals, globals):

            ob = locals["ob"]
            ac = locals["ac"]
            rew = locals["rew"][:,None]
            ob_next = locals["ob_next"]
            # add all agents' steps to replay buffer
            replay_buf.add_batch(ob, ac, rew, ob_next)

            if not do_train:
                return

            # gradient step
            for i in range(n_train_repeat):
                ot, at, rt, ot1 = replay_buf.sample(np_random, optim_batchsize)
                feed_dict = {
                    obs_ph: ot,
                    ac_ph: at,
                    rew_ph: rt[:,0],
                    nextob_ph: ot1,
                }
                # TODO get diagnostics
                sess.run(train_ops, feed_dict)
                sess.run(vf_target_moving_avg_ops)


        explore_epochs = int(np.ceil(float(init_explore_steps) / (N * env.ep_len)))
        print(f"random exploration stage: {explore_epochs} epochs...")

        policy_uniform = UniformPolicy(-np.ones(dim.ac), np.ones(dim.ac))
        policy_uniform.dim = dim

        exploration_gen = batch2.sysid_simple_generator(
            policy_uniform, env, stochastic=True, test=False,
            callback=per_step_callback)

        for i, seg in enumerate(it.islice(exploration_gen, explore_epochs)):
            # callback does almost all the work
            do_train = replay_buf.size > 2000
            print(f"exploration epoch {i} complete")

        do_train = True

        print("begin policy rollouts")
        policy_wrapper = TempPolicy(sess, obs_ph, policy.ac)
        policy_wrapper.dim = dim

        iters_so_far = 0
        tstart = time.time()

        logger.configure(dir=logdir, format_strs=['stdout', 'csv'])

        seg_gen = batch2.sysid_simple_generator(
            policy_wrapper, env, stochastic=True, test=False,
            callback=per_step_callback)

        for seg in seg_gen:
            if callback: callback(locals(), globals())
            if max_iters and iters_so_far >= max_iters:
                print("breaking due to max iters")
                break

            if schedule == 'constant':
                cur_lrmult = 1.0
            elif schedule == 'linear':
                cur_lrmult =  max(1.0 - float(iters_so_far) / max_iters, 0.10)
            else:
                raise NotImplementedError

            logger.log("********** Iteration %i ************"%iters_so_far)

            ob, ac, rew = seg["ob"], seg["ac"], seg["task_rews"]

            #if hasattr(pi, "ob_rms"):
                #ob_flat = ob.reshape((-1, dim.ob_concat))
                #pi.ob_rms.update(ob_flat) # update running mean/std for policy


            # TODO: uniform exploration policy
            if replay_buf.size < init_explore_steps:
                continue

            #logger.record_tabular("V_loss", V_loss_b)
            #logger.record_tabular("V_mean", np.mean(V_b.flat))
            #logger.record_tabular("Q1_rmse", Q1_rmse)
            #logger.record_tabular("Q2_rmse", Q2_rmse)
            #logger.record_tabular("Q_target_mean", np.mean(Q_target_b.flat))

            iters_so_far += 1
            ep_rew = np.sum(rew, axis=0)
            logger.record_tabular("EpRewMean", np.mean(ep_rew.flat))
            logger.record_tabular("EpRewWorst", np.amin(ep_rew))
            logger.record_tabular("TimeElapsed", time.time() - tstart)
            for i in range(N):
                logger.record_tabular("Env{}Rew".format(i), ep_rew[i])
            logger.dump_tabular()
