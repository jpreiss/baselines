import baselines.common.batch_util2 as batch2
from baselines import logger

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
# scale by entropy coefficient?


class UniformPolicy(object):
    def __init__(self, low, high):
        self.low = low
        self.high = high
        self.flavor = "plain"
        self.ob = tf.placeholder(tf.float32, (None, None))
        self.est_target = tf.constant(np.zeros((1,0)))
        dist = tf.distributions.Uniform(low, high)
        N = tf.shape(self.ob)[0]
        self.ac_stochastic = dist.sample(N)

    def estimate_sysid(self, *args):
        raise NotImplementedError


def learn(
    sess,              # TF Session
    np_random,         # numpy.random.RandomState for determinism
    env,               # environment w/ OpenAI Gym API
    policy_func,       # func takes (ob space, ac space, ob input placeholder)
    max_iters,         # stop after this many iters (defined by env.ep_len)
    logdir,            # directory to write csv logs

    learning_rate,     # ...
    init_explore_steps,# explore this many steps with random policy at start
    n_train_repeat,    # do this many gradient steps on replay buf each step
    buf_len,           # size of replay buf
    minibatch,         # size of per-step optimization minibatch
    TD_discount,       # ...
    reward_scale,      # SAC uses reward scaling instead of entropy weighting
    tau,               # exp. moving avg. speed for value fn estimator
    ):

    # get dims
    ob_full_dim = env.observation_space.shape[0]
    ac_dim = env.action_space.shape[0]
    N = env.N

    # placeholders
    ob_ph = tf.placeholder(tf.float32, (None, ob_full_dim), "ob")
    ac_ph = tf.placeholder(tf.float32, (None, ac_dim), "ac")
    rew_ph = tf.placeholder(tf.float32, (None, ), "rew")
    nextob_ph = tf.placeholder(tf.float32, (None, ob_full_dim), "next_ob")

    # construct policy, twice so we can reuse the embedder (if applicable)
    with tf.variable_scope("pi"):
        pi = policy_func(env.observation_space, env.action_space, ob_ph)
    with tf.variable_scope("pi", reuse=True):
        pi_nextob = policy_func(env.observation_space, env.action_space, nextob_ph)

    # TODO param
    vf_size = (256, 256, 256)

    # policy's probability of own stochastic action
    with tf.variable_scope("log_prob"):
        log_pi = pi.log_prob

    # value function
    vf = batch2.MLP("myvf", pi.vf_input, vf_size, 1, tf.nn.relu)

    # double q functions - these ones are used "on-policy" in the vf loss
    q_in = tf.concat([pi.vf_input, pi.ac_stochastic], axis=1, name="q_in")
    qf1 = batch2.MLP("qf1", q_in, vf_size, 1, tf.nn.relu)
    qf2 = batch2.MLP("qf2", q_in, vf_size, 1, tf.nn.relu)
    qf_min = tf.minimum(qf1.out, qf2.out, name="qf_min")

    # policy loss
    # TODO impose L2 regularization externally?
    with tf.variable_scope("policy_loss"):
        policy_kl_loss = tf.reduce_mean(log_pi - qf_min)
        pol_vars = set(pi.policy_vars)
        pi_reg_losses = [v for v in
            tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
            if v in pol_vars]
        pi_reg_losses += [pi.reg_loss]
        pi_reg_losses += [-var for var, name in pi.extra_rewards]
        policy_loss = policy_kl_loss + tf.reduce_sum(pi_reg_losses)
        tf.summary.scalar("policy_kl_loss", policy_kl_loss)

    # value function loss
    with tf.variable_scope("vf_loss"):
        vf_loss = 0.5 * tf.reduce_mean((vf.out - tf.stop_gradient(qf_min - log_pi))**2)
        tf.summary.scalar("vf_loss", vf_loss)

    # same q functions, but for the off-policy TD training
    qtrain_in = tf.concat([pi.vf_input, ac_ph], axis=1)
    qf1_t = batch2.MLP("qf1", qtrain_in, vf_size, 1, tf.nn.relu, reuse=True)
    qf2_t = batch2.MLP("qf2", qtrain_in, vf_size, 1, tf.nn.relu, reuse=True)

    # target (slow-moving) vf, used to update Q functions
    vf_TDtarget = batch2.MLP("vf_target", pi_nextob.vf_input, vf_size, 1, tf.nn.relu)

    # q fn TD-target & losses
    with tf.variable_scope("TD_target"):
        TD_target = tf.stop_gradient(
            reward_scale * rew_ph + TD_discount * vf_TDtarget.out)

    with tf.variable_scope("TD_loss1"):
        TD_loss1 = 0.5 * tf.reduce_mean((TD_target - qf1_t.out)**2)
        tf.summary.scalar("TD_loss1", TD_loss1)

    with tf.variable_scope("TD_loss2"):
        TD_loss2 = 0.5 * tf.reduce_mean((TD_target - qf2_t.out)**2)
        tf.summary.scalar("TD_loss2", TD_loss2)


    # training ops
    policy_opt_op = tf.train.AdamOptimizer(learning_rate, name="policy_adam").minimize(
        policy_loss, var_list=pi.policy_vars)

    vf_opt_op = tf.train.AdamOptimizer(learning_rate, name="vf_adam").minimize(
        vf_loss, var_list=vf.vars)

    qf1_opt_op = tf.train.AdamOptimizer(learning_rate, name="qf1_adam").minimize(
        TD_loss1, var_list=qf1.vars)

    qf2_opt_op = tf.train.AdamOptimizer(learning_rate, name="qf2_adam").minimize(
        TD_loss2, var_list=qf2.vars)

    train_ops = [policy_opt_op, vf_opt_op, qf1_opt_op, qf2_opt_op]

    # ops to update slow-moving target vf
    with tf.variable_scope("vf_target_assign"):
        vf_target_moving_avg_ops = [
            tf.assign(target, (1 - tau) * target + tau * source)
            for target, source in zip(vf_TDtarget.vars, vf.vars)
        ]


    buf_dims = (ob_full_dim, ac_dim, 1, ob_full_dim)
    replay_buf = batch2.ReplayBuffer(buf_len, buf_dims)


    writer = tf.summary.FileWriter('./board', sess.graph)
    summaries = tf.summary.merge_all()

    # init tf
    sess.run(tf.global_variables_initializer())
    sess.run(vf_target_moving_avg_ops)
    #sess.run([tf.assign(vtarg, v) for vtarg, v in zip(V_target.vars, V.vars)])

    do_train = False
    n_trains = [0] # get ref semantics in callback closure. lol @python

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
            ot, at, rt, ot1 = replay_buf.sample(np_random, minibatch)
            feed_dict = {
                ob_ph: ot,
                ac_ph: at,
                rew_ph: rt[:,0],
                nextob_ph: ot1,
            }
            # TODO get diagnostics
            sess.run(train_ops, feed_dict)
            sess.run(vf_target_moving_avg_ops)

            if i == 0 and n_trains[0] % 1e3 == 0:
                summary = sess.run(summaries, feed_dict)
                writer.add_summary(summary, i)

        n_trains[0] += 1


    explore_epochs = int(np.ceil(float(init_explore_steps) / (N * env.ep_len)))
    print(f"random exploration stage: {explore_epochs} epochs...")

    policy_uniform = UniformPolicy(-np.ones(ac_dim), np.ones(ac_dim))
    policy_uniform.dim = pi.dim

    exploration_gen = batch2.sysid_simple_generator(sess,
        policy_uniform, env, stochastic=True, test=False,
        callback=per_step_callback)

    for i, seg in enumerate(it.islice(exploration_gen, explore_epochs)):
        # callback does almost all the work
        do_train = replay_buf.size > 2000
        print(f"exploration epoch {i} complete")

    do_train = True

    print("begin policy rollouts")

    iters_so_far = 0
    tstart = time.time()

    logger.configure(dir=logdir, format_strs=['stdout', 'csv'])

    seg_gen = batch2.sysid_simple_generator(sess, pi, env,
        stochastic=True, test=False, callback=per_step_callback)

    tstart = time.time()
    for seg in seg_gen:
        tend = time.time()
        logger.record_tabular("IterSeconds", (tend - tstart))
        tstart = tend
        if max_iters and iters_so_far >= max_iters:
            print("breaking due to max iters")
            break

        logger.log("********** Iteration %i ************"%iters_so_far)

        ob, ac, rew = seg["ob"], seg["ac"], seg["task_rews"]

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
        logger.record_tabular("EpRewBest", np.amax(ep_rew.flat))
        logger.record_tabular("EpRewWorst", np.amin(ep_rew.flat))
        logger.record_tabular("TimeElapsed", time.time() - tstart)
        for i in range(N):
            logger.record_tabular("Env{}Rew".format(i), ep_rew[i])
        logger.dump_tabular()

    return pi
