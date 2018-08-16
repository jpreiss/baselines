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


def learn(np_random, env, policy_func, *,
        #entcoeff, # clipping parameter epsilon, entropy coeff
        optim_epochs, optim_batchsize,# optimization hypers
        gamma,
        max_iters=0,
        callback=None, # you can do anything in the callback, since it takes locals(), globals()
        schedule='constant', # annealing for stepsize parameters (epsilon and adam)
        logdir
        ):


    pi = policy_func("pi", env.observation_space, env.action_space)
    dim = pi.dim
    N = env.N

    def check_reuse(vars1, vars2):
        assert(set(vars1) == set(vars2))

    # SAC paper
    Q_lr = 3e-4
    V_lr = 3e-4
    pi_lr = 3e-4
    V_tau = 0.10
    buf_len = int(1e6)
    alpha = 0.0

    # placeholders
    #ph_ob = tf.placeholder(tf.float32, (None, dim.ob_concat))
    ph_ac = tf.placeholder(tf.float32, (None, dim.ac), name="ph_action")
    ph_rew = tf.placeholder(tf.float32, (None,), name="ph_reward")
    ph_nextob = tf.placeholder(tf.float32, (None, dim.ob_concat), name="ph_nextob")
    with tf.variable_scope("nextob_whiten"):
        nextobz = pi.whiten(ph_nextob, "next_obz")

    l2reg = tf.contrib.layers.l2_regularizer(1e-3)
    def regs():
        return tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
        #return 0.0

    # value network
    #import pdb; pdb.set_trace()
    V_hidden = (64, 64)
    V = batch2.MLP("V", pi.obz_all, V_hidden, 1, tf.nn.relu, reg=l2reg)
    V_target = batch2.MLP("V_target", pi.obz_all, V_hidden, 1, tf.nn.relu, reg=l2reg)
    V_target_nextob = batch2.MLP("V_target", nextobz, V_hidden, 1, tf.nn.relu, reg=l2reg, reuse=True)
    check_reuse(V_target.vars, V_target_nextob.vars)
    with tf.variable_scope("V_expavg_assign"):
        V_target_update_ops = [tf.assign(vtarg, batch2.lerp(vtarg, v, V_tau))
            for vtarg, v in zip(V_target.vars, V.vars)]

    # double Q networks
    with tf.variable_scope("Q_target"):
        Q_target = tf.stop_gradient(5.0 * ph_rew + gamma * V_target_nextob.out)
    Q_input_train = tf.concat([pi.obz_all, ph_ac], axis=1, name="Q_input_train")
    Q_input_eval = tf.concat([pi.obz_all, pi.ac], axis=1, name="Q_input_eval")
    def make_Q(name):
        trainer = batch2.MLP(name, Q_input_train, (64, 64), 1, tf.nn.relu, reg=l2reg)
        evaler = batch2.MLP(name, Q_input_eval, (64, 64), 1, tf.nn.relu, reg=l2reg, reuse=True)
        check_reuse(trainer.vars, evaler.vars)
        with tf.variable_scope(name + "_loss"):
            trainer.loss = 0.5 * tf.reduce_mean((trainer.out - Q_target) ** 2) + regs()
        with tf.variable_scope(name + "_adam"):
            adam = tf.train.AdamOptimizer(Q_lr)
            trainer.opt_op = adam.minimize(trainer.loss, var_list=trainer.vars)
        return trainer, evaler

    # TODO: why does the code call the Q values "log_target" when there's no log?
    Qs_train, Qs_eval = zip(*[make_Q(f"Q_{i}") for i in range(2)])
    with tf.variable_scope("Q_min_train"):
        Q_min_train = tf.minimum(*[Q.out for Q in Qs_train])
    #assert Q_min_train.shape.as_list() == Qs_train[0].out.shape.as_list()
    #Q_min_eval = Q_min_train
    with tf.variable_scope("Q_min_eval"):
        Q_min_eval = tf.minimum(*[Q.out for Q in Qs_eval])

    # value network training.
    # TODO: is it important to include the "action prior" from haarnoja/sac.py?
    with tf.variable_scope("V_loss"):
        pi_vf_logp = pi.logp_corrected(pi.raw_ac)
        V_loss = 0.5 * tf.reduce_mean((
            #V.out - tf.stop_gradient(Q_min_eval - alpha * pi.pd.logp(pi_sample))) ** 2)
            V.out - tf.stop_gradient(Q_min_eval - alpha * pi_vf_logp)) ** 2) + regs()

    with tf.variable_scope("V_adam"):
        V_opt_op = tf.train.AdamOptimizer(V_lr).minimize(V_loss, var_list=V.vars)

    # TODO SAC paper says the minimum Q is used in the policy gradient
    #      (below eqn. 13), but in the code (sac.py _init_actor_update method),
    #      only one of the Q networks is used in the policy gradient. why?
    #      (Note: here we are using the min!)
    # Note: Q_min_eval gradient is not stopped here,
    # so we have 2 "paths" to update the policy:
    # either change the log-probability of the action to match the Q functions,
    # OR change the action such that the outputted Q value is different.
    # TODO: read more DDPG etc. papers to fully understand.

    #pi_loss = tf.reduce_mean(pi.pd.logp(ph_ac) - Q_min_eval)
    with tf.variable_scope("pi_loss"):
        #action_prior = tf.distributions.Normal(loc=0.0, scale=1.0).log_prob(pi.ac)
        pi_Q_target = Q_min_eval
        with tf.variable_scope("log_prob"):
            pi_train_logp = pi.logp_corrected(pi.raw_ac)
        pi_loss = tf.reduce_mean(alpha * pi_train_logp - pi_Q_target) # + regs()
        pi_loss += 0.001 * tf.reduce_mean(pi.ac ** 2)
        pi_vars = [v for v in pi.get_trainable_variables() if v.name.startswith("pi/pol")]

    with tf.variable_scope("pi_adam"):
        pi_opt_op = tf.train.AdamOptimizer(pi_lr).minimize(pi_loss,
            var_list=pi_vars)


    buf_dims = (dim.ob_concat, dim.ac, 1, dim.ob_concat)
    replay_buf = batch2.ReplayBuffer(buf_len, buf_dims)

    with tf.Session() as sess:
        writer = tf.summary.FileWriter('./board', tf.get_default_session().graph)
        seg_gen = batch2.sysid_simple_generator(pi, env, stochastic=True, test=False)

        iters_so_far = 0
        tstart = time.time()

        logger.configure(dir=logdir, format_strs=['stdout', 'csv'])

        # init tf
        sess.run(tf.global_variables_initializer())
        #sess.run([tf.assign(vtarg, v) for vtarg, v in zip(V_target.vars, V.vars)])

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

            # add to replay buffer
            horizon = ob.shape[0]
            assert ac.shape[0] == horizon
            assert rew.shape[0] == horizon
            for t in range(horizon - 1):
                for agent in range(N):
                    replay_buf.add(ob[t,agent], ac[t,agent], rew[t,agent], ob[t+1,agent])

            if hasattr(pi, "ob_rms"):
                ob_flat = ob.reshape((-1, dim.ob_concat))
                pi.ob_rms.update(ob_flat) # update running mean/std for policy


            if replay_buf.size < 1e4:
                continue

            for _ in range(optim_epochs):
                ot, at, rt, ot1 = replay_buf.sample(np_random, optim_batchsize)

                #pi_act = sess.run(pi.pd.sample(), feed_dict={ pi.ob : ot })

                # update value func estimate
                _, V_loss_b, V_b, rawac, ac, logp = sess.run([V_opt_op, tf.sqrt(V_loss), V.out, pi.raw_ac, pi.ac,  pi_vf_logp], feed_dict={
                    pi.ob : ot,
                    pi.stochastic : True,
                    #ph_ac : pi_act,
                })
                mean_prob = np.mean(np.exp(logp))
                logger.record_tabular("pi_vf_mean_prob", mean_prob)
                if mean_prob > 1000:
                    import pdb; pdb.set_trace()

                # update both q func estimates
                Q_eval = [Q.opt_op for Q in Qs_train] + [tf.sqrt(Q.loss) for Q in Qs_train] + [Q_target]
                _, _, Q1_rmse, Q2_rmse, Q_target_b = sess.run(Q_eval, feed_dict={
                    pi.ob : ot,
                    #ph_ob : ot,
                    ph_ac : at,
                    ph_rew : rt.flat,
                    ph_nextob : ot1
                })

                if replay_buf.size >= 2e4:
                    # update policy actor-critic style
                    _, pi_loss_b, = sess.run([pi_opt_op, pi_loss], feed_dict={
                        pi.ob : ot,
                        pi.stochastic : True,
                        #ph_ob : ot,
                        #ph_ac : pi_act
                    })
                    logger.record_tabular("pi_loss", pi_loss_b)

                # update slow-moving target V fn
                sess.run(V_target_update_ops)

            logger.record_tabular("V_loss", V_loss_b)
            logger.record_tabular("V_mean", np.mean(V_b.flat))
            logger.record_tabular("Q1_rmse", Q1_rmse)
            logger.record_tabular("Q2_rmse", Q2_rmse)
            logger.record_tabular("Q_target_mean", np.mean(Q_target_b.flat))

            iters_so_far += 1
            ep_rew = np.sum(rew, axis=0)
            logger.record_tabular("EpRewMean", np.mean(ep_rew.flat))
            logger.record_tabular("EpRewWorst", np.amin(ep_rew))
            logger.record_tabular("TimeElapsed", time.time() - tstart)
            for i in range(N):
                logger.record_tabular("Env{}Rew".format(i), ep_rew[i])
            logger.dump_tabular()
