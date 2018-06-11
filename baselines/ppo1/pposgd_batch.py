from baselines.common import Dataset, explained_variance, fmt_row, zipsame
import baselines.common.batch_util2 as batch2
from baselines import logger
import baselines.common.tf_util as U
from baselines.common.mpi_adam import MpiAdam
from baselines.common.mpi_moments import mpi_moments

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from mpi4py import MPI

import time


def flatten_lists(listoflists):
    return [el for list_ in listoflists for el in list_]

def printstats(var, name):
    print("{}: mean={:3f}, std={:3f}, min={:3f}, max={:3f}".format(
        name, np.mean(var), np.std(var), np.min(var), np.max(var)))


def learn(np_random, env, policy_func, *,
        clip_param, entcoeff, # clipping parameter epsilon, entropy coeff
        optim_epochs, optim_stepsize, optim_batchsize,# optimization hypers
        gamma, lam, # advantage estimation
        max_timesteps=0, max_episodes=0, max_iters=0, max_seconds=0,  # time constraint
        callback=None, # you can do anything in the callback, since it takes locals(), globals()
        adam_epsilon=1e-5,
        schedule='constant', # annealing for stepsize parameters (epsilon and adam)
        logdir
        ):

    N = env.N

    # construct policy computation graphs
    # old policy needed to define surrogate loss (see PPO paper)
    pi = policy_func("pi", env.observation_space, env.action_space)
    oldpi = policy_func("oldpi", env.observation_space, env.action_space)
    dim = pi.dim

    # Target advantage function (if applicable)
    atarg = tf.placeholder(dtype=tf.float32, shape=[None])
    # Empirical return
    ret = tf.placeholder(dtype=tf.float32, shape=[None])

    # learning rate and PPO loss clipping multiplier, updated with schedule
    lrmult = tf.placeholder(name='lrmult', dtype=tf.float32, shape=[])
    clip_param = clip_param * lrmult

    ob = U.get_placeholder_cached(name="ob")
    ac = pi.pdtype.sample_placeholder([None])

    # KL divergence not actually part of PPO, only computed for logging
    kloldnew = oldpi.pd.kl(pi.pd)
    meankl = tf.reduce_mean(kloldnew)

    # policy entropy & regularization
    ent = pi.pd.entropy()
    meanent = tf.reduce_mean(ent)
    ent_rew = entcoeff * meanent

    # construct PPO's pessimistic surrogate loss (L^CLIP)
    ratio = tf.exp(pi.pd.logp(ac) - oldpi.pd.logp(ac))
    surr1 = ratio * atarg
    surr2 = tf.clip_by_value(ratio, 1.0 - clip_param, 1.0 + clip_param) * atarg
    pol_surr = tf.reduce_mean(tf.minimum(surr1, surr2))

    # value function loss
    vf_loss = tf.reduce_mean(tf.square(pi.vpred - ret))

    # total loss for reinforcement learning
    total_rew = pol_surr + ent_rew
    if len(pi.extra_rewards) > 0:
        total_rew += tf.add_n(pi.extra_rewards)
    total_loss = -total_rew

    # these losses are named so we can log them later
    losses = [pol_surr, ent_rew, vf_loss, meankl, meanent] + pi.extra_rewards
    loss_names = ["pol_surr", "ent_rew", "vf_loss", "kl", "ent"] + pi.extra_reward_names
    compute_losses = U.function([ob, ac, atarg, ret, lrmult], losses)

    var_list = pi.get_trainable_variables()
    vf_var_list = [v for v in var_list if v.name.split("/")[1].startswith("vf")]
    assert len(vf_var_list) > 0
    sysid_var_list = [v for v in var_list if v.name.split("/")[1].startswith("sysid")]

    def n_params(vars):
        return np.sum([np.prod(v.shape) for v in vars])
    for list, name in ((var_list, "pol"), (vf_var_list, "vf"), (sysid_var_list, "sysid")):
        print("{}: {} params".format(name, n_params(list)))

    # gradient and Adam optimizer for policy
    lossandgrad = U.function(
        [ob, ac, atarg, ret, lrmult],
        losses + [U.flatgrad(total_loss, var_list)])
    adam = MpiAdam(var_list, epsilon=adam_epsilon)

    # gradient and Adam optimizer for SysID network
    # they are separate so we can use different learning rate schedules, etc.
    traj_ob = U.get_placeholder_cached(name="ob_traj")
    traj_ac = U.get_placeholder_cached(name="ac_traj")
    lossandgrad_sysid = U.function(
        [ob, traj_ob, traj_ac, lrmult],
        [pi.sysid_err_supervised, U.flatgrad(pi.sysid_err_supervised, var_list)])
    adam_sysid = MpiAdam(var_list, epsilon=adam_epsilon)

    lossandgrad_vf = U.function([ob, ac, ret], [vf_loss, U.flatgrad(vf_loss, vf_var_list)])
    adam_vf = MpiAdam(vf_var_list, epsilon=adam_epsilon)

    assign_old_eq_new = U.function([],[], updates=[tf.assign(oldv, newv)
        for (oldv, newv) in zipsame(oldpi.get_variables(), pi.get_variables())])

    # get ready
    U.initialize()
    writer = tf.summary.FileWriter('./board', tf.get_default_session().graph)
    adam.sync()
    adam_sysid.sync()
    adam_vf.sync()
    seg_gen = batch2.sysid_simple_generator(pi, env, stochastic=True, test=False)

    episodes_so_far = 0
    timesteps_so_far = 0
    timesteps_since_last_episode_end = 0
    iters_so_far = 0
    tstart = time.time()

    assert sum([max_iters > 0, max_timesteps > 0, max_episodes > 0, max_seconds > 0]) == 1, \
        "Only one time constraint permitted"

    logger.configure(dir=logdir, format_strs=['stdout', 'csv'])

    while True:
        if callback: callback(locals(), globals())
        if max_timesteps and timesteps_so_far >= max_timesteps:
            print("breaking due to max timesteps")
            break
        if max_episodes and episodes_so_far >= max_episodes:
            print("breaking due to max episodes")
            break
        if max_iters and iters_so_far >= max_iters:
            print("breaking due to max iters")
            break
        if max_seconds and time.time() - tstart >= max_seconds:
            print("breaking due to max seconds")
            break

        if schedule == 'constant':
            cur_lrmult = 1.0
        elif schedule == 'linear':
            cur_lrmult =  max(1.0 - float(iters_so_far) / max_iters, 0.05)
        else:
            raise NotImplementedError

        logger.log("********** Iteration %i ************"%iters_so_far)

        seg = seg_gen.__next__()
        batch2.add_vtarg_and_adv(seg, gamma, lam)
        # flatten leading (agent, rollout) dims to one big batch
        # (note: must happen AFTER add_vtarg_and_adv)
        sysid_vals = seg["ob"][0,:,dim.ob:]
        batch2.seg_flatten_batches(seg)
        ob, ac, atarg, tdlamret = seg["ob"], seg["ac"], seg["adv"], seg["tdlamret"]
        ob_traj, ac_traj = seg["ob_traj"], seg["ac_traj"]

        vpredbefore = seg["vpred"] # predicted value function before udpate
        atarg = (atarg - atarg.mean()) / atarg.std() # standardized advantage function estimate

        if hasattr(pi, "ob_rms"): pi.ob_rms.update(ob) # update running mean/std for policy
        assign_old_eq_new() # set old parameter values to new parameter values

        # Dataset object handles minibatching for SGD
        d = Dataset(np_random,
            dict(
                ob=ob, ac=ac, atarg=atarg, vtarg=tdlamret,
                ob_traj=ob_traj, ac_traj=ac_traj,
            ), shuffle=not pi.recurrent)
        optim_batchsize = min((optim_batchsize or 2048), ob.shape[0])

        #np.seterr(all='raise')
        logger.log(fmt_row(13, loss_names + ["SysID"]))

        logger.log("Evaluating losses before...")
        # compute and log the final losses after this round of optimization.
        # TODO figure out why we do this complicated thing
        # instead of passing in the whole dataset - 
        # maybe it's to avoid filling up GPU memory?
        losses = []
        for batch in d.iterate_once(optim_batchsize):
            newlosses = compute_losses(
                batch["ob"], batch["ac"], batch["atarg"], batch["vtarg"], cur_lrmult)
            sysid_loss, _ = lossandgrad_sysid(
                batch["ob"], batch["ob_traj"], batch["ac_traj"], cur_lrmult)
            losses.append(newlosses + [sysid_loss])

        meanlosses,_,_ = mpi_moments(losses, axis=0)
        logger.log(fmt_row(13, meanlosses))

        logger.log("Optimizing...")
        # Here we do a bunch of optimization epochs over the data
        for _ in range(optim_epochs):
            losses = [] # list of tuples, each of which gives the loss for a minibatch
            for batch in d.iterate_once(optim_batchsize):
                *newlosses, g = lossandgrad(
                    batch["ob"], batch["ac"], batch["atarg"], batch["vtarg"], cur_lrmult)
                adam.update(g, optim_stepsize * cur_lrmult) 

                loss_sysid, g_sysid = lossandgrad_sysid(
                    batch["ob"], batch["ob_traj"], batch["ac_traj"], 0.1*cur_lrmult)
                adam_sysid.update(g_sysid, optim_stepsize * 0.2*cur_lrmult)

                losses.append(newlosses + [loss_sysid])

                _, g_vf = lossandgrad_vf(batch["ob"], batch["ac"], batch["vtarg"])
                adam_vf.update(g_vf, optim_stepsize * cur_lrmult)

            logger.log(fmt_row(13, np.mean(losses, axis=0)))

        logger.log("Evaluating losses...")
        # compute and log the final losses after this round of optimization.
        # TODO figure out why we do this complicated thing
        # instead of passing in the whole dataset - 
        # maybe it's to avoid filling up GPU memory?
        losses = []
        for batch in d.iterate_once(optim_batchsize):
            newlosses = compute_losses(
                batch["ob"], batch["ac"], batch["atarg"], batch["vtarg"], cur_lrmult)
            sysid_loss, _ = lossandgrad_sysid(
                batch["ob"], batch["ob_traj"], batch["ac_traj"], cur_lrmult)
            losses.append(newlosses + [sysid_loss])

        meanlosses,_,_ = mpi_moments(losses, axis=0)
        logger.log(fmt_row(13, meanlosses))


        for (lossval, name) in zipsame(meanlosses, loss_names + ["SysID"]):
            logger.record_tabular("loss_"+name, lossval)

        embeddings_before_update = seg["embed_true"]
        embeddings_after_update = pi.sysid_to_embedded(sysid_vals)
        emb_delta = embeddings_after_update - embeddings_before_update
        printstats(abs(emb_delta), "abs(embeddings change after gradient step)")

        # log some further information about this iteration
        logger.record_tabular("ev_tdlam_before", explained_variance(vpredbefore, tdlamret))
        lrlocal = (seg["ep_lens"], seg["ep_rews"]) # local values
        listoflrpairs = MPI.COMM_WORLD.allgather(lrlocal) # list of tuples
        lens, rews = map(flatten_lists, zip(*listoflrpairs))
        episodes_so_far += len(lens)
        timesteps_so_far += sum(lens)
        iters_so_far += 1
        logger.record_tabular("EpLenMean", np.mean(lens))
        logger.record_tabular("EpRewMean", np.mean(rews))
        logger.record_tabular("EpRewWorst", np.amin(rews))
        logger.record_tabular("EpThisIter", len(lens))
        logger.record_tabular("EpisodesSoFar", episodes_so_far)
        logger.record_tabular("TimestepsSoFar", timesteps_so_far)
        logger.record_tabular("TimeElapsed", time.time() - tstart)
        assert len(rews) == N
        for i in range(N):
            logger.record_tabular("Env{}Rew".format(i), rews[i])
        if MPI.COMM_WORLD.Get_rank()==0:
            logger.dump_tabular()
