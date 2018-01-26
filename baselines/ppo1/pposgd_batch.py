from baselines.common import Dataset, explained_variance, fmt_row, zipsame
from baselines import logger
import baselines.common.tf_util as U
import tensorflow as tf, numpy as np
import time
from baselines.common.mpi_adam import MpiAdam
from baselines.common.mpi_moments import mpi_moments
from mpi4py import MPI
from collections import deque
from embed_explore import embed_explore
import matplotlib.pyplot as plt

# reshape array of (..., n) to (..., < n, window)
def rolling_window(a, window):
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

# new: 1 where a new rollout started, 0 if continuing
# window_len: the length of input to sysid network
#
# returns: a list of indices where we can start a window
def list_valid_windows(new, window_len):
    assert len(new.shape) == 1
    traj_len = new.shape[0]
    valid_windows = []
    window_scales = []
    # note: mutating reference argument!!!
    new[0] = True
    starts = np.where(new)[0]
    run_lens = np.diff(np.append(starts, len(new)))
    # TODO: vectorize
    for start, run_len in zip(starts, run_lens):
        if run_len >= window_len:
            n_windows = run_len - window_len + 1
            these_windows = range(start, start + n_windows)
            valid_windows.extend(these_windows)
            window_scales.extend(1.0 for _ in these_windows)
    return valid_windows, window_scales

def make_sysid_trajs(pi, env, ob, ac, new):
    N = env.N
    dim = pi.dim
    ob_space = env.observation_space
    ac_space = env.action_space
    assert len(ob_space.shape) == 1
    assert len(ac_space.shape) == 1
    timesteps = ob.shape[0]
    assert ob.shape == (timesteps, N, ob_space.shape[0])
    assert ac.shape == (timesteps, N, dim.ac)
    assert new.shape == (timesteps, N)
    ob_trajs = ob.transpose([1,0,2])
    ac_trajs = ac.transpose([1,0,2])
    new_trajs = new.T
    # just make it "shape check" for now... not dealing with restarts yet

    ob_expanded = []
    ob_traj_batch = []
    ac_traj_batch = []

    # traj input should be [batch, window, ob/ac]
    all_window_starts = []
    all_window_scales = []

    for i in range(N):
        t_ob = ob_trajs[i,:,:]
        t_ob_only = t_ob[:,:dim.ob]

        window_starts, window_scales = list_valid_windows(new_trajs[i,:], dim.window)
        all_window_starts.extend((i, s) for s in window_starts)
        all_window_scales.extend(window_scales)

        # t is rollout * ob
        windows = rolling_window(t_ob_only.T, dim.window).transpose([1,2,0])
        assert windows.shape[0] < timesteps
        assert windows.shape[1] == dim.window
        assert windows.shape[2] == dim.ob
        # windows is < rollout, window, ob
        ob_traj_batch.append(windows[window_starts,:,:])

        t_ac = ac_trajs[i,:,:]
        windows = rolling_window(t_ac.T, dim.window).transpose([1,2,0])
        ac_traj_batch.append(windows[window_starts,:,:])

        n_windows = len(window_starts)
        true_sysid = t_ob[0,dim.ob:]
        #assert np.all(t_ob[:,dim.ob:] == true_sysid)
        others_same = [np.all(t_ob[j,dim.ob:] == true_sysid)
            for j in range(timesteps)]
        n_different = timesteps - sum(others_same)
        if n_different > 0:
            print("agent {}: {}/{} sysids different".format(
                i, n_different, timesteps))
            where_different = [j for j in range(timesteps) if not others_same[j]]
            print("differences:")
            for w in where_different:
                print(w)
        #assert n_different == 0
        ob_rep = np.tile(t_ob[0,:], (n_windows, 1))
        ob_expanded.append(ob_rep)

    ob_traj_batch = np.concatenate(ob_traj_batch, axis=0)
    ac_traj_batch = np.concatenate(ac_traj_batch, axis=0)
    ob_rep_batch = np.concatenate(ob_expanded, axis=0)
    assert(ac_traj_batch.shape[0] == ob_traj_batch.shape[0])
    assert(ob_rep_batch.shape[0] == ob_traj_batch.shape[0])
    return ob_traj_batch, ac_traj_batch, ob_rep_batch, all_window_starts, all_window_scales

def sysid_var_within_traj(sysid_estimated, sysid_rep):
    N, dim = sysid_rep.shape
    n_clusters = 0
    sum_var = 0
    i = 0
    j = 0
    while i < N:
        si = sysid_rep[i,:]
        j = i + 1
        while j < N and np.all(sysid_rep[j,:] == si):
            j += 1
        cluster = sysid_estimated[i:j,:]
        v = np.var(cluster, axis=0)
        sum_var += v
        n_clusters += 1
        i = j
    return sum_var / n_clusters


def eval_sysid_errors(env, pi, ob_traj, ac_traj, ob_rep, plot=False):
    # N is not number of agents, but total number of data points
    dim = pi.dim
    N, window, _ = ob_traj.shape
    assert ac_traj.shape[0:2] == (N, window)
    assert ob_rep.shape[0] == N
    n_uniq = len(np.unique(ob_rep[:,dim.ob]))
    sysid_rep = pi.sysid_to_embedded(ob_rep[:,dim.ob:])
    n_uniq_embed = len(np.unique(sysid_rep[:,0]))
    #print("n_uniq:", n_uniq, ", n_uniq_embed:", n_uniq_embed)
    assert n_uniq_embed == 1 or n_uniq == n_uniq_embed
    sysid_estimated = pi.estimate_sysid(ob_traj, ac_traj)
    var_all = np.var(sysid_estimated, axis=0)
    var_within = sysid_var_within_traj(sysid_estimated, sysid_rep)
    print("var_all: {}\nvar_within:{}".format(var_all, var_within))
    print("mean_true: {}, std_true: {}".format(
        np.mean(sysid_rep, axis=0), np.std(sysid_rep, axis=0)))
    print("mean_est: {}, std_est: {}".format(
        np.mean(sysid_estimated, axis=0), np.std(sysid_estimated, axis=0)))
    assert sysid_estimated.shape == sysid_rep.shape
    err2 = (sysid_rep - sysid_estimated) ** 2
    err2 = np.mean(err2, axis=1)
    assert err2.shape == (N,)

    if plot:
        embed_explore(env, pi, ob_rep, sysid_rep, sysid_estimated)

    return err2

def traj_segment_generator(pi, env, horizon, stochastic):
    dim = pi.dim
    t = 0
    N = env.N
    ac = env.action_space.sample() # not used, just so we have the datatype
    ac = np.tile(ac, (N, 1))
    new = True # marks if we're on first timestep of an episode
    ob = env.reset()
    assert ob.shape[0] == N

    cur_ep_rets = np.zeros(N) # return in current episode
    cur_ep_len = np.zeros(N) # len of current episode
    ep_rets = [] # returns of completed episodes in this segment
    ep_lens = [] # lengths of ...

    # Initialize history arrays
    obs = np.array([ob for _ in range(horizon)])
    rews = np.zeros((horizon, N), 'float32')
    vpreds = np.zeros((horizon, N), 'float32')
    news = np.zeros((horizon, N), 'int32')
    acs = np.array([ac for _ in range(horizon)])
    assert len(acs.shape) == 3
    assert acs.shape[0] == horizon
    assert acs.shape[1] == N
    prevacs = acs.copy()

    k_episodes = 0
    render_every = 20
    while True:
        if k_episodes % render_every == 1:
            env.render()
        prevac = ac
        ac, vpred = pi.act(stochastic, ob)

        # Slight weirdness here because we need value function at time T
        # before returning segment [0, T-1] so we get the correct
        # terminal value
        if t > 0 and t % horizon == 0:
            k_episodes += 1
            true_sysid = env.sysid_values()

            ob_trajs, ac_trajs, ob_reps, window_starts, window_scales = make_sysid_trajs(
                pi, env, obs, acs, news)

            plot = k_episodes % 5 == 1
            #plot = True
            err2s = eval_sysid_errors(env, pi, ob_trajs, ac_trajs, ob_reps, plot=plot)
            err2s = pi.alpha_sysid * err2s
            print("err2s mean val:", np.mean(err2s.flatten()))
            assert len(window_starts) == len(err2s)
            for j, (agent, ind) in enumerate(window_starts):
                rews[ind,agent] -= window_scales[j] * err2s[j]

            yield {"ob" : obs, "rew" : rews, "vpred" : vpreds, "new" : news,
                    "ac" : acs, "prevac" : prevacs, "nextvpred": vpred * (1 - new),
                    "ep_rets" : ep_rets, "ep_lens" : ep_lens}
            # Be careful!!! if you change the downstream algorithm to aggregate
            # several of these batches, then be sure to do a deepcopy
            env.sample_sysid()
            sysids = env.sysid_values()
            embeds = pi.sysid_to_embedded(sysids)
            ep_rets = []
            ep_lens = []
        i = t % horizon
        obs[i,:,:] = ob
        vpreds[i,:] = vpred
        news[i,:] = new
        acs[i,:,:] = ac
        prevacs[i,:,:] = prevac

        ob, rew, new, _ = env.step(ac)
        rews[i,:] = rew

        cur_ep_rets += rew
        cur_ep_len += 1
        new_rets = cur_ep_rets[new]
        new_lens = cur_ep_len[new]
        ep_rets.extend(new_rets)
        ep_lens.extend(new_lens)
        cur_ep_rets[new] = 0
        cur_ep_len[new] = 0
        t += 1

def vtarg_and_adv_1d(rew, new, vpred, nextvpred, gamma, lam):
    new = np.append(new, 0) # last element is only used for last vtarg, but we already zeroed it if last new = 1
    vpred = np.append(vpred, nextvpred)
    T = len(rew)
    adv = gaelam = np.empty(T, 'float32')
    lastgaelam = 0
    for t in reversed(range(T)):
        nonterminal = 1-new[t+1]
        delta = rew[t] + gamma * vpred[t+1] * nonterminal - vpred[t]
        gaelam[t] = lastgaelam = delta + gamma * lam * nonterminal * lastgaelam
    tdlamret = adv + vpred[1:]
    return tdlamret, adv

def add_vtarg_and_adv_old(seg, gamma, lam):
    """
    Compute target value using TD(lambda) estimator, and advantage with GAE(lambda)
    """
    new = np.append(seg["new"], 0) # last element is only used for last vtarg, but we already zeroed it if last new = 1
    vpred = np.append(seg["vpred"], seg["nextvpred"])
    T = len(seg["rew"])
    seg["adv"] = gaelam = np.empty(T, 'float32')
    rew = seg["rew"]
    lastgaelam = 0
    for t in reversed(range(T)):
        nonterminal = 1-new[t+1]
        delta = rew[t] + gamma * vpred[t+1] * nonterminal - vpred[t]
        gaelam[t] = lastgaelam = delta + gamma * lam * nonterminal * lastgaelam
    seg["tdlamret"] = seg["adv"] + seg["vpred"]

def add_vtarg_and_adv(N, seg, gamma, lam):
    new, vpred, nextvpred, rew = seg["new"], seg["vpred"], seg["nextvpred"], seg["rew"]
    k = rew.shape[0]
    tdlamret = np.empty((k,N))
    gaelam = np.empty((k,N))
    for i in range(N):
        tdlamret[:,i], gaelam[:,i] = vtarg_and_adv_1d(
            rew[:,i], new[:,i], vpred[:,i], nextvpred[i], gamma, lam)
    seg["tdlamret"] = tdlamret
    seg["adv"] = gaelam 

def seg_flatten_batches(seg):
    for s in ("ob", "ac", "adv", "tdlamret", "vpred"):
        sh = seg[s].shape
        newshape = [sh[0] * sh[1]] + list(sh[2:])
        seg[s] = np.reshape(seg[s], newshape)

def learn(env, policy_func, *,
        timesteps_per_actorbatch, # timesteps per actor per update
        clip_param, entcoeff, # clipping parameter epsilon, entropy coeff
        optim_epochs, optim_stepsize, optim_batchsize,# optimization hypers
        gamma, lam, # advantage estimation
        max_timesteps=0, max_episodes=0, max_iters=0, max_seconds=0,  # time constraint
        callback=None, # you can do anything in the callback, since it takes locals(), globals()
        adam_epsilon=1e-5,
        schedule='constant' # annealing for stepsize parameters (epsilon and adam)
        ):

    N = env.N

    # Setup losses and stuff
    # ----------------------------------------
    ob_space = env.observation_space
    ac_space = env.action_space
    pi = policy_func("pi", ob_space, ac_space) # Construct network for new policy
    dim = pi.dim
    oldpi = policy_func("oldpi", ob_space, ac_space) # Network for old policy
    atarg = tf.placeholder(dtype=tf.float32, shape=[None]) # Target advantage function (if applicable)
    ret = tf.placeholder(dtype=tf.float32, shape=[None]) # Empirical return

    lrmult = tf.placeholder(name='lrmult', dtype=tf.float32, shape=[]) # learning rate multiplier, updated with schedule
    clip_param = clip_param * lrmult # Annealed cliping parameter epsilon

    ob = U.get_placeholder_cached(name="ob")
    ac = pi.pdtype.sample_placeholder([None])

    kloldnew = oldpi.pd.kl(pi.pd)
    ent = pi.pd.entropy()
    meankl = U.mean(kloldnew)
    meanent = U.mean(ent)
    pol_entpen = (-entcoeff) * meanent

    ratio = tf.exp(pi.pd.logp(ac) - oldpi.pd.logp(ac)) # pnew / pold
    surr1 = ratio * atarg # surrogate from conservative policy iteration
    surr2 = U.clip(ratio, 1.0 - clip_param, 1.0 + clip_param) * atarg #
    pol_surr = - U.mean(tf.minimum(surr1, surr2)) # PPO's pessimistic surrogate (L^CLIP)
    vf_loss = U.mean(tf.square(pi.vpred - ret))
    total_loss = pol_surr + pol_entpen + vf_loss
    losses = [pol_surr, pol_entpen, vf_loss, meankl, meanent]
    loss_names = ["pol_surr", "pol_entpen", "vf_loss", "kl", "ent"]

    var_list = pi.get_trainable_variables()
    lossandgrad = U.function([ob, ac, atarg, ret, lrmult], losses + [U.flatgrad(total_loss, var_list)])
    adam = MpiAdam(var_list, epsilon=adam_epsilon)

    traj_ob = U.get_placeholder_cached(name="traj_ob")
    traj_ac = U.get_placeholder_cached(name="traj_ac")
    lossandgrad_sysid = U.function(
        [ob, traj_ob, traj_ac, lrmult],
        [pi.sysid_err_supervised, U.flatgrad(pi.sysid_err_supervised, var_list)])
    adam_sysid = MpiAdam(var_list, epsilon=adam_epsilon)

    assign_old_eq_new = U.function([],[], updates=[tf.assign(oldv, newv)
        for (oldv, newv) in zipsame(oldpi.get_variables(), pi.get_variables())])
    compute_losses = U.function([ob, ac, atarg, ret, lrmult], losses)

    U.initialize()
    writer = tf.summary.FileWriter('./board', tf.get_default_session().graph)
    adam.sync()
    adam_sysid.sync()

    # Prepare for rollouts
    # ----------------------------------------
    seg_gen = traj_segment_generator(pi, env, timesteps_per_actorbatch, stochastic=True)

    episodes_so_far = 0
    timesteps_so_far = 0
    iters_so_far = 0
    tstart = time.time()
    lenbuffer = deque(maxlen=100) # rolling buffer for episode lengths
    rewbuffer = deque(maxlen=100) # rolling buffer for episode rewards

    assert sum([max_iters>0, max_timesteps>0, max_episodes>0, max_seconds>0])==1, "Only one time constraint permitted"

    timesteps_since_last_episode_end = 0

    while True:
        if callback: callback(locals(), globals())
        if max_timesteps and timesteps_so_far >= max_timesteps:
            break
        elif max_episodes and episodes_so_far >= max_episodes:
            break
        elif max_iters and iters_so_far >= max_iters:
            break
        elif max_seconds and time.time() - tstart >= max_seconds:
            break

        if schedule == 'constant':
            cur_lrmult = 1.0
        elif schedule == 'linear':
            cur_lrmult =  max(1.0 - float(timesteps_so_far) / max_timesteps, 0)
        else:
            raise NotImplementedError
        print("lrmult:", cur_lrmult)

        logger.log("********** Iteration %i ************"%iters_so_far)

        seg = seg_gen.__next__()
        add_vtarg_and_adv(N, seg, gamma, lam)

        # slice-n-dice the ob, ac trajectories to get the training data for sysid
        ob, ac, new, atarg, tdlamret = seg["ob"], seg["ac"], seg["new"], seg["adv"], seg["tdlamret"]
        # don't care about indices of the rolling windows anymore, hence ignoring
        ob_traj_batch, ac_traj_batch, ob_rep_batch, _, _= make_sysid_trajs(
            pi, env, ob, ac, new)

        # ob, ac, atarg, ret, td1ret = map(np.concatenate, (obs, acs, atargs, rets, td1rets))
        seg_flatten_batches(seg)
        ob, ac, atarg, tdlamret = seg["ob"], seg["ac"], seg["adv"], seg["tdlamret"]
        vpredbefore = seg["vpred"] # predicted value function before udpate
        atarg = (atarg - atarg.mean()) / atarg.std() # standardized advantage function estimate
        traj_ob = U.get_placeholder(name="traj_ob",
            dtype=tf.float32, shape=[None, dim.window, dim.ob])
        traj_ac = U.get_placeholder(name="traj_ac",
            dtype=tf.float32, shape=[None, dim.window, dim.ac])

        d = Dataset(dict(ob=ob, ac=ac, atarg=atarg, vtarg=tdlamret), shuffle=not pi.recurrent)
        optim_batchsize = optim_batchsize or ob.shape[0]

        if hasattr(pi, "ob_rms"): pi.ob_rms.update(ob) # update running mean/std for policy

        assign_old_eq_new() # set old parameter values to new parameter values
        logger.log("Optimizing...")
        logger.log(fmt_row(13, loss_names))
        # Here we do a bunch of optimization epochs over the data
        for _ in range(optim_epochs):
            losses = [] # list of tuples, each of which gives the loss for a minibatch
            sysid_losses = []
            for batch in d.iterate_once(optim_batchsize):
                *newlosses, g = lossandgrad(batch["ob"], batch["ac"], batch["atarg"], batch["vtarg"], cur_lrmult)
                adam.update(g, optim_stepsize * cur_lrmult) 

                # seems that we need a lower learning rate here - 
                # perhaps bc the sysid gradient is a less noisy estimate
                # than the RL policy gradient, so there's less cancellation
                # due to momentum in Adam
                *newlosses_sysid, g_sysid = lossandgrad_sysid(
                    ob_rep_batch, ob_traj_batch, ac_traj_batch, cur_lrmult)
                adam_sysid.update(g_sysid, 0.3*optim_stepsize * cur_lrmult)

                losses.append(newlosses)
                sysid_losses.append(newlosses_sysid[0])
            logger.log(fmt_row(13, np.mean(losses, axis=0)))

        logger.log("Evaluating losses...")
        losses = []
        sysid_losses = []
        for batch in d.iterate_once(optim_batchsize):
            newlosses = compute_losses(batch["ob"], batch["ac"], batch["atarg"], batch["vtarg"], cur_lrmult)
            losses.append(newlosses)
            sysid_newlosses, _ = lossandgrad_sysid(
                ob_rep_batch, ob_traj_batch, ac_traj_batch, cur_lrmult)
            sysid_losses.append(sysid_newlosses)
        meanlosses,_,_ = mpi_moments(losses, axis=0)
        logger.log(fmt_row(13, meanlosses))

        for (lossval, name) in zipsame(meanlosses, loss_names):
            logger.record_tabular("loss_"+name, lossval)
        meanlosses_sysid, _, _ = mpi_moments(sysid_losses, axis=0)
        assert len(meanlosses_sysid) == 1
        logger.record_tabular("SysID loss", meanlosses_sysid[0])

        logger.record_tabular("ev_tdlam_before", explained_variance(vpredbefore, tdlamret))
        lrlocal = (seg["ep_lens"], seg["ep_rets"]) # local values
        listoflrpairs = MPI.COMM_WORLD.allgather(lrlocal) # list of tuples
        lens, rews = map(flatten_lists, zip(*listoflrpairs))
        lenbuffer.extend(lens)
        rewbuffer.extend(rews)
        logger.record_tabular("EpLenMean", np.mean(lenbuffer))
        logger.record_tabular("EpRewMean", np.mean(rewbuffer))
        logger.record_tabular("EpThisIter", len(lens))
        episodes_so_far += len(lens)
        timesteps_so_far += sum(lens)
        if len(lens) == 0:
            timesteps_since_last_episode_end += N * timesteps_per_actorbatch
            if timesteps_since_last_episode_end > 400000:
                print("timesteps since last episode ended > 400000 - env learned")
                break
        else:
            timesteps_since_last_episode_end = 0
        iters_so_far += 1
        logger.record_tabular("EpisodesSoFar", episodes_so_far)
        logger.record_tabular("TimestepsSoFar", timesteps_so_far)
        logger.record_tabular("TimeElapsed", time.time() - tstart)
        if MPI.COMM_WORLD.Get_rank()==0:
            logger.dump_tabular()

def flatten_lists(listoflists):
    return [el for list_ in listoflists for el in list_]
