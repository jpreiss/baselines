import numpy as np
from plot_sysid import plot_sysid

# reshape array of (..., n) to (..., < n, window)
def _rolling_window(a, window):
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

def make_sysid_trajs(dim, ob, ac, new):
    N = dim.agents
    timesteps = ob.shape[0]
    #print("ob_shape:",ob.shape)
    #print("N, ob_concat:", N, dim.ob_concat)
    assert ob.shape == (timesteps, N, dim.ob_concat)
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
        windows = _rolling_window(t_ob_only.T, dim.window).transpose([1,2,0])
        assert windows.shape[0] < timesteps
        assert windows.shape[1] == dim.window
        assert windows.shape[2] == dim.ob
        # windows is < rollout, window, ob
        ob_traj_batch.append(windows[window_starts,:,:])

        t_ac = ac_trajs[i,:,:]
        windows = _rolling_window(t_ac.T, dim.window).transpose([1,2,0])
        ac_traj_batch.append(windows[window_starts,:,:])

        n_windows = len(window_starts)
        true_sysid = t_ob[0,dim.ob:]
        #assert np.all(t_ob[:,dim.ob:] == true_sysid)
        others_same = [np.all(t_ob[j,dim.ob:] == true_sysid)
            for j in range(timesteps)]
        n_different = timesteps - sum(others_same)
        #if n_different > 0:
            #print("agent {}: {}/{} sysids different".format(
                #i, n_different, timesteps))
            #where_different = [j for j in range(timesteps) if not others_same[j]]
            #print("differences:")
            #for w in where_different:
                #print(w)
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


def eval_embed_errors(env, pi, ob_traj, ac_traj, ob_rep, plot=True):
    # N is not number of agents, but total number of data points
    if ob_traj.size == 0:
        return [np.array([])] * 3
    dim = pi.dim
    N, window, _ = ob_traj.shape
    assert ac_traj.shape[0:2] == (N, window)
    assert ob_rep.shape[0] == N
    sysid_rep = ob_rep[:,dim.ob:]
    embed_rep = pi.sysid_to_embedded(sysid_rep)
    embed_estimated = pi.estimate_sysid(ob_traj, ac_traj)
    assert embed_estimated.shape == embed_rep.shape
    err2 = np.mean((embed_rep - embed_estimated) ** 2, axis=1)
    assert err2.shape == (N,)
    return err2, embed_rep, embed_estimated

def traj_segment_generator(pi, env, horizon, stochastic, test=False, callback=None):
    dim = pi.dim
    t = 0
    N = env.N
    ac = env.action_space.sample() # not used, just so we have the datatype
    ac = np.tile(ac, (N, 1))
    new = True # marks if we're on first timestep of an episode
    ob = env.reset()
    #ob = env._get_obs()
    assert ob.shape[0] == N

    # HACK TODO
    if pi.flavor == "traj":
        test = True

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

    # rolling window, starting with zeros
    ob_traj_input = np.zeros((horizon, N, dim.window, dim.ob))
    ac_traj_input = np.zeros((horizon, N, dim.window, dim.ac))

    k_episodes = 0
    render_every = 25
    while True:
        if callback is not None:
            callback(env, pi)
        if k_episodes % render_every == 0:
            pass
            #env.render()

        prevac = ac

        i = t % horizon
        if test:
            ac, vpred = pi.act_traj(stochastic, ob, ob_traj_input[i], ac_traj_input[i])
        else:
            ac, vpred = pi.act(stochastic, ob)

        # Slight weirdness here because we need value function at time T
        # before returning segment [0, T-1] so we get the correct
        # terminal value
        if t > 0 and t % horizon == 0:
            k_episodes += 1
            true_sysid = env.sysid_values()
            # TODO:explain
            obs[0,:,dim.ob:] = true_sysid

            ob_trajs, ac_trajs, ob_reps, window_starts, window_scales = make_sysid_trajs(
                pi.dim, obs, acs, news)

            plot = k_episodes % 25 == 1
            #plot = True
            err2s, embed_true, embed_estimate = eval_embed_errors(
                env, pi, ob_trajs, ac_trajs, ob_reps, plot=plot)
            err2s = pi.alpha_sysid * err2s
            #if err2s.size > 0:
                #print("err2s mean val:", np.mean(err2s.flatten()))
            assert len(window_starts) == len(err2s)
            for j, (agent, ind) in enumerate(window_starts):
                rews[ind,agent] -= window_scales[j] * err2s[j]

            # yield the batch to PPO
            yield {
                "ob" : obs, "rew" : rews, "vpred" : vpreds, "new" : news,
                "ac" : acs, "prevac" : prevacs, "nextvpred": vpred * (1 - new),
                "ep_rets" : ep_rets, "ep_lens" : ep_lens,
                "ob_trajs" : ob_trajs, "ac_trajs" : ac_trajs, "ob_reps": ob_reps,
                "ob_traj_input" : ob_traj_input, "ac_traj_input" : ac_traj_input,
                "embed_true" : embed_true, "embed_estimate" : embed_estimate,
            }

            # Be careful!!! if you change the downstream algorithm to aggregate
            # several of these batches, then be sure to do a deepcopy
            #env.reset()
            env.sample_sysid()
            sysids = env.sysid_values()
            embeds = pi.sysid_to_embedded(sysids)
            ep_rets = []
            ep_lens = []
            ob_traj_input *= 0
            ac_traj_input *= 0

        i = t % horizon
        obs[i,:,:] = ob
        vpreds[i,:] = vpred
        news[i,:] = new
        acs[i,:,:] = ac
        prevacs[i,:,:] = prevac

        # ob_traj_input = np.zeros((horizon, N, dim.window, dim.ob))
        if i < horizon - 1:
            ob_traj_input[i+1] = np.roll(ob_traj_input[i], -1, axis=1)
            ac_traj_input[i+1] = np.roll(ac_traj_input[i], -1, axis=1)
            ob_traj_input[i+1,:,-1,:] = ob[:,:dim.ob]
            ac_traj_input[i+1,:,-1,:] = ac

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


def add_vtarg_and_adv(N, seg, gamma, lam):

    def vtarg_and_adv_1d(rew, new, vpred, nextvpred):
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

    new, vpred, nextvpred, rew = seg["new"], seg["vpred"], seg["nextvpred"], seg["rew"]
    k = rew.shape[0]
    tdlamret = np.empty((k,N))
    gaelam = np.empty((k,N))
    for i in range(N):
        tdlamret[:,i], gaelam[:,i] = vtarg_and_adv_1d(
            rew[:,i], new[:,i], vpred[:,i], nextvpred[i])
    seg["tdlamret"] = tdlamret
    seg["adv"] = gaelam 


def seg_flatten_batches(seg):
    for s in ("ob", "ac", "ob_traj_input", "ac_traj_input", "adv", "tdlamret", "vpred"):
        sh = seg[s].shape
        newshape = [sh[0] * sh[1]] + list(sh[2:])
        seg[s] = np.reshape(seg[s], newshape)

