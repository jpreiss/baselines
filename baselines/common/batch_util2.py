import itertools
import numpy as np

RENDER = False

# for fixed length episodes
# expects env to have ep_len member variable
def sysid_simple_generator(pi, env, stochastic, test=False):

    N = env.N
    dim = pi.dim
    horizon = env.ep_len

    # Initialize history arrays
    obs = np.zeros((horizon, N, dim.ob_concat))
    acs = np.zeros((horizon, N, dim.ac))
    rews = np.zeros((horizon, N))
    vpreds = np.zeros((horizon, N))
    # rolling window, starting with zeros
    ob_trajs = np.zeros((horizon, N, dim.window, dim.ob))
    ac_trajs = np.zeros((horizon, N, dim.window, dim.ac))

    for episode in itertools.count():

        ob = env.reset()
        assert ob.shape == (N, dim.ob_concat)

        for step in range(horizon):

            render_every = 25
            if RENDER and episode % render_every == 0:
                env.render()

            obs[step,:,:] = ob

            if test:
                ac, vpred = pi.act_traj(
                    stochastic, ob, ob_trajs[step], ac_trajs[step])
            else:
                ac, vpred = pi.act(stochastic, ob)

            acs[step,:,:] = ac
            vpreds[step,:] = vpred

            if step < horizon - 1:
                ob_trajs[step+1] = np.roll(ob_trajs[step], -1, axis=1)
                ac_trajs[step+1] = np.roll(ac_trajs[step], -1, axis=1)
                ob_trajs[step+1,:,-1,:] = ob[:,:dim.ob]
                ac_trajs[step+1,:,-1,:] = ac

            ob, rew, _, _ = env.step(ac)
            rews[step,:] = rew

        # Episode over.

        # in console we want to print the task reward only
        ep_rews = np.sum(rews, axis=0)

        # evaluate SysID errors and add to the main rewards.
        sysids = obs[0,:,dim.ob:]
        assert np.all((sysids[None,:,:] == obs[:,:,dim.ob:]).flat)
        embed_trues = pi.sysid_to_embedded(sysids)
        embed_estimates = pi.estimate_sysid(
            ob_trajs.reshape((horizon * N, dim.window, dim.ob)),
            ac_trajs.reshape((horizon * N, dim.window, dim.ac)))
        embed_estimates = embed_estimates.reshape((horizon, N, -1))
        err2s = np.mean((embed_trues - embed_estimates) ** 2, axis=-1)
        # apply the err2 for each window to *all* actions in that window
        sysid_rews = 0 * rews
        for i in range(horizon):
            begin = max(i - dim.window, 0)
            sysid_rews[begin:i,:] += err2s[i,:]
        rews += pi.alpha_sysid * sysid_rews
        # TODO keep these separate and let the RL algorithm reason about it?


        # yield the batch to the RL algorithm
        yield {
            "ob" : obs, "rew" : rews, "vpred" : vpreds, "ac" : acs, 
            "ob_traj" : ob_trajs, "ac_traj" : ac_trajs,
            "embed_true" : embed_trues, "embed_estimate" : embed_estimates,
            "ep_rews" : ep_rews, "ep_lens" : horizon + 0 * ep_rews,
        }

        # TODO could make it possible to include more than one reset in a batch
        # without also resampling SysIDs. But is it actually useful?
        env.sample_sysid()
        env.reset()


def add_vtarg_and_adv(seg, gamma, lam):
    rew = seg["rew"]
    vpred = seg["vpred"]
    T, N = rew.shape
    # making the assumption that vpred is a smooth function of (non-sysid) state
    # and the error here is small
    # also assuming no special terminal rewards
    vpred = np.vstack((vpred, vpred[-1,:]))
    gaelam = np.zeros((T + 1, N))
    for t in reversed(range(T)):
        delta = rew[t] + gamma * vpred[t+1] - vpred[t]
        gaelam[t] = delta + gamma * lam * gaelam[t+1]
    vpred = vpred[:-1]
    gaelam = gaelam[:-1]
    seg["adv"] = gaelam
    seg["tdlamret"] = gaelam + vpred
    # TODO: need to trim off T+1 row?


# flattens arrays that are (horizon, N, ...) shape into (horizon * N, ...)
def seg_flatten_batches(seg):
    for s in ("ob", "ac", "ob_traj", "ac_traj", "adv", "tdlamret", "vpred"):
        sh = seg[s].shape
        newshape = [sh[0] * sh[1]] + list(sh[2:])
        seg[s] = np.reshape(seg[s], newshape)
