import numpy as np
import matplotlib.pyplot as plt

def embed_explore(env, pi, ob_reps, sysid_true, sysid_est):

	N = ob_reps.shape[0]

	# scatter the actual SysID params with the embedding
	plt.figure(1)
	plt.clf()
	sysid_reps = ob_reps[:,pi.ob_dim:]
	assert sysid_reps.shape == (N, 4)
	for i, name in enumerate(env.sysid_params.keys()):
		plt.subplot(3, 2, i + 1)
		#plt.scatter(sysid_est, sysid_reps[:,i])
		#order = np.argsort(sysid_true.flatten())
		#plt.plot(sysid_true.flatten()[order], sysid_reps[order,i])
		plt.scatter(sysid_true.flatten(), sysid_reps[:,i])
		plt.xlabel('embedding')
		plt.ylabel(name)

	# scatter actual and estimated embedding
	plt.subplot(3, 2, 5)
	plt.scatter(sysid_true, sysid_est)
	plt.ylim(plt.xlim())
	plt.xlabel('true embed')
	plt.ylabel('est embed')

	# hold the observation constant, vary the embedding, see how the action changes
	plt.figure(2)
	plt.clf()
	n_obs = 6
	i_obs = np.random.choice(N, n_obs, replace=False)
	for i, ind in enumerate(i_obs):
		k = 50
		embed_mid = sysid_reps[ind][0]
		embed_range = np.linspace(embed_mid - 3, embed_mid + 3, num=k).reshape(-1, 1)
		ob = np.tile(ob_reps[ind,:], (k, 1))
		stochastic = False
		ac = pi.act_embed(stochastic, ob, embed_range)
		plt.subplot(2,3,i+1)
		plt.plot(embed_range.flatten(), ac)



	plt.show(block=False)
	plt.pause(0.001)
