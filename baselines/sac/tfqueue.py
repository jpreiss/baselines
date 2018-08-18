import tensorflow as tf
import numpy as np
import batch_util2 as batch2
import time
import os
import _thread
import matplotlib.pyplot as plt


DIM_IN = 10
DIM_OUT = 1
BATCH = 256
ROLLOUT = 500
EPISODES = 20
BUF = int(1e6)
HID = (32, 32)
LR = 1e-3



do_train = True


def main():

    print("my PID:", os.getpid())

    buf = batch2.ReplayBuffer(BUF, (DIM_IN, DIM_OUT))
    buf_mtx = _thread.allocate_lock()


    # fill buffer - fake interacting with the environment
    def buf_producer(n_rollouts):

        global do_train

        # random convex quadratic function
        root_hessian = 0.01 * np.random.normal(size=(DIM_IN, DIM_IN))

        for i in range(n_rollouts):
            x = np.random.uniform(-4, 4, size=(ROLLOUT, DIM_IN))
            y = np.sum((x @ root_hessian) ** 2, axis=1)
            time.sleep(0.1) # seconds
            assert y.shape == (ROLLOUT,)
            with buf_mtx:
                print(f"added a batch.")
                buf.add_batch(x, y[:,None])

        # episodes done. terminate
        with buf_mtx:
            do_train = False


    # generator to draw shuffled samples from the buffer.
    # TODO: is there any way to get tf to do the random sampling
    #       that allows the circular buffer to mutate constantly?
    def buf_sampler():

        global do_train

        npr = np.random.RandomState()
        while True:
            with buf_mtx:
                if buf.size >= BATCH:
                    break
            time.sleep(0.01)

        while True:
            with buf_mtx:
                if not do_train:
                    break
                batch = buf.sample(npr, BATCH)
            yield tuple(batch)

    # Dataset for the buf_sampler
    #shapes = [tf.TensorShape((d,)) for d in (DIM_IN, DIM_OUT)]
    shapes = ((BATCH, DIM_IN), (BATCH, DIM_OUT))
    dtypes = (tf.float32, tf.float32)
    generator = tf.data.Dataset.from_generator(buf_sampler, dtypes, shapes)
    iter = generator.make_one_shot_iterator()
    batch_x, batch_y = iter.get_next()


    # 2-layer ReLU MLP for fn approximation
    x = batch_x
    for i, sz in enumerate(HID):
        x = tf.layers.dense(x, sz, activation=tf.nn.relu, name=f"fc_{i}")
    y_est = tf.layers.dense(x, DIM_OUT, name="fc_out")
    loss = tf.losses.mean_squared_error(y_est, batch_y)
    adam = tf.train.AdamOptimizer(LR).minimize(loss)


    # training loop - nondeterministic # of gradient steps
    def train(sess):

        global do_train

        losses = []
        while True:
            with buf_mtx:
                if not do_train:
                    break
            try:
                batch_loss, _ = sess.run([loss, adam])
                losses.append(batch_loss)
                print(f"training step {len(losses)}")
            except tf.errors.OutOfRangeError:
                time.sleep(0.1)
        return losses


    _thread.start_new_thread(buf_producer, (EPISODES,))
    _thread.start_new_thread(buf_sampler, ())
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        losses = train(sess)

    print(f"completed {len(losses)} training steps. final rmse = {np.sqrt(losses[-1])}.")
    plt.plot(losses)
    plt.show()


main()
