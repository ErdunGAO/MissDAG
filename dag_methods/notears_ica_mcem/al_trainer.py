import logging
import numpy as np
import tensorflow as tf

class ALTrainer(object):
    """
    Augmented Lagrangian method with gradient-based optimization
    """
    _logger = logging.getLogger(__name__)

    def __init__(self, init_rho, rho_max, h_factor, rho_multiply, init_iter, learning_rate, h_tol):
        self.init_rho = init_rho
        self.rho_max = rho_max
        self.h_factor = h_factor
        self.rho_multiply = rho_multiply
        self.init_iter = init_iter
        self.learning_rate = learning_rate
        self.h_tol = h_tol

    def train(self, model, X, weight_index, max_iter, iter_step):

        model.sess.run(tf.compat.v1.global_variables_initializer())
        rho, alpha, h, h_new = self.init_rho, 0.0, np.inf, np.inf

        self._logger.info('Started training for {} iterations'.format(max_iter))
        for epoch in range(1, max_iter + 1):
            while rho < self.rho_max:
                self._logger.info('rho {:.3E}, alpha {:.3E}'.format(rho, alpha))
                loss_new, mle_new, h_new, W_new = self.train_step(model, iter_step, X, weight_index, rho, alpha)
                if h_new > self.h_factor * h:
                    rho *= self.rho_multiply
                else:
                    break

            W_est, h = W_new, h_new
            alpha += rho * h

            if h <= self.h_tol and epoch > self.init_iter:
                self._logger.info('Early stopping at {}-th iteration'.format(epoch))
                break

        model.sess.close()
        return W_est

    def train_step(self, model, iter_step, X, weight_index, rho, alpha):
        for _ in range(iter_step):
            _, curr_loss, curr_mle, curr_h, curr_W \
                = model.sess.run([model.train_op, model.loss, model.mle, model.h, model.W_prime],
                                 feed_dict={model.X: X,
                                            model.weight_index: weight_index,
                                            model.rho: rho,
                                            model.alpha: alpha,
                                            model.lr: self.learning_rate})

        return curr_loss, curr_mle, curr_h, curr_W
