import sys

import tensorflow

import keras.backend as K
from keras.legacy import interfaces
from keras.optimizers import Optimizer


def convert_to_accumulate_gradient_optimizer(orig_optimizer, update_params_frequency, accumulate_sum_or_mean=True):
    if update_params_frequency < 1:
        raise ValueError('update_params_frequency must be >= 1')
    print('update_params_frequency: %s' % update_params_frequency)
    print('accumulate_sum_or_mean: %s' % accumulate_sum_or_mean)
    orig_get_gradients = orig_optimizer.get_gradients
    orig_get_updates = orig_optimizer.get_updates
    accumulated_iterations = K.variable(0, dtype='int64', name='accumulated_iterations')
    orig_optimizer.accumulated_iterations = accumulated_iterations

    def updated_get_gradients(self, loss, params):
        return self.accumulate_gradient_accumulators

    def updated_get_updates(self, loss, params):
        self.accumulate_gradient_accumulators = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        updates_accumulated_iterations = K.update_add(accumulated_iterations, 1)
        new_grads = orig_get_gradients(loss, params)
        if not accumulate_sum_or_mean:
            new_grads = [g / K.cast(update_params_frequency, K.dtype(g)) for g in new_grads]
        self.updated_grads = [K.update_add(p, g) for p, g in zip(self.accumulate_gradient_accumulators, new_grads)]
        def update_function():
            with tensorflow.control_dependencies(orig_get_updates(loss, params)):
                reset_grads = [K.update(p, K.zeros(K.int_shape(p), dtype=K.dtype(p))) for p in self.accumulate_gradient_accumulators]
            return tensorflow.group(*(reset_grads + [updates_accumulated_iterations]))
        def just_store_function():
            return tensorflow.group(*[updates_accumulated_iterations])
        
        update_switch = K.equal((updates_accumulated_iterations) % update_params_frequency, 0)
        
        with tensorflow.control_dependencies(self.updated_grads):
            self.updates = [K.switch(update_switch, update_function, just_store_function)]
            return self.updates

    orig_optimizer.get_gradients = updated_get_gradients.__get__(orig_optimizer, type(orig_optimizer))
    orig_optimizer.get_updates = updated_get_updates.__get__(orig_optimizer, type(orig_optimizer))


class AdamAccumulate(Optimizer):

    def __init__(self, lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False, accum_iters=1, **kwargs):
        if accum_iters < 1:
            raise ValueError('accum_iters must be >= 1')
        super(AdamAccumulate, self).__init__(**kwargs)
        with K.name_scope(self.__class__.__name__):
            self.iterations = K.variable(0, dtype='int64', name='iterations')
            self.lr = K.variable(lr, name='lr')
            self.beta_1 = K.variable(beta_1, name='beta_1')
            self.beta_2 = K.variable(beta_2, name='beta_2')
            self.decay = K.variable(decay, name='decay')
        if epsilon is None:
            epsilon = K.epsilon()
        self.epsilon = epsilon
        self.initial_decay = decay
        self.amsgrad = amsgrad
        self.accum_iters = K.variable(accum_iters, K.dtype(self.iterations))
        self.accum_iters_float = K.cast(self.accum_iters, K.floatx())

    @interfaces.legacy_get_updates_support
    def get_updates(self, loss, params):
        grads = self.get_gradients(loss, params)
        self.updates = [K.update_add(self.iterations, 1)]

        lr = self.lr

        completed_updates = K.cast(K.tf.floordiv(self.iterations, self.accum_iters), K.floatx())

        if self.initial_decay > 0:
            lr = lr * (1. / (1. + self.decay * completed_updates))

        t = completed_updates + 1

        lr_t = lr * (K.sqrt(1. - K.pow(self.beta_2, t)) / (1. - K.pow(self.beta_1, t)))

        # self.iterations incremented after processing a batch
        # batch:              1 2 3 4 5 6 7 8 9
        # self.iterations:    0 1 2 3 4 5 6 7 8
        # update_switch = 1:        x       x    (if accum_iters=4)  
        update_switch = K.equal((self.iterations + 1) % self.accum_iters, 0)
        update_switch = K.cast(update_switch, K.floatx())

        ms = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        vs = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        gs = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]

        if self.amsgrad:
            vhats = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        else:
            vhats = [K.zeros((1, )) for _ in params]

        self.weights = [self.iterations] + ms + vs + vhats

        for p, g, m, v, vhat, tg in zip(params, grads, ms, vs, vhats, gs):

            sum_grad = tg + g
            #reest_grad = sum_grad
            avg_grad = sum_grad / self.accum_iters_float
            reest_grad = avg_grad

            m_t = (self.beta_1 * m) + (1. - self.beta_1) * reest_grad
            v_t = (self.beta_2 * v) + (1. - self.beta_2) * K.square(reest_grad)

            if self.amsgrad:
                vhat_t = K.maximum(vhat, v_t)
                p_t = p - lr_t * m_t / (K.sqrt(vhat_t) + self.epsilon)
                self.updates.append(K.update(vhat, (1 - update_switch) * vhat + update_switch * vhat_t))
            else:
                p_t = p - lr_t * m_t / (K.sqrt(v_t) + self.epsilon)

            self.updates.append(K.update(m, (1 - update_switch) * m + update_switch * m_t))
            self.updates.append(K.update(v, (1 - update_switch) * v + update_switch * v_t))
            self.updates.append(K.update(tg, (1 - update_switch) * sum_grad))
            new_p = p_t

            # Apply constraints.
            if getattr(p, 'constraint', None) is not None:
                new_p = p.constraint(new_p)

            self.updates.append(K.update(p, (1 - update_switch) * p + update_switch * new_p))
        return self.updates

    def get_config(self):
        config = {'lr': float(K.get_value(self.lr)),
                  'beta_1': float(K.get_value(self.beta_1)),
                  'beta_2': float(K.get_value(self.beta_2)),
                  'decay': float(K.get_value(self.decay)),
                  'epsilon': self.epsilon,
                  'amsgrad': self.amsgrad}
        base_config = super(AdamAccumulate, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
