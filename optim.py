import numpy as np
class AdamW:

    def __init__(
            self,
            params: dict,
            grads: dict,
            lr=0.001,
            betas: tuple[float, float] = (0.9, 0.999),
            eps: float = 1e-8,
            weight_decay: float = 1e-2,
            amsgrad: bool = False,
    ):
        self.params = params
        self.grads = grads
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.amsgrad = amsgrad

        self.n_iters = 0

        self.m = {k: np.zeros_like(v) for k, v in params.items()}

        self.v = {k: np.zeros_like(v) for k, v in params.items()}

        self.v_m = (
            {k: np.zeros_like(v) for k, v in params.items()}
            if amsgrad
            else None
        )

    def zero_grad(self):
        for v in self.grads.values():
            v[:] = 0


    def step(self):

        self.n_iters += 1

        beta1, beta2 = self.betas

        for (name, param), grad in zip(
                self.params.items(), self.grads.values()
        ):

            m_t = self.m[name] = beta1 * self.m[name] + (1 - beta1) * grad

            v_t = self.v[name] = beta2 * self.v[name] + (1 - beta2) * (
                    grad**2
            )



            m_t_hat = m_t / (1 - beta1**self.n_iters)


            v_t_hat = v_t / (1 - beta2**self.n_iters)

            if self.amsgrad:
                v_t_hat = self.v_m[name] = np.maximum(self.v_m[name], v_t_hat)



            g_hat = m_t_hat / (np.sqrt(v_t_hat) + self.eps)

            update = g_hat + self.weight_decay * param

            self.params[name] -= self.lr * update
