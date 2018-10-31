import math


def delta_default(n: int):
    return min(.01, n ** (-0.5))


class TuningParam:
    def __init__(self, value, acceptance_count: int=0, current_iter: int=0,
                 batch_size: int=50, delta=delta_default,
                 target_acceptance_rate: float=0.44):
        self.value = value
        self.acceptance_count = acceptance_count
        self.current_iter = current_iter
        self.batch_size = batch_size
        self.delta = delta
        self.target_acceptance_rate = target_acceptance_rate

    def update(self, accept: bool):
        if accept:
            self.acceptance_count += 1

        self.current_iter += 1

        if self.current_iter % self.batch_size == 0:
            n = math.floor(self.current_iter / self.batch_size)
            factor = math.exp(self.delta(n))
            if self.acceptance_rate() > self.target_acceptance_rate:
                self.value *= factor
            else:
                self.value /= factor

            self.acceptance_count += 1

    def acceptance_rate(self):
        return self.acceptance_count / self.batch_size
