import numpy as np

class CyclicLambdaScheduler():

    def __init__(self, step_size=64, min_lda=0.0, max_lda=1.0, val_lda=None, benchmark=False):
        self.step_size = step_size
        self.min_lda = min_lda
        self.max_lda = max_lda
        self.val_lda = val_lda
        if self.val_lda == None:
            self.val_lda = self.max_lda
        self.benchmark = benchmark
        self.step = 0.0

    def select_lambda(self, training):
        if self.benchmark:
            return 0.0
        if not training:
            return self.val_lda
        assert self.step < 2 * self.step_size
        if self.step < self.step_size:
            lda = self.min_lda + (self.max_lda - self.min_lda) * self.step / self.step_size
            self.step += 1
            return lda
        lda = self.max_lda - (self.max_lda - self.min_lda) * (self.step - self.step_size) / self.step_size
        self.step = (self.step + 1) % (2 * self.step_size)
        return lda

class LinearDecayScheduler():
    def __init__(self, step_size=64, min_lda=0.0, max_lda=1.0, benchmark=False):
        self.step_size = step_size
        self.min_lda = min_lda
        self.max_lda = max_lda
        self.val_lda = (min_lda + max_lda) / 2
        self.benchmark = benchmark
        self.step = 0.0

    def select_lambda(self, training):
        if self.benchmark:
            return 0.0
        if not training:
            return self.val_lda
        assert self.step < self.step_size
        lda = self.max_lda + self.step * (self.min_lda - self.max_lda) / (self.step_size - 1)
        self.step = (self.step + 1) % self.step_size
        return lda

class ExpDecayScheduler():
    def __init__(self, step_size=64, warmup_steps=0, min_lda=1e-7, max_lda=1.0, val_lda=1.0, benchmark=False):
        assert min_lda > 0.0
        self.step_size = step_size
        self.min_lda = min_lda
        self.max_lda = max_lda
        self.val_lda = val_lda
        self.benchmark = benchmark
        self.step = 0.0

    def select_lambda(self, training):
        if self.benchmark:
            return 0.0
        if not training:
            return self.val_lda
        assert self.step < self.step_size
        lda = self.max_lda * np.exp(self.step / (self.step_size - 1) * np.log(self.min_lda / self.max_lda))
        self.step = (self.step + 1) % self.step_size
        return lda

class ExpDecaySchedulerWarmup():
    def __init__(self, step_size=64, warmup_steps=0, min_lda=1e-7, max_lda=1.0, val_lda=1.0, benchmark=False):
        assert min_lda > 0.0
        assert warmup_steps < step_size
        self.step_size = step_size
        self.warmup_steps = warmup_steps
        self.min_lda = min_lda
        self.max_lda = max_lda
        self.val_lda = val_lda
        self.benchmark = benchmark
        self.step = 0.0

    def select_lambda(self, training):
        if self.benchmark:
            return 0.0
        if not training:
            return self.val_lda
        assert self.step < self.step_size
        if self.step < self.warmup_steps:
            lda = self.step * (self.max_lda - self.min_lda) / (self.warmup_steps - 1) + self.min_lda
            self.step += 1
            return lda
        e = (self.step - self.warmup_steps) / (self.step_size - self.warmup_steps - 1)
        lda = self.max_lda * np.exp(e * np.log(self.min_lda / self.max_lda))
        self.step = (self.step + 1) % self.step_size
        return lda