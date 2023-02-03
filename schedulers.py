import numpy as np

class CyclicLambdaScheduler():

    def __init__(self, step_size=64, min_lda=0.0, max_lda=1.0):
        self.step_size = step_size
        self.min_lda = min_lda
        self.max_lda = max_lda
        self.step = 0.0

    def select_lambda(self, training):
        if not training:
            return self.max_lda
        assert self.step < 2 * self.step_size
        if self.step < self.step_size:
            lda = self.min_lda + (self.max_lda - self.min_lda) * self.step / self.step_size
            self.step += 1
            return lda
        lda = self.max_lda - (self.max_lda - self.min_lda) * (self.step - self.step_size) / self.step_size
        self.step = (self.step + 1) % (2 * self.step_size)
        return lda