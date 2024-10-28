#!/usr/bin/env python3
# encoding: utf-8


import torch
import numpy as np

class EarlyStopping:
    def __init__(self, patience=5, threshold=1e-7, threshold_mode='rel'):
        self.patience = patience
        self.threshold = threshold
        self.threshold_mode = threshold_mode
        self.best = None
        self.num_bad_epochs = 0
        self.early_stopped = False

    def __call__(self, metrics, epoch):
        current = metrics
        if self.best is None:
            self.best = current
        if self.threshold_mode == 'rel':
            rel_improve = -(current - self.best) / self.best
            if rel_improve < -self.threshold:
                self.num_bad_epochs += 1
            else:
                self.best = current
                self.num_bad_epochs = 0
        else:
            if current > self.best - self.threshold:
                self.num_bad_epochs += 1
            else:
                self.best = current
                self.num_bad_epochs = 0

        if self.num_bad_epochs >= self.patience:
            self.early_stopped = True
            return True

        return False

# define your network, criterion, optimizer
# ...

