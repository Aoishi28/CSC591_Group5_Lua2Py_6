import math
from utils import rnd
from the import *

class NUM:
    def __init__(self, at=0, txt=""):
        self.at = at
        self.txt = txt
        self.n = 0
        self.mu = 0
        self.m2 = 0
        self.lo = float('inf')
        self.hi = float('-inf')
        self.w = -1 if "-" in self.txt else 1
        self.has = {}

    def add(self, n):
        if n != "?":
            self.n += 1
            if self.n <= the['Max']:
                self.has[n]= n
            d = n - self.mu
            self.mu += d / self.n
            self.m2 += d * (n - self.mu)
            self.lo = min(n, self.lo)
            self.hi = max(n, self.hi)

    def mid(self):
        return self.mu

    def div(self):
        if (self.m2 < 0 or self.n < 2):
            return 0
        else:
            return math.sqrt(self.m2 / (self.n - 1))

    def rnd(self, x, n):
        if x == "?":
            return x
        else:
            return rnd(x, n)

    def norm(self, n):
        return n if n == "?" else (n - self.lo) / (self.hi - self.lo + 1e-32)

    def dist(self, n1, n2):
        if n1 == "?" and n2 == "?":
            return 1
        n1, n2 = self.norm(n1), self.norm(n2)
        if n1 == "?":
            n1 = 1 if n2 < 0.5 else 0
        if n2 == "?":
            n2 = 1 if n1 < 0.5 else 0
        return abs(n1 - n2)


