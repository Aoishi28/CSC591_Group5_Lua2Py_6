import math


class SYM:
    def __init__(self, at=None, txt=None):
        self.at = at if at else 0
        self.txt = txt if txt else ""
        self.n = 0
        self.has = {}
        self.most = 0
        self.mode = None

    def add(self, x):
        if not x == "?":
            self.n += 1
            self.has[x] = 1 + (self.has[x] if x in self.has.keys() else 0)
            if self.has[x] > self.most:
                self.most = self.has[x]
                self.mode = x

    def mid(self):
        return self.mode

    def div(self):
        def fun(p):
            return p * math.log2(p)

        e = 0
        for _, n in self.has.items():
            e = e + fun(n / self.n)
        return -e

    def rnd(self, x, n):
        return x

    def dist(self, s1, s2):
        if s1 == "?" and s2 == "?":
            return 1
        elif s1 == s2:
            return 0
        else:
            return 1