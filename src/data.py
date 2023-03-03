from row import ROW
from cols import COLS
from utils import *
from operator import itemgetter


class DATA:
    def __init__(self, src):
        self.rows = []
        self.cols = None
        if isinstance(src, str):
            csv(src, self.add)
        else:
            for row in src:
                self.add(row)

    def add(self, t):
        if self.cols:
            t = ROW(t) if type(t) == list else t
            self.rows.append(t)
            self.cols.add(t)
        else:
            self.cols = COLS(t)

    def stats(self, what, cols, nPlaces):
        def fun(_, col):
            if what == 'div':
                val = col.div()
            else:
                val = col.mid()
            return col.rnd(val, nPlaces), col.txt

        return kap(cols or self.cols.y, fun)

    def dist(self, row1, row2, cols=None):
        n, d = 0, 0
        for col in cols or self.cols.x:
            n = n + 1
            d = d + col.dist(row1.cells[col.at], row2.cells[col.at]) ** the['p']
        return (d / n) ** (1 / the['p'])

    def clone(self, init={}):
        data = DATA([self.cols.names])
        _ = list(map(data.add, init))
        return data

    def around(self, row1, rows=None, cols=None):
        def function(row2):
            return {'row': row2, 'dist': self.dist(row1, row2, cols)}

        return sorted(list(map(function, rows or self.rows)), key=itemgetter('dist'))

    def half(self, rows=None, cols=None, above=None):
        def dist(row1, row2):
            return self.dist(row1, row2, cols)

        rows = rows or self.rows
        some = many(rows, the['Halves'])
        A = above or any(some)
        B = self.around(A, some)[int(the['Far'] * len(rows)) // 1]['row']
        c = dist(A, B)
        left, right = [], []

        def project(row):
            return {'row': row, 'dist': cosine(dist(row, A), dist(row, B), c)}

        for n, tmp in enumerate(sorted(list(map(project, rows)), key=itemgetter('dist'))):
            if n < len(rows) // 2:
                left.append(tmp['row'])
                mid = tmp['row']
            else:
                right.append(tmp['row'])
        return left, right, A, B, mid, c

    def cluster(self, rows=None, min=None, cols=None, above=None):
        rows = rows or self.rows
        min = min or len(rows) ** the['min']
        cols = cols or self.cols.x
        node = {'data': self.clone(rows)}
        if len(rows) >= 2 * min:
            left, right, node['A'], node['B'], node['mid'], _ = self.half(rows, cols, above)
            node['left'] = self.cluster(left, min, cols, node['A'])
            node['right'] = self.cluster(right, min, cols, node['B'])
        return node

    def better(self, row1, row2):
        s1, s2, ys = 0, 0, self.cols.y
        for col in ys:
            x = col.norm(row1.cells[col.at])
            y = col.norm(row2.cells[col.at])
            s1 = s1 - math.exp(col.w * (x - y) / len(ys))
            s2 = s2 - math.exp(col.w * (y - x) / len(ys))
        return s1 / len(ys) < s2 / len(ys)

    def tree(self, rows=None, min=None, cols=None, above=None):
        rows = rows or self.rows
        min = min or len(rows) ** the['min']
        cols = cols or self.cols.x
        node = {'data': self.clone(rows)}
        if len(rows) >= 2 * min:
            left, right, node['A'], node['B'], node['mid'], _ = self.half(rows, cols, above)
            node['left'] = self.tree(left, min, cols, node['A'])
            node['right'] = self.tree(right, min, cols, node['B'])
        return node

    def sway(self):
        data = self

        def worker(rows, worse, evals0=None, above=None):
            if len(rows) <= len(data.rows) ** the['min']:
                return rows, many(worse, the['rest'] * len(rows)), evals0
            else:
                l, r, A, B, c, evals = self.half(rows, None, above)
                if self.better(B, A):
                    l, r, A, B = r, l, B, A
                for row in r:
                    worse.append(row)
                return worker(l, worse, evals + evals0, A)

        best, rest, evals = worker(data.rows, [], 0)
        return self.clone(best), self.clone(rest), evals
    
    def xpln(self, best, rest):
    
        def v(has):
            return value(has, len(best.rows), len(rest.rows), "best")
        
        def score(ranges, maxSizes):
            rule = RULE(ranges, maxSizes)
            if rule:
                print(showRule(rule))
                bestr = selects(rule, best.rows)
                restr = selects(rule, rest.rows)
                if len(bestr) + len(restr) > 0 :
                    return v({best : len(bestr), rest : len(restr)}), rule
        
        tmp, maxSizes = [], dict()
        for ranges in bins(self.cols.x,{"best" : best.rows, "rest" : rest.rows}):
            maxSizes[ranges[1]["txt"]] = len(ranges)
            print("\n")
            for range in ranges:
                print(range["txt"], range["lo"], range["hi"])
                tmp.append({"range" : range, "max" : len(ranges), "val" : v(range["y"].has)})
        rule, most = firstN(sorted(tmp, key = lambda x: x["val"]), score)
        return rule, most




