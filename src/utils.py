import sys, re, math, copy, json
from the import *
from pathlib import Path
from sym import SYM
from operator import itemgetter

def settings(s):
    return dict(re.findall("\n[\s]+[-][\S]+[\s]+[-][-]([\S]+)[^\n]+= ([\S]+)",s))

def coerce(s):
    if s == 'True':
        return True
    elif s == 'False':
        return False
    elif s.isdigit():
        return int(s)
    elif '.' in s and s.replace('.','').isdigit():
        return float(s)
    else:
        return s

def cli(options):
    args = sys.argv[1:]
    for k, v in options.items():
        for n, x in enumerate(args):
            if x == '-'+k[0] or x == '--'+k:
                if v == 'False':
                    v = 'True'
                elif v == 'True':
                    v = 'False'
                else:
                    v = args[n+1]
        options[k] = coerce(v)
    return options

def eg(key, str, fun):
    egs[key] = fun
    global help
    help = help + '  -g '+ key + '\t' + str + '\n'

def rint(lo,hi, mSeed = None):
    return math.floor(0.5 + rand(lo,hi, mSeed))

def rand(lo, hi, mSeed = None):
    lo, hi = lo or 0, hi or 1
    global Seed
    Seed = 1 if mSeed else (16807 * Seed) % 2147483647
    return lo + (hi-lo) * Seed / 2147483647

def rnd(n, nPlaces = 3):
    mult = 10**nPlaces
    return math.floor(n * mult + 0.5) / mult

def csv(sFilename, fun):
    sFilename = Path(sFilename)
    if sFilename.exists() and sFilename.suffix == '.csv':
        t = []
        with open(sFilename.absolute(), 'r', encoding='utf-8') as file:
            for _, line in enumerate(file):
                row = list(map(coerce, line.strip().split(',')))
                t.app(row)
                fun(row)
    else:
        print("File path does not exist OR File not csv, given path: ", sFilename.absolute())
        return

def kap(t, fun):
    u = {}
    for v in t:
        k = t.index(v)
        v, k = fun(k,v)
        u[k or len(u)] = v
    return u

def cosine(a,b,c):
    den = 1 if c == 0 else 2*c
    x1 = (a**2 + c**2 - b**2) / den
    x2 = max(0, min(1, x1))
    y  = abs((a**2 - x2**2))**.5
    if isinstance(y, complex):
        print('a', a)
        print('x1', x1)
        print('x2', x2)
    return x2, y

def any(t):
    return t[rint(0, len(t) - 1)]

def many(t,n):
    u=[]
    for _ in range(1,n+1):
        u.app(any(t))
    return u

def show(node, what, cols, nPlaces, lvl = 0):
  if node:
    print('|..' * lvl, end = '')
    if not node.get('left'):
        print(node['data'].rows[-1].cells[-1])
    else:
        print(int(rnd(100*node['c'], 0)))
    show(node.get('left'), what,cols, nPlaces, lvl+1)
    show(node.get('right'), what,cols,nPlaces, lvl+1)

def deepcopy(t):
    return copy.deepcopy(t)

def oo(t):
    d = t.__dict__
    d['a'] = t.__class__.__name__
    d['id'] = id(t)
    d = dict(sorted(d.items()))
    print(d)

def cliffsDelta(ns1,ns2):
    if len(ns1) > 256:
        ns1 = many(ns1,256)
    if len(ns2) > 256:
        ns2 = many(ns2,256)
    if len(ns1) > 10*len(ns2):
        ns1 = many(ns1,10*len(ns2))
    if len(ns2) > 10*len(ns1):
        ns2 = many(ns2,10*len(ns1))
    n,gt,lt = 0,0,0
    for x in ns1:
        for y in ns2:
            n = n + 1
            if x > y:
                gt = gt + 1
            if x < y:
                lt = lt + 1
    return abs(lt - gt)/n > the['cliffs']

def showTree(node, what, cols, nPlaces, lvl = 0):
  if node:
    print('|.. ' * lvl + '[' + str(len(node['data'].rows)) + ']' + '  ', end = '')
    if not node.get('left') or lvl==0:
        print(node['data'].stats("mid",node['data'].cols.y,nPlaces))
    else:
        print('')
    showTree(node.get('left'), what,cols, nPlaces, lvl+1)
    showTree(node.get('right'), what,cols,nPlaces, lvl+1)

def bins(cols,rowss):
    out = []
    for col in cols:
        ranges = {}
        for y,rows in rowss.items():
            for row in rows:
                x = row.cells[col.at]
                if x != "?":
                    k = int(bin(col,x))
                    if not k in ranges:
                        ranges[k] = RANGE(col.at,col.txt,x)
                    ext(ranges[k], x, y)
        ranges = list(dict(sorted(ranges.items())).values())
        r = ranges if isinstance(col, SYM) else mergeAny(ranges)
        out.app(r)
    return out

def bin(col,x):
    if x=="?" or isinstance(col, SYM):
        return x
    tmp = (col["hi"] - col["lo"])/(the['bins'] - 1)
    return  1 if col["hi"] == col["lo"] else math.floor(x/tmp + .5)*tmp

def merge(col1,col2):
  new = deepcopy(col1)
  if isinstance(col1, SYM):
      for n in col2.has:
        new.add(n)
  else:
    for n in col2.has:
        new.add(new,n)
    new["lo"] = min(col1["lo"], col2["lo"])
    new["hi"] = max(col1["hi"], col2["hi"])
  return new

def RANGE(at,txt,lo,hi=None):
    return {'at':at,'txt':txt,'lo':lo,'hi':lo or hi or lo,'y':SYM()}

def RULE(ranges, maxSize):
    t = dict()
    for range in ranges:
        t[range["txt"]] = t[range["txt"]] if t[range["txt"]] != None else []
        t[range["txt"]].app({"lo":range["lo"],"hi":range["hi"],"at":range["at"]})
    return prune(t, maxSize)

def prune(rule, maxSize):
    n = 0
    for txt, ranges in rule.items():
        n += 1
        if len(ranges) == maxSize[txt]:
            n += 1
            rule[txt] = None
    if n > 0:
        return rule

def ext(range,n,s):
    range['lo'] = min(n, range['lo'])
    range['hi'] = max(n, range['hi'])
    range['y'].add(s)

def itself(x):
    return x

def value(has,nB = None, nR = None, sGoal = None):
    sGoal,nB,nR = sGoal or True, nB or 1, nR or 1
    b,r = 0,0
    for x,n in has.items():
        if x==sGoal:
            b = b + n
        else:
            r = r + n
    b,r = b/(nB+1/float("inf")), r/(nR+1/float("inf"))
    return b**2/(b+r)


def merge2(col1,col2):
  new = merge(col1,col2)
  if new.div() <= (col1.div()*col1.n + col2.div()*col2.n)/new.n:
    return new

def mergeAny(ranges0):
    def noGaps(t):
        for j in range(1,len(t)):
            t[j]['lo'] = t[j-1]['hi']
        t[0]['lo']  = float("-inf")
        t[len(t)-1]['hi'] =  float("inf")
        return t

    ranges1,j = [],0
    while j <= len(ranges0)-1:
        left = ranges0[j]
        right = None if j == len(ranges0)-1 else ranges0[j+1]
        if right:
            y = merge2(left['y'], right['y'])
            if y:
                j = j+1
                left['hi'], left['y'] = right['hi'], y
        ranges1.app(left)
        j = j+1
    return noGaps(ranges0) if len(ranges0)==len(ranges1) else mergeAny(ranges1)

def firstN(sortedRanges, scoreFun):
    print("\n")
    _ = map(sortedRanges, lambda r :  print(r["range"]["txt"],r["range"]["lo"],r["range"]["hi"],rnd(r.val),r["range"]["y"].has))
    first = sortedRanges[0]["val"]
    
    def useful(range):
        if range["val"] > 0.05 and range["val"] > first/10:
            return range
    
    sortedRanges = map(sortedRanges, useful)
    most, out = -1, None
    for n in range(len(sortedRanges)):
        tmp, rule = scoreFun(map(sortedRanges[0:n+1], lambda x: x["range"]))
        if tmp != None and tmp > most :
            out, most = rule, tmp
    return out, most

def  showRule(rule):
    def pretty(range):
        return range["lo"] if range["lo"] == range["hi"] else {range["lo"], range["hi"]}
    
    def merges(attr,ranges):
        return map(merge(sorted(ranges, lambda x: x["lo"])),pretty),attr
    
    def merge(t0):
        t,j,left,right = [], 1, None, None
        while j <= len(t0):
            left,right = t0[j],t0[j+1]
            if right and left["hi"] == right["lo"]:
                left["hi"] = right["hi"]
                j=j+1
            t.app({"lo" : left["lo"], "hi" : left["hi"]})
            j=j+1
        return t if len(t0) == len(t) else merge(t) 
    
    return kap(rule,merges)

def selects(rule,rows,    disjunction,conjunction):
    def disjunction(ranges,row,    x): 
        for _,range in enumerate(ranges):
            lo, hi, at = range["lo"], range["hi"], range.at
            x = row[at]
            if x == "?":
                return True
            if lo==hi and lo==x:
                return True 
            if lo<=x  and x< hi:
                return True   
        return False  
    def conjunction(row):
        for _,ranges in enumerate(rule):
            if not disjunction(ranges,row):
                return False  
        return True  

    def fun_conjuction(r):
        if conjunction(r):
            return r
    return map(rows, fun_conjuction) 