import sys, re, math, copy, json
from the import *
from pathlib import Path
from sym import SYM
from operator import itemgetter

def settings(s):
    return dict(re.findall("\n[\s]+[-][\S]+[\s]+[-][-]([\S]+)[^\n]+= ([\S]+)",s))

def coerce(s):
    if s == 'true':
        return True
    elif s == 'false':
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
                if v == 'false':
                    v = 'true'
                elif v == 'true':
                    v = 'false'
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
                t.append(row)
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
        u.append(any(t))
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
                    extend(ranges[k], x, y)
        ranges = list(dict(sorted(ranges.items())).values())
        r = ranges if isinstance(col, SYM) else mergeAny(ranges)
        out.append(r)
    return out

def bin(col,x):
    if x=="?" or isinstance(col, SYM):
        return x
    tmp = (col.hi - col.lo)/(the['bins'] - 1)
    return  1 if col.hi == col.lo else math.floor(x/tmp + .5)*tmp

def merges(ranges0,nSmall,nFar):

    def noGaps(t):
        for j in range(1,len(t)+1):
            t[j]['lo'] = t[j-1]['hi']
        t[0]['lo'] = -math.inf
        t[len(t)-1]['hi'] = math.inf
        return t

    def try2Merge(left,right,j):
        y = merged(left['y'],right['y'],nSmall,nFar)
        if y:
            j = j+1
            left['hi'] , left['y'] = right['hi'], y
        return j,left
    
    ranges1, j, here = [], 1, None
    while j <= len(ranges0):
        here = ranges0[j-1]
        if j < len(ranges0):
            j, here = try2Merge(here, ranges0[j], j)
        j += 1
        ranges1.append(here)
    return (len(ranges0) == len(ranges1) and noGaps(ranges0)) or merges(ranges1, nSmall, nFar)

def merged(col1,col2,nSmall,nFar):
    new = merge(col1,col2)
    if nSmall and col1['n'] < nSmall or col2.n < nSmall:
        return new
    if nFar and not col1.isSym and abs(col1.mid() - col2.mid()) <nFar:
        return new
    if new.div() <= (col1.div() * col1['n'] + col2.div() * col2['n']) / new['n']:
        return new

def merge(col1,col2):
  new = deepcopy(col1)
  if isinstance(col1, SYM):
      for n in col2.has:
        new.add(n)
  else:
    for n in col2.has:
        new.add(new,n)
    new.lo = min(col1.lo, col2.lo)
    new.hi = max(col1.hi, col2.hi)
  return new

def RANGE(at,txt,lo,hi=None):
    return {'at':at,'txt':txt,'lo':lo,'hi':lo or hi or lo,'y':SYM()}

def RULE(ranges,maxSize):
    t = {}
    for range in ranges:
        t[range['txt']] = t[range['txt']] if t[range['txt']] else []
        t[range['txt']].append({'lo':range['lo'],'hi':range['hi'],'at':range['at'] })
    return prune(t,maxSize)

def prune(rule,maxSize):
    n = 0
    for txt,ranges in enumerate(rule):
        n = n+1
        if len(ranges) == maxSize[txt]:
            n = n+1
            rule[txt] = None
    if (n>0) :
        return rule

def extend(range,n,s):
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


