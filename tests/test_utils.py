import sys
sys.path.append("./src")
from utils import *
from num import NUM
from sym import SYM
from data import DATA

def test_the():
    print(the.__repr__())

def test_rand():
    Seed = 1
    t=[]
    for i in range(1,1000+1):
        t.append(rint(0,100,1))
    u=[]
    for i in range(1,1000+1):
        u.append(rint(0,100,1))
    for k,v in enumerate(t):
        assert(v==u[k])

def test_some():
    the['Max'] = 32
    num1 = NUM()
    for i in range(1,1000+1):
        num1.add(i)
    print(num1.has)

def test_num():
    num1, num2 = NUM(), NUM()
    global Seed
    Seed = the['seed']
    for i in range(1,10**3+1):
        num1.add(rand(0,1))
    Seed = the['seed']
    for i in range(1,10**3+1):
        num2.add(rand(0,1)**2)
    m1,m2 = rnd(num1.mid(),1), rnd(num2.mid(),1)
    d1,d2 = rnd(num1.div(),1), rnd(num2.div(),1)
    print(1, m1, d1)
    print(2, m2, d2)
    return m1 > m2 and .5 == rnd(m1,1)

def test_sym():
    sym = SYM()
    for x in ["a","a","a","a","b","b","c"]:
        sym.add(x)
    print(sym.mid(), rnd(sym.div()))
    return 1.379 == rnd(sym.div())

def no_of_chars_in_file(t):
    global n
    n += len(t)

def test_csv():
    csv(the['file'], no_of_chars_in_file)
    return n == 3192

def test_data():
    data = DATA(the['file'])
    col=data.cols.x[1]
    print(col.lo,col.hi, col.mid(),col.div())
    print(data.stats('mid', data.cols.y, 2))

def test_clone():
    data1 = DATA(the['file'])
    data2 = data1.clone(data1.rows)
    print(data1.stats('mid', data1.cols.y, 2))
    print(data2.stats('mid', data2.cols.y, 2))

def test_cliffs():
    assert(False == cliffsDelta( [8,7,6,2,5,8,7,3],[8,7,6,2,5,8,7,3]))
    assert(True  == cliffsDelta( [8,7,6,2,5,8,7,3], [9,9,7,8,10,9,6]))
    t1,t2=[],[]
    for i in range(1,1000+1):
        t1.append(rand(0,1))
    for i in range(1,1000+1):
        t2.append(rand(0,1)**.5)
    assert(False == cliffsDelta(t1,t1))
    assert(True  == cliffsDelta(t1,t2))
    diff,j=False,1.0
    while not diff:
        def function(x):
            return x*j
        t3=list(map(function, t1))
        diff=cliffsDelta(t1,t3)
        print(">",rnd(j),diff)
        j=j*1.025

def test_dist():
    data = DATA(the['file'])
    num  = NUM()
    for row in data.rows:
        num.add(data.dist(row, data.rows[1]))
    print({'lo' : num.lo, 'hi' : num.hi, 'mid' : rnd(num.mid()), 'div' : rnd(num.div())})

def test_half():
    data = DATA(the['file'])
    left,right,A,B,c,_ = data.half()
    print(len(left),len(right))
    l,r = data.clone(left), data.clone(right)
    print("l",l.stats('mid', l.cols.y, 2))
    print("r",r.stats('mid', r.cols.y, 2))

def test_tree():
    data = DATA(the['file'])
    showTree(data.tree(),"mid",data.cols.y,1)
    return True

def test_sway():
    data = DATA(the['file'])
    best,rest,_ = data.sway()
    print("\nall ", data.stats('mid', data.cols.y, 2))
    print("    ", data.stats('div', data.cols.y, 2))
    print("\nbest",best.stats('mid', best.cols.y, 2))
    print("    ", best.stats('div', best.cols.y, 2))
    print("\nrest", rest.stats('mid', rest.cols.y, 2))
    print("    ", rest.stats('div', rest.cols.y, 2))
    return True

def test_bins():
    global b4
    data = DATA(the['file'])
    best,rest,_ = data.sway()
    print("all","","","",{'best':len(best.rows), 'rest':len(rest.rows)})
    for k,t in enumerate(bins(data.cols.x,{'best':best.rows, 'rest':rest.rows})):
        for range in t:
            if range['txt'] != b4:
                print("")
            b4 = range['txt']
            print(range['txt'],range['lo'],range['hi'],
            rnd(value(range['y'].has, len(best.rows),len(rest.rows),"best")),
            range['y'].has)

def test_xpln():
    data = DATA(the['file'])
    best,rest,evals = data.sway()
    rule,most= data.xpln(best,rest)
    print("\n-----------\nexplain=", data.showRule(rule))
    selects = data.selects(rule,data.rows)
    data_selects = [s for s in selects if s!=None]
    data1= data.clone(data_selects)
    print("all               ",data.stats('mid', data.cols.y, 2),data.stats('div', data.cols.y, 2))
    print("sway with",evals,"evals",best.stats('mid', best.cols.y, 2),best.stats('div', best.cols.y, 2))
    print("xpln on",evals,"evals",data1.stats('mid', data1.cols.y, 2),data1.stats('div', data1.cols.y, 2))
    top,_ = data.betters(len(best.rows))
    top = data.clone(top)
    print("sort with",len(data.rows),"evals",top.stats('mid', top.cols.y, 2),top.stats('div', top.cols.y, 2))