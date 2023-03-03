#!/usr/bin/env python3

from utils import *
from test_engine import *

def main():
    y,n,saved = 0,0,deepcopy(the)
    for k,v in cli(settings(help)).items():
        the[k] = v
        saved[k] = v
    if the['help'] == True:
        print(help)
    else:
        for what, fun in egs.items():
            if the['go'] == 'all' or the['go'] == what:
                for k,v in saved.items():
                    the[k] = v
                Seed = the['seed']
                print('▶️ ',what,("-")*(60))
                if egs[what]() == False:
                    n += 1
                    print('❌ fail:', what)
                else:
                    y += 1
                    print('✅ pass:', what)
    if y+n>0:
        print("🔆",{'pass' : y, 'fail' : n, 'success' :100*y/(y+n)//1})
    sys.exit(n)

if __name__ == '__main__':
    eg('the', 'show options', test_the)
    eg('rand', 'demo random number generation', test_rand)
    eg('some', 'demo of reservoir sampling', test_some)
    eg('nums', 'demo of NUM', test_num)
    eg('sym', 'demo SYMS', test_sym)
    eg('csv', 'reading csv files', test_csv)
    eg('data', 'showing DATA sets', test_data)
    eg('clone', 'replicate structure of a DATA', test_clone)
    eg('cliffs', 'stat tests', test_cliffs)
    eg('dist', 'distance test', test_dist)
    eg('half', 'divide data in half', test_half)
    eg('tree', 'make snd show tree of clusters', test_tree)
    eg('sway', 'optimizing', test_sway)
    eg('bins', 'find deltas between best and rest', test_bins)
    eg('xpln', 'explore explanation sets', test_xpln)
    main()