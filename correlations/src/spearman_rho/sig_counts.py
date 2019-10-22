import pandas as pd
import collections
import operator

# display adjustments
pd.set_option('display.width', None)
pd.options.display.max_rows = 500
pd.options.display.max_columns = 33

cd projects/
cd gbm_immunosuppression/
cd correlations/
cd data/
cd spearman_correlation_analysis/
cd orthant/
cd significant_corrs_tables/
cd gl261_unfiltered vs. naive_unfiltered (naive_only)

naive = pd.read_csv('gl261_unfiltered vs. naive_unfiltered (naive_only).csv')

cd ..
cd gl261_unfiltered vs. naive_unfiltered (gl261_only)
gl261 = pd.read_csv('gl261_unfiltered vs. naive_unfiltered (gl261_only).csv')

nh = naive.head(n=10)
nt = naive.tail(n=10)
gh = gl261.head(n=10)
gt = gl261.tail(n=10)

top_n = nh.append(nt)
top_g = gh.append(gt)

n_all = top_n['x'].append(top_n['y'])
g_all = top_g['x'].append(top_g['y'])

a = n_all.append(g_all)
b = [i.split('_', 2)[0] for i in a]

mydict = {}
for i in set(b):
    num = b.count(i)
    mydict[i] = num

sorted_dict = sorted(mydict.items(), key=operator.itemgetter(1))


nn = [i.split('_', 2)[0] for i in n_all]
ndict = {}
for i in set(nn):
    num = nn.count(i)
    ndict[i] = num

n_dict = sorted(ndict.items(), key=operator.itemgetter(1))


gg = [i.split('_', 2)[0] for i in g_all]
gdict = {}
for i in set(gg):
    num = gg.count(i)
    gdict[i] = num

g_dict = sorted(gdict.items(), key=operator.itemgetter(1))
