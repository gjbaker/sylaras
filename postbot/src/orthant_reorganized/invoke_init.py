# invoke required libraries
import pandas as pd
import sys
import os
import glob
import numpy as np
from pyeda.inter import exprvar
from pyeda.inter import iter_points
from pyeda.inter import expr
from pyeda.inter import expr2truthtable
import collections
import itertools
import scipy.stats
from scipy.stats import ttest_ind
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.ticker import AutoMinorLocator
from datetime import datetime
import math
from itertools import cycle, islice
import matplotlib.gridspec as gridspec
from matplotlib.gridspec import GridSpec
from operator import itemgetter
from decimal import Decimal
from rpy2.robjects.packages import importr
from rpy2.robjects.vectors import FloatVector
from inspect import getmembers, isclass
import pickle
import shelve

# map matplotlib color codes to the default seaborn palette
sns.set()
sns.set_color_codes()
_ = plt.plot([0, 1], color='r')
sns.set_color_codes()
_ = plt.plot([0, 2], color='b')
sns.set_color_codes()
_ = plt.plot([0, 3], color='g')
sns.set_color_codes()
_ = plt.plot([0, 4], color='m')
sns.set_color_codes()
_ = plt.plot([0, 5], color='y')
plt.close('all')

# display adjustments
pd.set_option('display.width', None)
pd.options.display.max_rows = 150
pd.options.display.max_columns = 33