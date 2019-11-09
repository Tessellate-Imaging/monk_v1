import os
import sys
import random
import numpy as np
from matplotlib import pyplot as plt
import matplotlib._color_data as mcd
import matplotlib.patches as mpatch
from textwrap import wrap


overlap = {name for name in mcd.CSS4_COLORS if "xkcd:" + name in mcd.XKCD_COLORS}
overlap = list(overlap)
random.shuffle(overlap)

