import numpy as np
from scipy.spatial.distance import cdist



h = [i for i in range(10)]
b = [i for i in range(10, 20)]
g=[h, b]
bh = np.vstack(g)


a = [i for i in range(10)]
b = [i for i in range(10,20)]
c = [i for i in range(100, 110)]
abc = np.vstack((a,b,c))


print(cdist(g, abc))