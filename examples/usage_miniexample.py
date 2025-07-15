import nacvi
import numpy as np

labels1=np.asarray([0,1,-1])
data1=np.asarray([[0,0],[1,1],[-1,-1]])
val1=nacvi.sil_plus_score(data1,labels1)
print(val1)

labels2=np.asarray([0,1,-1])
data2=np.asarray([[0,0],[1,1],[0,0]])
val2=nacvi.sil_plus_score(data2,labels2)
print(val2)