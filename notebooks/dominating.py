import numpy as np
import faiss

def dominating_dataset(X,y,epsilon=1):
    
    if epsilon <=0:
        return np.arange(len(y))
    else: 
        picks = np.array([],dtype=int) 
        classes = np.unique(y)
    
        for cl in classes:
        
            pool_cl = np.where(y==cl)
            X_cl = X[pool_cl]
            pool_cl = np.reshape(pool_cl,(-1,))
            lenpool_cl, dim = np.shape(X_cl)
            pool_aux = [1]*lenpool_cl
            
            index = faiss.IndexFlatL2(dim)
            index.add(X_cl)
            
            while lenpool_cl != 0:
                r = pool_aux.index(1)
                picks = np.append(picks, pool_cl[r])
                delete = index.range_search(X_cl[[r]], epsilon**2)[2]
                for d in delete:
                    if pool_aux[d]==1:
                        pool_aux[d]=0
                        lenpool_cl -=1
            
    picks = np.sort(picks)
    return picks