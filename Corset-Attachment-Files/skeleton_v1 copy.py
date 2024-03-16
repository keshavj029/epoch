"""*****************************************************************************************
IIIT Delhi License
Copyright (c) 2023 Supratim Shit
*****************************************************************************************"""

from sklearn.datasets import fetch_kddcup99
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist
import wkpp as wkpp 
import numpy as np
import random

import math
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.utils import check_random_state
from sklearn.utils.extmath import stable_cumsum 


# Real data input
dataset = fetch_kddcup99()								# Fetch kddcup99 
data = dataset.data										# Load data
data = np.delete(data,[0,1,2,3],1) 						# Preprocess
data = data.astype(float)								# Preprocess
data = StandardScaler().fit_transform(data)				# Preprocess

n = np.size(data,0)										# Number of points in the dataset
d = np.size(data,1)										# Number of dimension/features in the dataset.
k = 17													# Number of clusters (say k = 17)
Sample_size = 100										# Desired coreset size (say m = 100)




def D2(data,k):											# D2-Sampling function.
    center = np.empty((n, d), dtype=data.dtype)	#initialized B 
     
	 
    w = np.ones(len(data))
    prob = 1/w
    prob = prob/sum(prob)
     
	#sampled first cluster center 
    sample=data[np.random.choice(range(n), size=1, p=prob),:]
    center[0] = sample
     
	# Initialize list of closest distances and calculate current potential
    closest_dist_sq = w*euclidean_distances(
        center[0, np.newaxis], data, squared=True
     )
    current_pot = closest_dist_sq.sum()
    random_state=check_random_state(None)

	# Pick the remaining n_clusters-1 points
    for c in range(1, k):
        # Choose center candidates by sampling with probability proportional
        # to the squared distance to the closest existing center
        rand_vals = random_state.uniform(size=100) * current_pot
        candidate_ids = np.searchsorted(stable_cumsum(closest_dist_sq), rand_vals)
        # XXX: numerical imprecision can result in a candidate_id out of range
        np.clip(candidate_ids, None, closest_dist_sq.size - 1, out=candidate_ids)

        # Compute distances to center candidates
        distance_to_candidates = w*euclidean_distances(
            data[candidate_ids], data, squared=True
        )

        # update closest distances squared and potential for each candidate
        np.minimum(closest_dist_sq, distance_to_candidates, out=distance_to_candidates)
        candidates_pot = distance_to_candidates.sum(axis=1)

        # Decide which candidate is the best
        best_candidate = np.argmin(candidates_pot)
        current_pot = candidates_pot[best_candidate]
        closest_dist_sq = distance_to_candidates[best_candidate]
        best_candidate = candidate_ids[best_candidate]

        # Permanently add best center candidate found in local tries
        

        center[c] = data[best_candidate]
        
    return center 										# Returns B from Algo-1.

centers = D2(data,k)									# Call D2-Sampling (D2())

def dist(x_,B_):
        d=[]
        for i,b_i in enumerate(centers):
            d.append(np.sqrt(np.sum((dt-b_i)*(dt-b_i))))
        min_d=min(d)
        return min_d

def Sampling(data,k,centers,Sample_size):				# Coreset construction function.
	alpha=16*(math.log(k)+2)
	
	B_i=[[] for b_i in centers]
	B_track=[0]*len(centers)
    for dt in data:
          d=[]
          for i,b_i in enumerate(centers):
                d.append(np.sqrt(np.sum((dt-b_i)*(dt-b_i))))
          min_d=min(d)
          min_d_idx=d.index(min_d)
          B_i[min_d_idx][B_track[i]]=dt
          B_track[i]+=1
          
    c_phi=0
    for dt in data:
          
          d=[]
          for i,b_i in enumerate(centers):
            d.append(np.sqrt(np.sum((dt-b_i)*(dt-b_i))))
          min_d=min(d)
          c_phi+=sum([min_d])
          
	c_phi*=1/n
    
	

	s=[]
	for dt in data:
        
        d=[]
		for i,b_i in enumerate(centers):
			d.append(np.sqrt(np.sum((dt-b_i)*(dt-b_i))))
		min_d=min(d)
		min_d_idx=d.index(min_d)
		s.append((alpha*dist(x,centers)/c_phi)+((2*alpha*sum([dist(x_i,centers) for x_i in B_i[min_d_idx]]))/(len(B-i[min_d_idx]*c_phi)))+((4*n)/len(B_i[min_d_idx])))
    

		"""s=[[] for b_i in centers]
		for i,b_i in enumerate(centers):
			for j,x in enumerate(B_i[i]):
				s[i][j]=(alpha*dist(x,centers)/c_phi)+((2*alpha*sum([dist(x_i,centers) for x_i in B_i[i]]))/(len(B-i[i]*c_phi)))+((4*n)/len(B_i[i]))"""
		
	total_s= sum(s)
		
		
	p=[]
	for i,dt in enumerate(data):
			p.append(s[i]/total_s)
    
	c=np.random.choice(data,size=Sample_size,p=p)
    




	return coreset, weight 								# Return coreset points and its weights.

coreset, weight = Sampling(data,k,centers,Sample_size)	# Call coreset construction algorithm (Sampling())

#---Running KMean Clustering---#
fkmeans = KMeans(n_clusters=k,init='k-means++')
fkmeans.fit_predict(data)

#----Practical Coresets performance----# 	
Coreset_centers, _ = wkpp.kmeans_plusplus_w(coreset, k, w=weight, n_local_trials=100)						# Run weighted kMeans++ on coreset points
wt_kmeansclus = KMeans(n_clusters=k, init=Coreset_centers, max_iter=10).fit(coreset,sample_weight = weight)	# Run weighted KMeans on the coreset, using the inital centers from the above line.
Coreset_centers = wt_kmeansclus.cluster_centers_															# Compute cluster centers
coreset_cost = np.sum(np.min(cdist(data,Coreset_centers)**2,axis=1))										# Compute clustering cost from the above centers
reative_error_practicalCoreset = abs(coreset_cost - fkmeans.inertia_)/fkmeans.inertia_						# Computing relative error from practical coreset, here fkmeans.inertia_ is the optimal cost on the complete data.

#-----Uniform Sampling based Coreset-----#
tmp = np.random.choice(range(n),size=Sample_size,replace=False)		
sample = data[tmp][:]																						# Uniform sampling
sweight = n*np.ones(Sample_size)/Sample_size 																# Maintain appropriate weight
sweight = sweight/np.sum(sweight)																			# Normalize weight to define a distribution

#-----Uniform Samling based Coreset performance-----# 	
wt_kmeansclus = KMeans(n_clusters=k, init='k-means++', max_iter=10).fit(sample,sample_weight = sweight)		# Run KMeans on the random coreset
Uniform_centers = wt_kmeansclus.cluster_centers_															# Compute cluster centers
uniform_cost = np.sum(np.min(cdist(data,Uniform_centers)**2,axis=1))										# Compute clustering cost from the above centers
reative_error_unifromCoreset = abs(uniform_cost - fkmeans.inertia_)/fkmeans.inertia_						# Computing relative error from random coreset, here fkmeans.inertia_ is the optimal cost on the full data.
	

print("Relative error from Practical Coreset is",reative_error_practicalCoreset)
print("Relative error from Uniformly random Coreset is",reative_error_unifromCoreset)"""