import numpy as np

#TaskA / Matrix Standarization
def standardize_matrix(X):
    mean = np.mean(X, axis=0)  #on calcule par colonne parce qu on veut les moyennes et les variances des caracteristiques
    
    std = np.std(X, axis=0)

    std[std == 0] = 1  #selectionne la ou les valeurs sont egales a zero 
    
    return (X - mean) / std



#TaskB / Pairwise Distances in the Plane 

def PairwiseDistances(P,Q): 
    if P.shape[1]!=2 or Q.shape!=2 :
        raise ValueError("P and Q should have number f columns equal to 2 ")

    p , q = P.shape[0], Q.shape[0]

    T = np.zeros((p, q))  
    for i in range (p) : 
        for j in range (q) : 
            T[i,j] = np.linalg.norm(P[i] - Q[j])   #il veut la norme euclidienne
    return T  


#TaskC / Likelihood of a Data Sample


# PDF gaussienne 1D
def gaussian_pdf_1d(x, mu, sigma):
    return (1 / np.sqrt(2*np.pi*sigma)) * np.exp(-(x-mu)**2 / (2*sigma))

def LikelihoodAss(X, mu1, sigma1, mu2, sigma2) : 
    assignments = []
    for x in X : 
        p1 = gaussian_pdf(x, mu1, Sigma1)
        p2 = gaussian_pdf(x, mu2, Sigma2)
        if p1 > p2 :
            assignments.append(1) 
        else :
            assignments.append(2)
    return assignments 
        
