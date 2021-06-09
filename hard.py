import numpy as np
import matplotlib.pyplot as plt
import copy
np.random.seed(42)

def sum_until(Z, order, mui):
    soma = 0
    i = 0
    gap = np.inf
    indexes = []
    while (i<len(order)) and (abs(gap - Z[order[i]]) <= gap):
        soma += Z[order[i]]
        indexes.append(order[i])
        i+=1
        gap = mui - soma
    return soma, indexes, gap

def pert_ident(x):
    """Computa o grau de pertencimento baseado na função identidade"""
    return x/np.sum(x)

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def pert_quad(x):
    """Computa o grau de pertencimento baseado na função quadrática"""
    return np.array(x)**2/np.sum(np.array(x)**2)

def kmeans_capacity_constraints(X,Z,n_clusters,pertencimento=pert_quad,CENTROIDS = None, mu=None):

    n_samples, dim = X.shape

    #Inicialização
    k = 0
    k_max = 100
    count_fails = 0
    desempenho_old = np.inf
    colors = ['b','g','r','y']

    labels = np.random.randint(0,n_clusters,size=n_samples)
    LABELS = [np.copy(labels)]
    ORDER = [np.array([0,1,2])]
    
    while (k<=k_max) and (count_fails<=50):
        SOMA = []
        INDEXES = []
        GAPS = []
        used = []
        indexes = []

        dist_centr = np.array([np.linalg.norm(X-CENTROIDS[i], ord=2, axis=1) for i in range(n_clusters)])
        grau_pert = np.array([pertencimento(dist_centr[:,i]) for i in range(n_samples)])

        if k==0:
            ordem_centroids = np.random.choice(np.arange(0,n_clusters),n_clusters,replace=False)

        for i in ordem_centroids:
            order = np.argsort(-grau_pert[:,i]) #ordenação por distância do ponto ao centróide
            order = np.array([x for x in order if x not in used]) #retirada dos pontos já alocados nos outros clusters
            soma, indexes, gap = sum_until(Z, order, mu[i]) #efetuação do método de seleção
            used += indexes #atualização da lista de pontos utilizados
            
            #Atualizações dos outputs das seleções
            SOMA.append(soma)
            INDEXES.append(indexes)
            GAPS.append(gap)
            CENTROIDS[i] = np.mean(X[indexes,:],axis=0)

        for i in range(n_clusters):
            labels[INDEXES[i]] = ordem_centroids[i]
        
        k+=1
                
        INDEXES = []
        SUM_DIST = []
        for i in range(n_clusters):
            INDEXES.append([x for x in range(n_samples) if labels[x] == i])
            centroid = np.copy(CENTROIDS[i])
            SUM_DIST.append(np.sum(np.linalg.norm(X[INDEXES[i]]-centroid, ord=2, axis=1)))
        desempenho = np.sum(SUM_DIST)
        #print(desempenho)

        if (desempenho < desempenho_old) or (k<=1):
            desempenho_old = desempenho
            CENTROIDS_BEST = np.copy(CENTROIDS)
            INDEXES_BEST = INDEXES
            fail = False
            count_fails = 0
        else:
            CENTROIDS = np.copy(CENTROIDS_BEST)
            INDEXES = INDEXES_BEST
            fail = True
            count_fails +=1

        if fail:
            #print('Iteração falhou')
            ordem_centroids = np.random.choice(np.arange(0,n_clusters),n_clusters,replace=False)
        else:
            ordem_centroids = np.argsort(-np.array(SUM_DIST))
            LABELS.append(np.copy(labels))
            ORDER.append(np.copy(ordem_centroids))
    return LABELS, ORDER

def kmeans_modified(X,Z,clusters,CENTROIDS=None,mu=None):

    n_samples, dim = X.shape
    
    #Inicialização
    k = 0
    k_max = 50
    count_fails = 0
    desempenho_old = np.inf
    colors = ['b','g','r','y']

    labels = np.random.randint(0,clusters,size=n_samples)
    LABELS = [np.copy(labels)]
    ORDER = [np.array([0,1,2])]


    while (k<=k_max) and (count_fails<=3):

        SOMA = []
        INDEXES = []
        GAPS = []
        used = []
        indexes = []

        dist_centr = np.array([np.linalg.norm(X-CENTROIDS[i], ord=2, axis=1) for i in range(clusters)])

        if k==0:
            ordem_centroids = np.random.choice(np.arange(0,clusters),clusters,replace=False)

        ordem_centroids = range(clusters)
        for i in ordem_centroids:
            order = np.argsort(dist_centr[i,:]) #ordenação por distância do ponto ao centróide
            order = np.array([x for x in order if x not in used]) #retirada dos pontos já alocados nos outros clusters
            soma, indexes, gap = sum_until(Z, order, mu[i]) #efetuação do método de seleção
            used += indexes #atualização da lista de pontos utilizados
            #Atualizações dos outputs das seleções
            SOMA.append(soma)
            INDEXES.append(indexes)
            GAPS.append(gap)
            CENTROIDS[i] = np.mean(X[indexes,:],axis=0)

        for i in range(clusters):
            labels[INDEXES[i]] = ordem_centroids[i]

        k+=1

        INDEXES = []
        SUM_DIST = []
        for i in range(clusters):
            INDEXES.append([x for x in range(n_samples) if labels[x] == i])
            centroid = np.copy(CENTROIDS[i])
            SUM_DIST.append(np.sum(np.linalg.norm(X[INDEXES[i]]-centroid, ord=2, axis=1)))
        desempenho = np.sum(SUM_DIST)

        if (desempenho < desempenho_old) or (k<=1):
            desempenho_old = desempenho
            CENTROIDS_BEST = np.copy(CENTROIDS)
            INDEXES_BEST = INDEXES
            fail = False
            count_fails = 0
        else:
            CENTROIDS = np.copy(CENTROIDS_BEST)
            INDEXES = INDEXES_BEST
            fail = True
            count_fails +=1

        if fail:
            ordem_centroids = np.random.choice(np.arange(0,clusters),clusters,replace=False)
        else:
            ordem_centroids = np.argsort(-np.array(SUM_DIST))
            LABELS.append(np.copy(labels))
            ORDER.append(np.copy(ordem_centroids))
    return LABELS, ORDER

from mip import Model, xsum, minimize, BINARY, OptimizationStatus

def kmeans_constraints_exato(X,Z,clusters,CENTROIDS=None,mu=None):
    CENTROIDS = np.copy(CENTROIDS)
    if X.shape[0]/clusters == np.mean(mu):
        epsilon = 0
    else:
        epsilon = 30
    n_samples = X.shape[0] # número de pontos
    dim = X.shape[1] #dimensão dos pontos
    n_clusters = clusters
    
    labels = np.random.randint(0,n_clusters,size=n_samples)
    LABELS = [np.copy(labels)]

    k = 0
    kmax = 100 #15
    u_old = 0
    u_new = 1

    while (k<=kmax) and (np.mean(u_old==u_new)!=1):
        #print(k)
        u_old = np.copy(u_new)
        k+=1
        dist_centr = np.array([np.linalg.norm(X-CENTROIDS[i], ord=2, axis=1) for i in range(n_clusters)])**2
        g, n = dist_centr.shape

        #Minimização em u
        m = Model()

        u = [[m.add_var('u({} ,{} )'.format(i, j), var_type=BINARY) for j in range(n)] for i in range(g)]

        for j in range(n):
            m += xsum(u[i][j] for i in range(g)) == 1 #, 'row({} )'.format(j)

        for i in range(g):
            m += xsum(u[i][j]*Z[j] for j in range(n)) >= mu[i]-epsilon #, 'row({} )'.format(i)

        m.objective = minimize(xsum(u[i][j]*dist_centr[i][j] for i in range(g) for j in range(n)))

        # optimizing
        m.optimize(max_seconds=30)
        u_new = []
        for v in m.vars:
            u_new.append(v.x)
        u_new = np.array(u_new).reshape((g,n))
        
        #Atualização dos centróides
        for i in range(g):
            CENTROIDS[i,:] = X[u_new[i,:] ==1].mean(axis=0)
        #print(u_new)
        LABELS.append(np.array(np.argmax(u_new, axis = 0).tolist()))
    return LABELS