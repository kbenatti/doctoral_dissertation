import numpy as np
import matplotlib.pyplot as plt
import copy
import cvxpy as cp

def distancias(X,P):
    n_samples = X.shape[0]
    D = []
    clusters = len(P)
    for i in range(clusters):
        d = []
        for j in range(n_samples):
            d.append(np.linalg.norm(X[j,:] - P[i], 2)**2)
        D.append(d)
    return D

def sistem(D,Z):
    clusters, n_samples = np.array(D).shape
    A = np.zeros((clusters,clusters))
    b = np.zeros((clusters,1))

    for k in range(clusters):
        for l in range(clusters):
            aux1 = 0
            for j in range(n_samples):
                aux2 = 0
                for i in range(clusters):
                    aux2 = aux2 + D[k][j]/D[i][j]
                aux2 = 2*aux2
                aux1 = aux1 - ((Z[j]**2)/D[l][j])/aux2 
            if k==l:
                for j in range(n_samples):
                    aux1 = aux1 + ((Z[j]**2)/(2*D[k][j]))
            A[k,l] = aux1

        aux1 = np.sum(Z)/clusters
        for j in range(n_samples):
            aux2 = 0
            for i in range(clusters):
                aux2 = aux2 + D[k][j]/D[i][j]
            aux1 = aux1 - Z[j]/aux2
        b[k,0] = aux1
    return A, b

def beta_alpha(A,b,D,Z):
    
    clusters, n_samples = np.array(D).shape
    
    #Resolução do sistema em beta
    beta = np.zeros((clusters,1))
    beta[0,0] = 0
    beta[1:] = np.linalg.solve(np.dot(A[:,1:].transpose(),A[:,1:]),np.dot(A[:,1:].transpose(),b))

    #Determinação de alpha
    alpha = np.zeros((n_samples,1))
    for j in range(n_samples):
        aux1 = 0
        aux2 = 0
        for i in range(clusters):
            aux1 = aux1 - beta[i,0]/D[i][j]
            aux2 = aux2 + 1/D[i][j]
        aux1 = 2 + aux1*Z[j]
        alpha[j,0] = aux1/aux2
    return beta, alpha

def def_U(alpha, beta, Z, D):
    clusters, n_samples = np.array(D).shape
    U = []
    for i in range(clusters):
        u = []
        for j in range(n_samples):
            u.append((alpha[j,0]+Z[j]*beta[i,0])/(2*D[i][j]))
        U.append(u)
    return U

def centroides(X, U, m):
    clusters, n_samples = np.array(U).shape
    dim = X.shape[1]
    P = np.zeros((clusters,dim))
    for i in range(clusters):
        aux1 = 0
        aux2 = 0
        for j in range(n_samples):
            aux1 = aux1 + U[i][j]**m * X[j]
            aux2 = aux2 + U[i][j]**m
        P[i] = aux1/aux2
    return P

def fuzzy_capacity_constraints(X,Z,clusters,salvaguardas=0,P=None):
    n_samples, dim = X.shape
    
    #Escolha m>1 (tipicamente m=2)
    m=2
    
    #Critério de parada
    num_iter = 50
    tolerance = 10e-8
    
    #Inicialização de parâmetros
    iteracao = 0
    error = np.inf
    U = []
    
    if not isinstance(P,(np.ndarray,list)):
        #Inicialize p_i aleatóriamente, 1<=i<=c
        P = []
        for i in range(clusters):
            P.append(np.random.rand(1,dim))
        
    LABELS = []

    while (error>tolerance) and (iteracao < num_iter):
        
        iteracao = iteracao + 1
        
        #Definição d_{ij}
        D = distancias(X,P)

        #Definição do sistema linear (para determinação de beta)
        A, b = sistem(D,Z)
        
        #Definição de alpha e beta
        beta, alpha = beta_alpha(A,b,D,Z)

        U_old = copy.deepcopy(U[:])
        U = def_U(alpha, beta, Z, D)
        
        if salvaguardas and (np.array(U).min()<0):
            outside = [[(i,j) for j in range(n_samples) if U[i][j]<0] for i in range(clusters)]
            for i in range(clusters):
                for tuples in outside[i]:
                    D[tuples[0]][tuples[1]] = 10**20#100*D[tuples[0]][tuples[1]]#np.inf
            A, b = sistem(D,Z)
            beta, alpha = beta_alpha(A,b,D,Z)
            U_old = copy.deepcopy(U[:])
            U = def_U(alpha, beta, Z, D)

        P = centroides(X, U, m)

        if iteracao>1:
            error = np.linalg.norm(np.array(U) - np.array(U_old))
        LABELS.append(np.argmax(np.array(U), axis = 0).tolist())
    
    return LABELS, np.array(U)

def fuzzy_constraints_exato(X, Z, clusters, P=None):
    n_samples, dim = X.shape

    #Escolha m>1 (tipicamente m=2)
    m=2

    #Critério de parada
    num_iter = 50
    tolerance = 10e-8

    #Inicialização de parâmetros
    iteracao = 0
    error = np.inf
    U = []

    if not isinstance(P,(np.ndarray,list)):
        #Inicialize p_i aleatóriamente, 1<=i<=c
        P = []
        for i in range(clusters):
            P.append(np.random.rand(1,dim))

    LABELS = []

    while (error>tolerance) and (iteracao < num_iter):

        iteracao = iteracao + 1

        #Definição d_{ij}
        D = np.array(distancias(X,P))
        g, n = D.shape
        D = np.diag(D.reshape(g*n))

        A1 = np.array([[Z if i==t else np.zeros(n,) for i in range(g)] for t in range(g)]).reshape(g,n*g)

        A2 = np.zeros((n,g*n))
        for j in range(n):
            for i in range(g):
                A2[j,n*i+j] = 1

        A = np.concatenate((A1,A2), axis=0)
        del A1, A2

        mu = Z.sum()/clusters*np.ones((clusters,))
        b = np.concatenate((mu,np.ones(n)))

        G = np.concatenate((-np.eye(n*g),np.eye(n*g)))
        h = np.concatenate((np.zeros((n*g,)),np.ones((n*g,))))

        u = cp.Variable(n*g)
        prob = cp.Problem(cp.Minimize((1/2)*cp.quad_form(u, D)),
                         [G @ u <= h,
                          A @ u == b])
        prob.solve()

        U_old = copy.deepcopy(U)
        U = u.value.reshape(g,n)

        P = centroides(X, U, m)

        if iteracao>1:
            error = np.linalg.norm(np.array(U) - np.array(U_old))
        LABELS.append(np.argmax(np.array(U), axis = 0).tolist())

    return LABELS, np.array(U)
