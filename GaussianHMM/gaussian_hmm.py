
"""
Created on Thu May 27 17:02:23 2021

@author: Nathan Liittschwager
"""

import numpy as np
from scipy.stats import multivariate_normal
from sklearn.mixture import GaussianMixture
from hmmlearn import hmm


class GaussianHiddenMarkov:
    """
        Parameters
        ----------
        n_states : int, optional
            The number of latent states assumed by the model. This is a 
            discrete set of states that each hidden variable Z[t] can emit.
        lam : float, optional
            A regularizer for the covariance matrix. This is primarily to 
            prevent any issues of singularity or positive definiteness,
            but increasing this parameter can also reduce the role that the
            covariance plays in the algorithm.
        max_iter : TYPE, optional
            Maximum number of EM steps to attempt when learning parameters
        tol : float, optional
            A tolerance value. If the change in the log-likelihood betweeen
            every step of EM is < tol, then EM will terminate and the algorithm
            will be considered converged

        Attributes
        -------
        K : int
            The number of latent states
        state_set : [int]
            A set of the possible hidden states
        
        lam : float
            Regularizer for the covariance
        tol : float
            Tolerance value for EM algorithm
        A : n-by-n array-like
            Represents the transition probability matrix. The entry at index
            [i,j] is the probability of transitioning from state i to state j
            in a time step.
            I.e., A[i,j] == Pr[ Z[t]=j | Z[t-1]=i ]
        B : dict of tuple of array-like
            The observation model. Indexers are the state classes, 0,1,...,K-1.
            B[k] = (mu, cov) where mu = mean vector of observation for state k
            and cov = m-by-m covariance matrix for state k.
        """
    def __init__(self, n_states=3, lam=0.001, max_iter=100, tol=0.0001):
        
        self.K = n_states
        self.state_set = [k for k in range(n_states)]
        self.max_iter=max_iter
        self.lam = lam
        self.tol = tol
        
        self.A = np.random.uniform(size=(n_states,n_states))
        self.normalize_prob_mat()
        self.B = dict()
        self.pi = np.ones(shape=(n_states, 1)) / n_states
        
    def initialize(self, X):
        self.X_dim = X.ndim
        self.T = len(X)
        
        
        if self.X_dim <= 1:
            raise Exception("Time series should be 2D but got 1D")
        # if self.U_dim <= 1:
        #     raise Exception("Expected 2D input array, but got 1D")
            
        # initialization trick using a Gaussian Mixture Model
        gmm = GaussianMixture(n_components=self.K).fit(X)
        for k in range(self.K):
            #idxs = np.random.choice(range(self.T), size=2, replace=False)
            #x1 = X[idxs[0],:].reshape(-1, 1)
            #x2 = X[idxs[1],:].reshape(-1, 1)
            #mu = (x1 + x2) / 2
            #cov = (x1 - mu).T.dot(x1 - mu) + (x2 - mu).T.dot(x2 - mu)
            #cov = cov / 2
            #cov = np.ones(shape=(mu.shape[0], mu.shape[0])) * 1000
            mu = gmm.means_[k,:]
            cov = gmm.covariances_[k,:,:]
            self.B[k] = (mu, cov)

        #print(self.B)
        self.evidence = np.zeros(shape=(self.T, self.K))
        self.alpha = np.zeros(shape=(self.T, self.K))
        self.beta = np.zeros(shape=(self.T, self.K))
        self.marginals = np.zeros(shape=(self.T, self.K))
    
    def normalize_prob_mat(self, prob_mat=None):
        """
        Parameters
        ----------
        prob_mat : array-like, optional
            Transition Probability matrix. If None, then normalizes the model's
            transition probability matrix A. Used for normalizing accumulated
            counts.

        Returns
        -------
        prob_mat : array-like
            Normalized transition probability matrix. The columns sum to 1.
        """
        if prob_mat is None:
            norm = self.A.sum(axis=1)
            self.A = self.A / norm[:, np.newaxis]
        else:
            norm = prob_mat.sum(axis=1)
            return prob_mat / norm[:, np.newaxis]
        
    def normalize_vec(self, vec):
        """

        Parameters
        ----------
        vec : array-like
            A vector of counts to be normalized to probabilities.

        Returns
        -------
        vec : array-like
            Normalized vector of counts. I.e., a vector of probabilities
        Z : float
            The normalization constant.

        """
        Z = vec.sum()
        return vec / Z, Z
    
    def emission_prob(self, x, k, B):
        """

        Parameters
        ----------
        x : array-like
            A vector of 'continuous' observations, assumed to be generated
            from the normal distribution. I.e., x ~ N(mu, sigma)
        k : int
            The state indexer
        B : dict of tuple of array-like
            The emission model. Should have the form B[k] = (mu_k, cov_k),
            where mu_k is a m-by-1 vector of means, and cov_k is a m-by-m
            covariance matrix

        Returns
        -------
        prob : float
            The emission probability vector of the observations.
            I.e., Pr(x | mu_k, cov_k) == N(x | mu_k, cov_k).

        """
        mu, cov = B[k]
        prob = multivariate_normal.pdf(x, mean=mu, cov=cov)
        return prob
        
    def compute_evidence(self, X, B):
        """

        Parameters
        ----------
        X : T-by-M array like
            The time series of interest. Rows indicate time steps 1,...,T and
            columns are the M observations made at each time step.
        B : dict of tuple of array-likes. 
            See class attributes.

        Returns
        -------
        evidence : array-like, float
            T-by-K vector of emission probabilties. I.e., the probability
            of observing each observation vector X[t] at time t, given state k.
            evidence[t,k] == Pr[X[t] | Z[t] = k] == N(X[t] | mu_k, cov_k)

        """
        evidence = np.zeros(shape=(X.shape[0], self.K))
        for k in range(self.K):
            evidence[:, k] = self.emission_prob(X, k, B)
        evidence = self.normalize_prob_mat(evidence)
        return evidence
    
    def forward(self, A, evidence, pi):
        T = evidence.shape[0]
        alpha = np.zeros(shape=evidence.shape)
        Zt = np.zeros(shape=(T, 1))
        #print(evidence[0,:].reshape(-1, 1).shape)
        #print(pi.shape)
        temp = np.multiply(evidence[0,:].reshape(-1, 1), pi)
        alpha[0], Zt[0] = self.normalize_vec(temp[:,0])
        
        for t in range(1, T):
            a = alpha[t-1,:].reshape(-1,1)
            temp = A.T.dot(a)
            temp = np.multiply(evidence[t,:].reshape(-1,1), temp)
            alpha[t], Zt[t] = self.normalize_vec(temp.reshape(-1))
            
        return alpha, np.sum(np.log(Zt))
        
        
    def backward(self, A, evidence):
        T = evidence.shape[0]
        beta = np.zeros(shape=evidence.shape)
        beta[T-1] = 1
        Zt = np.ones(shape=(T, 1))
        
        for t in range(1, T):
            t0 = T - t
            temp = np.multiply(evidence[t0], beta[t0]).reshape(-1, 1)
            temp = np.dot(A, temp).reshape(-1)
            beta[t0-1], Zt[t0-1] = self.normalize_vec(temp)
            
        
        return beta, np.sum(np.log(Zt))
    
    def predict_logprobs(self, X):
        """

        Parameters
        ----------
        X : T-by-M array-like
            Input data. Must be of the same shape column-wise as the training data

        Returns
        -------
        logprobs : array-like, float
            The log probabilities of each time step
        log-like : float
            The final log likelihood of the sequence

        """
        evidence = self.compute_evidence(X, self.B)
        forward, log_like = self.forward(self.A, evidence, self.pi)
        logprobs = np.log(forward[:,:]).sum(axis=1)
        return logprobs, log_like
            
    def viterbi(self, X, A, evidence, pi):
        delta = np.zeros(shape=(len(X),self.K))
        ptrs = np.zeros(shape=(len(X),self.K))
        
        delta[0,:] = np.log(np.multiply(evidence[0,:], pi.reshape(-1)) + 0.001)
        ptrs[0,:] = 0
        
        for t in range(1, self.T):
            for j in range(self.K):
                temp = delta[t-1,:] + np.log(A[:,j]) + np.log(evidence[t,j])
                delta[t,j] = np.max(temp)
                ptrs[t,j] = np.argmax(temp)
                
        bestptr = int(np.argmax(delta[-1,:]))
        latent_path = np.zeros(len(X)).astype(int)
        latent_path[-1] = bestptr
        for t in range(1,self.T):
            t0 = self.T - t
            bestptr = int(ptrs[t0,bestptr])
            latent_path[t0-1] = bestptr
        return latent_path
            
    def compute_marginals(self, alpha, beta):
        
        marginals = np.multiply(alpha, beta)
        marginals = self.normalize_prob_mat(marginals)
        return marginals
        
    def EM_step(self, X, A, B, alpha, beta, lam):
        
        eta = np.zeros(shape=(self.T, self.K, self.K))
        
        # E step
        marginals = self.compute_marginals(alpha, beta)
        evidence = self.compute_evidence(X, B)
            
        for t in range(self.T-1):
            for i in range(self.K):
                for j in range(self.K):
                    eta[t, i, j] = alpha[t,i] * A[i,j] * beta[t+1,j] * evidence[t+1,j]

            eta[t,:,:] /= eta[t,:,:].sum()


            
        A_hat = np.zeros(shape=A.shape)
        B_hat = dict()
        pi_hat = np.zeros(shape=self.pi.shape)
        for i in range(self.K):
            for j in range(self.K):
                A_hat[i,j] = eta[:,i,j].sum()# / eta[:,i,:].sum()
                A_hat[i,j] /= marginals[:,i].sum()
        A_hat = self.normalize_prob_mat(A_hat)
        for k in range(self.K):
            N_k = marginals[:,k].sum()

            mu_k = X.T.dot(marginals[:,k].reshape(-1, 1)) / N_k
            #print(mu_k)
            xx_k = np.zeros(shape=(X.shape[1], X.shape[1]))
            for t in range(self.T):
                x = X[t,:].reshape(-1, 1)
                xx_k += marginals[t,k] * x.T.dot(x)
            cov_k = (xx_k - (N_k * mu_k.T.dot(mu_k))) / N_k
            cov_k[np.diag_indices_from(cov_k)] += lam 

            B_hat[k] = (mu_k, cov_k)
            
            pi_hat[k] = marginals[0,k] / marginals[0,:].sum()
            
        return A_hat, B_hat, pi_hat


    def EM_update(self, X, A, B, pi):
        self.evidence = self.compute_evidence(X, B)
        alpha, logprob = self.forward(A, self.evidence, pi)
        #print(alpha)
        beta, _ = self.backward(A, self.evidence)
        
        A_hat, B_hat, pi_hat = self.EM_step(X, A, B, alpha, beta, self.lam)
        

        return A_hat, B_hat, pi_hat, alpha, beta, logprob
    
        
            
    
        
        
    def logp(self, X):
        evidence = self.compute_evidence(X, self.B)
        logp, _ = self.forward(self.A, evidence, self.pi)
        return logp
        
    def fit(self, X, inputs=None):
        
        self.initialize(X)
        #print(self.A)
        old_logp = -np.inf
        for i in range(self.max_iter):
            #print(self.A)
            A_hat, B_hat, pi_hat, alpha, beta, logp = self.EM_update(X, self.A, self.B, self.pi)
            print("round : {}, logp: {}".format(i+1, logp))
            
            
            self.A = A_hat
            self.B = B_hat
            #self.pi = pi_hat
            self.alpha = alpha
            self.beta = beta
            self.logp = logp
            self.converged = abs(old_logp - logp) < self.tol
            if self.converged:
                break
            else:
                old_logp = logp
                continue
                
             
