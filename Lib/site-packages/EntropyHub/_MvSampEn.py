""" Base Multivariate Sample Entropy function."""
import numpy as np       

def MvSampEn(Data, m=None, tau=None, r=0.2, Norm=False, Logx=np.exp(1)):
    """MvSampEn  estimates the multivariate sample entropy of a multivariate dataset.

    .. code-block:: python
    
        MSamp, B0, Bt, B1 = MvSampEn(Data) 
    
    Returns the multivariate sample entropy estimate (``MSamp``) and the
    average number of matched delay vectors (``m``: ``B0``; joint total 
    ``m+1`` subspace: ``Bt``; all possible ``m+1`` subspaces: ``B1``),
    from the M multivariate sequences in ``Data`` using the default parameters: 
    embedding dimension = 2*np.ones(M), time delay = np.ones(M), radius threshold = 0.2,
    logarithm = natural, data normalization = false
    
    .. attention::
 
        The entropy value returned as ``MSamp`` is estimated using the "full" 
        method [i.e.  -log(Bt/B0)] which compares delay vectors across all possible ``m+1`` 
        expansions of the embedding space as applied in [1][2]. Contrary to
        conventional definitions of sample entropy, this method does not provide a
        lower bound of 0!!
        Thus, it is possible to obtain negative entropy values for multivariate 
        sample entropy, even for stochastic processes...
      
        Alternatively, one can calculate ``MSamp`` via the "naive" method, 
        which ensures a lower bound of 0, by using the average number of matched
        vectors for an individual ``m+1`` subspace (B1) [e.g. -log(B1(1)/B0)],
        or the average for all ``m+1`` subspaces [i.e. -log(mean(B1)/B0)].

    .. note::
      
        To maximize the number of points in the embedding process, this algorithm 
        uses N-max(m*tau) delay vectors and **not** N-max(m)*max(tau) as employed 
        in [1][2].
    

 
    .. code-block:: python
    
        MSamp, B0, Bt, B1 = MvSampEn(Data, keyword = value, ...)
        
    Returns the multivariate sample entropy estimates (``MSamp``) estimated
    from the M multivariate data sequences in ``Data`` using the specified 
    keyword  arguments:
        :Data:  - Multivariate dataset, NxM matrix of N (>10) observations (rows) and M (cols) univariate data sequences 
        :m:     - Embedding Dimension, a vector of M positive integers
        :tau:   - Time Delay, a vector of M positive integers
        :r:     - Radius Distance threshold, a positive scalar
        :Norm:  - Normalisation of all M sequences to unit variance, a boolean
        :Logx:  - Logarithm base, a positive scalar 

            
    :See also:
        ``SampEn``, ``XSampEn``, ``SampEn2D``, ``MSEn``, ``MvFuzzEn``, ``MvPermEn``.

    :References:
       [1] Ahmed Mosabber Uddin, Danilo P. Mandic
            "Multivariate multiscale entropy: A tool for complexity
            analysis of multichannel data."
            Physical Review E 84.6 (2011): 061918.

       [2] Ahmed Mosabber Uddin, Danilo P. Mandic
            "Multivariate multiscale entropy analysis."
            IEEE signal processing letters 19.2 (2011): 91-94.

    """
   
    Data = np.squeeze(Data)
    assert Data.shape[0]>10 and Data.ndim==2 and Data.shape[1]>1,  "Data:   must be an NxM numpy matrix where N>10 and M>1"
    N, Dn = Data.shape 
    if m is None:    m = 2*np.ones(Dn, dtype=int)
    if tau is None:  tau = np.ones(Dn, dtype=int)
    m = m.astype(int)
    tau = tau.astype(int) 
  
    # and np.issubdtype(m.dtype, np.integer)
    # and np.issubdtype(tau.dtype, np.integer)
    assert isinstance(m,np.ndarray) and all(m>0) and m.size==Dn and m.ndim==1, "m:     must be numpy vector of M positive integers"
    assert isinstance(tau,np.ndarray) and all(tau>0) and tau.size==Dn and tau.ndim==1, "tau:   must be numpy vector of M positive integers"
    assert isinstance(r,(int,float)) and (r>=0), "r:     must be a positive value"
    assert isinstance(Logx,(int,float)) and (Logx>0), "Logx:     must be a positive value"
    assert isinstance(Norm,bool), "Norm:     must be a Boolean"    

    if Norm: Data = Data/np.std(Data,axis=0)
    
    Nx = N - max((m-1)*tau)
    Ny = N - max(m*tau)       
    Vex = np.zeros((Nx,sum(m)))
    q = 0
    for k in range(Dn):
        for p in range(m[k]):
            Vex[:,q] = Data[p*tau[k]:Nx+p*tau[k],  k]
            q += 1
            
    Count0 = Distx(Vex,r)
    B0 = np.sum(Count0)/(Nx*(Nx-1)/2)
            
    B1 = np.zeros(Dn)
    Vez = np.inf*np.ones((1,sum(m)+1));
    Temp = np.cumsum(m)
    for k in range(Dn):
        Sig = np.expand_dims(Data[m[k]*tau[k]:Ny+m[k]*tau[k], k],1)
        Vey = np.hstack((Vex[:Ny, :Temp[k]], Sig, Vex[:Ny, Temp[k]:]))
        Vez = np.vstack((Vez, Vey))
        Count1 = Distx(Vey, r)
        B1[k] = np.sum(Count1)/(Ny*(Ny-1)/2)
    Vez = Vez[1:,:]
    Count1 = Distx(Vez, r)
    Bt = np.sum(Count1)/(Dn*Ny*((Dn*Ny)-1)/2)
       
    with np.errstate(divide='ignore', invalid='ignore'):
        Samp = -np.log(Bt/B0)/np.log(Logx) 
   
    return Samp, B0, Bt, B1    


def Distx(Vex, r):
    Nt = Vex.shape[0]
    Counter = np.zeros((Nt-1,Nt-1),dtype=bool)
    for x in range(Nt-1):
         Counter[x,x:] = np.all(abs(Vex[x+1:,:] - Vex[x,:]) <= r, axis=1)
     
    return Counter


"""
    Copyright 2024 Matthew W. Flood, EntropyHub
    
    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at
    
     http://www.apache.org/licenses/LICENSE-2.0
    
    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
    
    For Terms of Use see https://github.com/MattWillFlood/EntropyHub
"""