"""Base Dispersion Entropy function."""
import numpy as np       
from scipy.special import ndtr
from scipy.cluster.vq import kmeans2

def DispEn(Sig, m=2, tau=1, c=3, Typex='NCDF', Logx=np.exp(1), Fluct=False, Norm=False, rho=1):
    """DispEn  estimates the dispersion entropy of a univariate data sequence.

    .. code-block:: python
    
        Dispx, Ppi = DispEn(Sig)
    
    Returns the dispersion entropy (``Dispx``) and the reverse dispersion entropy
    (``RDE``) estimated from the data sequence (``Sig``) using the default parameters:
    embedding dimension = 2, time delay = 1, symbols = 3, logarithm = natural,
    data transform = normalised cumulative density function (ncdf)
 
    .. code-block:: python
        
        Dispx, Ppi = DispEn(Sig, keyword = value, ...)
        
    Returns the dispersion entropy (``Dispx``) and the reverse dispersion entropy (``RDE``)
    estimated from the data sequence (``Sig``) using the specified 'keyword' arguments:
        :m:     - Embedding Dimension,  a positive integer
        :tau:   - Time Delay, a positive integer
        :c:     - Number of symbols, an integer > 1
        :Typex: - Typex of data-to-symbolic sequence transform, one of the following strings: {``'linear'``, ``'kmeans'``, ``'ncdf'``, ``'finesort'``, ``'equal'``}
            See the `EntropyHub guide <https://github.com/MattWillFlood/EntropyHub/blob/main/EntropyHub%20Guide.pdf>`_ for more info on these transforms.
                    
        :Logx:  - Logarithm base, a positive scalar
        :Fluct: - When ``Fluct == True``, DispEn returns the fluctuation-based Dispersion entropy.   [default: False]
        :Norm:  - Normalisation of ``Dispx`` and ``RDE`` values, a boolean:
            * ``False``  no normalisation - default
            * ``True``  normalises w.r.t # possible vector permutations 
              (``c^m``  or ``(2c -1)^m-1`` if ``Fluct == True``).

        :rho:   - *If ``Typex == 'finesort'``, rho is the tuning parameter (default: 1)
 
    :See also:
        ``PermEn``, ``SyDyEn``, ``MSEn``.
    
    :References:
        [1] Mostafa Rostaghi and Hamed Azami,
            "Dispersion entropy: A measure for time-series analysis." 
            IEEE Signal Processing Letters 
            23.5 (2016): 610-614.
    
        [2] Hamed Azami and Javier Escudero,
            "Amplitude-and fluctuation-based dispersion entropy." 
            Entropy 
            20.3 (2018): 210.
    
        [3] Li Yuxing, Xiang Gao and Long Wang,
            "Reverse dispersion entropy: A new complexity measure for 
            sensor signal." 
            Sensors 
            19.23 (2019): 5203.
    
        [4] Wenlong Fu, et al.,
            "Fault diagnosis for rolling bearings based on fine-sorted 
            dispersion entropy and SVM optimized with mutation SCA-PSO."
            Entropy
            21.4 (2019): 404.
    
    """

    Sig = np.squeeze(Sig)
    N = Sig.shape[0]       
        
    assert N>10 and Sig.ndim == 1,  "Sig:   must be a numpy vector"
    assert isinstance(c,int) and (c > 1), "c:     must be an integer > 1"
    assert isinstance(m,int) and (m > 0), "m:     must be an integer > 0"
    assert isinstance(tau,int) and (tau > 0), "tau:   must be an integer > 0"
    assert isinstance(Logx,(int,float)) and Logx>0, "Logx:     must be a positive value"
    assert isinstance(Fluct,bool), "Fluct:    must be boolean - True or False"
    assert Typex.lower() in ['linear','kmeans','ncdf','finesort','equal'], \
            "Typex:    must be one of the following strings - \
            'linear', 'kmeans', 'ncdf', 'finesort', 'equal'"     
    assert isinstance(Norm,bool), "Norm:     must be boolean - True or False"  
            
    if Typex.lower() == 'linear':
        Zi = np.digitize(Sig,np.arange(min(Sig),max(Sig),np.ptp(Sig)/c))
        
    elif Typex.lower() == 'kmeans':        
        Clux, Zx = kmeans2(Sig, c, iter=200)
        Zx += 1;  xx = np.argsort(Clux) + 1;     Zi = np.zeros(N);
        for k in range(1,c+1):
            Zi[Zx==xx[k-1]] = k;
        
    elif Typex.lower() == 'ncdf':  
        Zx = ndtr((Sig-np.mean(Sig))/np.std(Sig))
        Zi = np.digitize(Zx,np.arange(0,1,1/c))  
        
        del Zx
        
    elif Typex.lower() == 'finesort':        
        assert isinstance(rho,(float,int)) and rho>0 \
        and Typex.lower()=='finesort', "rho:  must be >0 and Typex must be 'finesort'"
        
        Zx = ndtr((Sig-np.mean(Sig))/np.std(Sig))
        Zi = np.digitize(Zx,np.arange(0,1,1/c))        
        Ym = np.zeros((N-(m-1)*tau, m))
        for n in range(m):
            Ym[:,n] = Zx[n*tau:N-((m-n-1)*tau)]
        
        Yi = np.floor(np.max(abs(np.diff(Ym)),axis=1)/(rho*np.std(abs(np.diff(Sig)))))         
        del Zx, Ym, rho
                
    elif Typex.lower() == 'equal':
        ix = np.argsort(Sig, kind='mergesort')
        xx = np.round(np.arange(0,2*N,N/c)).astype(int)
        Zi = np.zeros(N)
        for k in range(c):
            Zi[ix[xx[k]:xx[k+1]]] = k+1 
        del ix, xx, k        
            
    Zm = np.zeros((N-((m-1)*tau),m))   
    for n in np.arange(m):
        Zm[:,n] = Zi[n*tau:N-((m-n-1)*tau)]
    
    if Typex.lower() == 'finesort':
        Yi = np.expand_dims(Yi,axis=1)        
        Zm = np.hstack((Zm, Yi))        
        
    if Fluct:
        Zm = np.diff(Zm,axis=1)
        if m < 2:
           print('Warning:  Fluctuation-based Dispersion Entropy',
               'is undefined for m = 1.\n',
               'An embedding dimension (m) > 1 should be used.')       
           
    T  = np.unique(Zm,axis=0)
    Nx = T.shape[0]
    Counter = np.zeros(Nx)
       
    for n in range(Nx):
        Counter[n] = sum(np.any(Zm - T[n,:],axis=1)==False)

    Ppi = Counter[Counter!= 0]/Zm.shape[0]    
    if Fluct:
        RDE = sum((Ppi - (1/((2*c - 1)**(m-1))))**2);
    else:
        RDE = sum((Ppi - (1/(c**m)))**2);
        
    #RDE = sum(Ppi**2) - (1/Nx)
    Dispx = -sum(Ppi*np.log(Ppi)/np.log(Logx))    
    
    if round(sum(Ppi)) != 1:
        print('Potential error calculating probabilities')    
        
    if Norm:
        #Dispx = Dispx/(np.log(Nx)/np.log(Logx))
        #RDE = RDE/(1 - (1/(Nx)))                
        if Fluct:
            Dispx = Dispx/(np.log((2*c - 1)**(m-1))/np.log(Logx))
            RDE = RDE/(1 - (1/((2*c - 1)**(m-1))))
        else:
            Dispx = Dispx/(np.log(c**m)/np.log(Logx))
            RDE = RDE/(1 - (1/(c**m)))
                
    return Dispx, RDE
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