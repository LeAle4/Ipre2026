""" Base Distribution Entropy function."""
import numpy as np
from scipy.stats import skew

def DistEn(Sig,  m=2, tau=1, Bins='Sturges', Logx=2, Norm=True):
    """DistEn  estimates the distribution entropy of a univariate data sequence.

    .. code-block:: python
        
        Dist, Ppi = DistEn(Sig) 
        
    Returns the distribution entropy estimate (``Dist``) and the
    corresponding distribution probabilities (``Ppi``) estimated from the data 
    sequence (``Sig``)  using the default  parameters: 
    embedding dimension = 2, time delay = 1, binning method = 'Sturges',
    logarithm = base 2, normalisation = w.r.t # of histogram bins
 
    .. code-block:: python
    
        Dist, Ppi = DistEn(Sig, keyword = value, ...)
        
    Returns the distribution entropy estimate (``Dist``) estimated from the data
    sequence (``Sig``) using the specified 'keyword' arguments:
        :m:     - Embedding Dimension, a positive integer
        :tau:   - Time Delay, a positive integer
        :Bins:  - Histogram bin selection method for distance distribution, one of the following:
            
                  * an integer > 1 indicating the number of bins, 
                  * or one of the following strings {``'sturges'``, ``'sqrt'``, ``'rice'``, ``'doanes'``} [default: ``'sturges'``]
              
        :Logx:  - Logarithm base, a positive scalar (enter 0 for natural log) 
        :Norm:  - Normalisation of ``Dist`` value, a boolean:
            * [False]  no normalisation.
            * [True]   normalises w.r.t # of histogram bins - default
 
    :See also:
        ``XDistEn``, ``DistEn2D``, ``MSEn``, ``K2En``
  
    :References:
        [1] Li, Peng, et al.,
            "Assessing the complexity of short-term heartbeat interval 
            series by distribution entropy." 
            Medical & biological engineering & computing 
            53.1 (2015): 77-87. 
    
    """
    
    Sig = np.squeeze(Sig)
    if Logx == 0:
         Logx = np.exp(1)
    
    assert Sig.shape[0]>10 and Sig.ndim == 1,  "Sig:   must be a numpy vector"
    assert isinstance(m,int) and (m > 0), "m:     must be an integer > 0"
    assert isinstance(tau,int) and (tau > 0), "tau:   must be an integer > 0"
    assert isinstance(Logx,(int,float)) and Logx > 0, "Logx:     must be a positive value"
    assert isinstance(Norm,bool), "Norm:     must be boolean - True or False"    
    assert (isinstance(Bins,int) and Bins>1) \
            or (Bins.lower() in ['sturges','sqrt','rice','doanes']), \
            "Bins:    an integer > 1, or can be one of the following strings - \
            'sturges', 'sqrt', 'rice', 'doanes'"    
    
    Nx = Sig.shape[0] - ((m-1)*tau)   
    Zm = np.zeros((Nx,m))
    for n in np.arange(m):
        Zm[:,n] = Sig[n*tau:Nx+(n*tau)]
        
    DistMat = np.zeros(round(Nx*(Nx-1)/2))
    for k in range(1,Nx):
        Ix = (int((k-1)*(Nx - k/2)), int(k*(Nx-((k+1)/2))))
        DistMat[Ix[0]:Ix[1]] = np.max(abs(np.tile(Zm[k-1,:],(Nx-k,1)) - Zm[k:,:]),axis=1)
        
    Ny = len(DistMat)
    if isinstance(Bins, str):
        if Bins.lower() == 'sturges':
            Bx = np.ceil(np.log2(Ny) + 1)
        elif Bins.lower() == 'rice':
            Bx = np.ceil(2*(Ny**(1/3)))
        elif Bins.lower() == 'sqrt':
            Bx = np.ceil(np.sqrt(Ny))
        elif Bins.lower()== 'doanes':
            sigma = np.sqrt(6*(Ny-2)/((Ny+1)*(Ny+3)))
            Bx = np.ceil(1+np.log2(Ny)+np.log2(1+abs(skew(DistMat)/sigma)))
        else:
            raise Exception('Please enter a valid binning method')               
    else:
        Bx = Bins
        
    Ppi,_ = np.histogram(DistMat,int(Bx))        
    Ppi = Ppi/Ny    
    if round(sum(Ppi),6) != 1:
        print('Warning: Potential error estimating probabilities.')
        Ppi = Ppi[Ppi!=0]
    elif any(Ppi==0):
        print('Note: '+str(sum(Ppi==0))+'/'+str(len(Ppi))+' bins were empty')
        Ppi = Ppi[Ppi!=0]
           
    Dist = -sum(Ppi*np.log(Ppi)/np.log(Logx))
    if Norm:
        Dist = Dist/(np.log(Bx)/np.log(Logx))
      
    return Dist, Ppi
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