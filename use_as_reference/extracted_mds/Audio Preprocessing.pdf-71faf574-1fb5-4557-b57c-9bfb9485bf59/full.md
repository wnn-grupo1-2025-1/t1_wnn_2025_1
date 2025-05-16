# Audio Preprocessing  

# MFCC  

Mel Frequency Cepstral Coefficients   
cepstral - spectral   
time signal $\Rightarrow$ Fourier transform $\phantom { 0 } \mathrm { - } >$ frequency spectrum Fs  

![](images/cc62e27b2a2f6c3c4398da3c855d49630ac5e8692bd69a8069c813e5d63618a1.jpg)  

# MFCC  

# cepstrum $\mathbf { \tau } = \mathbf { \tau }$ Discrete Cosine Transform (log(|Fs|))  

![](images/ed2a6ae5e99528009946440809fbcf873d7b66cca47ca50b543cefe884bcf3a3.jpg)  

# MFCC  

Mel scale - scale that relates the perceived frequency of a tone to the actual measured frequency  

$$
( f ) = 2 5 9 5 \log \left( 1 + { \frac { \ l } { 7 } } \right.
$$  

● Filter banks - Representation of the signal obtained after dividing the audio into slices, forming bandpass filters, which are excited by the original signal. MFCC - Cepstral obtained from filter banks  

![](images/164ff3c4518934564d77583ecffa90f205d6a3eb98f2b5d06cd441d11246669a.jpg)  

# Pre-processing pipeline  

MFCC   
Append Sum (integration)   
Database Z-score (each signal has it means subtracted and is later divided by   
its standard deviation)   
Tanh   
Kernel Canvas  