# Redes Sem Peso WiSARD - Soluções para Memória Eficiente  

Leandro Santiago  

# WiSARD - Soluções para Memória Eficiente  

# WiSARD (Wilkie, Stoneham and Aleksander’s Recognition Device)  

Neurônio representado como nó RAM Múltiplos Discriminadores Discriminador = Conjunto de RAMs Entradas endereçam blocos de RAMs usando mapeamento pseudo-aleatório biunívoco  

# WiSARD  

# Discriminador representa uma classe a ser reconhecida  

N RAMs e tuplas M bits para cada tupla A RAM armazena 2 M bits Entrada binária: M $\times$ N bits  

# Etapa de Treinamento  

![](images/7f62923ab6e4a92a8dcbee2d1b6f78282a69d15c5856177355e53e8ae581f019.jpg)  

![](images/5da5e52ceb5e87f1ee5c53fda7bee362acdd448b5246983ea5fc97893e33f50d.jpg)  

![](images/ef0cc98abc200f1b6bff525aaa3b7d31c79ab122468d97d2b6aacde68d3cd1e5.jpg)  

# Etapa de Treinamento  

![](images/77cdb81b1ce877d0dcc482fcf3030f77b25792c0fb13842e5088b3e0ac3bdbfb.jpg)  

![](images/c3872e16fc4e37a5a41826ba70645e833b6aadf0f595d05db4435c3339677f58.jpg)  

![](images/f61fca61c5cc0fb39ca05d3b6bbc133aee0d5993697c8047a868891435f514cd.jpg)  

# Etapa de Treinamento  

![](images/7807eab362feb7f1e84820c93d8ab9f399692eb50be18556245d31c63b446883.jpg)  

# Etapa de Treinamento  

![](images/25096b52d5cae41efb46333a7de42179fccc2514c2ecdf2135217dad4b291cab.jpg)  

# Etapa de Treinamento  

![](images/b97d580dd9678c91c7880431f2933a604da1df588fd62b731052b413a4b44b9a.jpg)  

# Etapa de Classificação  

![](images/2bb494328be5d7784698611b896a5e3aa115656c9645438f41de491a534a1c5f.jpg)  

# Etapa de Classificação  

![](images/7ffd3a3e1d27d2ff0919a5fe8108c42fb52fccc653473c863b3b19841b172c6c.jpg)  

# Etapa de Classificação  

![](images/5331aeaf59977701ac8574edb4a16f53bd614a9c36ef72fbe1c32c27c997804f.jpg)  

# Confiança Relativa  

![](images/8d00087e230e2e5f688f338b7f41195514f1fe4ae08a89a566590f7768b39279.jpg)  

# WiSARD: Problemas  

Capacidade de generalização do modelo depende da quantidade de posições de RAMs preenchidas  

# WiSARD: Problemas  

Capacidade de generalização do modelo depende da quantidade de posições de RAMs preenchidas  

Como lidar com saturação?  

Aumentar tamanho da tupla $$ mais endereços por neurônio Utilizar q-RAM com $q > 1$ (até 8 bits)  

# WiSARD: Problemas  

Capacidade de generalização do modelo depende da quantidade de posições de RAMs preenchidas  

Como lidar com saturação?  

Aumentar tamanho da tupla $$ mais endereços por neurônio ◦ Utilizar q-RAM com $q > 1$ (até 8 bits)  

→ Como lidar com empate nas respostas dos discriminadores? ◦ Bleaching  

# WiSARD: Problemas  

Capacidade de generalização do modelo depende da quantidade de posições de RAMs preenchidas  

Como lidar com saturação?  

Aumentar tamanho da tupla $$ mais endereços por neurônio ◦ Utilizar q-RAM com $q > 1$ (até 8 bits)  

Como lidar com empate nas respostas dos discriminadores? ◦ Bleaching  

Como lidar com grande espaço de memória?  

Tabela Hash - Dicionário Bloom WiSARD - Leandro Santiago (2020) AMQ WiSARD - Leandro Santiago (?)  

# Problema da Memória  

![](images/fc466452edadfb59080004d2abf0abbae028af27d254fd01a3559c22ddf5f435.jpg)  

# Bloom WiSARD  

# Approximate Membership Query - AMQ  

Exact Membership Query  

![](images/1031a10819c47499ea55c989942b97344370d406fd25d8afb196e100a22cc974.jpg)  

Approximate Membership Query  

![](images/31038e5a4b0eeca931d37940d1695869c51973356563e007787308bba776138a.jpg)  

# Bloom Filter  

Conjunto ${ \cal { S } } = \{ x _ { 1 } , x _ { 2 } , . . . , x _ { n } \}$ de $\boldsymbol { n }$ elementos é descrito como m-bit array  

As posições do array $\{ 1 , . . . , m \}$ são geradas por k funções hash independenes $h _ { 1 } , . . . , h _ { k }$  

Todas as posições começam zeradas  

Probabilidade falso positivo baixa  

# Bloom Filter - Insertion Operation  

![](images/a540bb78263afabf656e925a723df1c6d1178c345aeedcce18004d7eb4bb62ee.jpg)  

# Bloom Filter - Insertion Operation  

![](images/b683d1c71f98ae8ad2403fb177d429b79a94e77412136515643421a3930675cc.jpg)  

# Bloom Filter - Insertion Operation  

![](images/e1e06a306e5f4064a70cd14bbf3ffac61c3fb6431c91fc8476ca9c524c730bca.jpg)  

# Bloom Filter - Query Operation  

![](images/9f2115fb81eeb23ac3ce250fa8b4c2bac95daf04ad05c7bf718053c6cf9b5161.jpg)  

# Bloom Filter - Query Operation  

![](images/96bb3d6a40ead4a9fe89542b2edb5e013171ee67b646fbb9adf79c38a807d5af.jpg)  

# Bloom Filter - Query Operation  

![](images/30e8bda2acc15bbde98962c62a5fa2f23fa9df44b9672dc1e9388dd6f9bf676d.jpg)  

# Optimal Parameters to Minimize False Positive  

$m =$ Number of bits.   
$c =$ Capacity of set.   
$k =$ Number of hashes.   
error = Probability of false positive.  

# Formulas:  

$$
{ \begin{array} { r l } & { \circ \ \mathbf { m } = - \mathbf { c } \times { \frac { \mathsf { l n } \mathbf { e r r o r } } { ( \mathsf { l n } 2 ) ^ { 2 } } } . } \\ & { \circ \ \mathbf { k } = \mathbf { m } \times { \frac { \mathsf { l n } 2 } { \mathbf { c } } } . } \end{array} }
$$  

# AMQ com Tuplas  

![](images/480be3490d04d28ae286f7a4f0ff3a6c19830f174b0726cac87c8eba257cc1fd.jpg)  

# Bloom WiSARD Design  

Discriminadores = Conjunto de Bloom Filters  

Função hash: MurmurHash Função hash não criptográfica eficiente para pesquisa baseada em hash  

Double Hashing:  

$$
h ( i , k ) = ( h _ { 1 } ( k ) + i \times h _ { 2 } ( k ) ) { \pmod { n } }
$$  

# Bloom WiSARD - Treinamento  

![](images/baec4d671b396fc7cf8e78f3dc96013baac9c03e0f2132eb4997f309e2546c9a.jpg)  
Bloom Filter 2   
Bloom Filter 3  

# Bloom WiSARD - Inferência  

![](images/341d36692b3a00d98620c182eaed01b67ee9df16f7dd683bb5747bedfc003d67.jpg)  

# Experimental Setup  

WiSARD versions: Standard WiSARD Dictionary WiSARD Bloom WiSARD  

Total of 17 datasets 6 Binary Classification 11 Multiclass Classification  

Random 3-fold cross validation when dataset does not provide separately training and testing set   
20 runs for each scenario  

# Results for Binary Classification  

<html><body><table><tr><td>Dataset</td><td>WNN</td><td>Acc</td><td>Train</td><td>Test</td><td>Memory (KB)</td></tr><tr><td>Adult</td><td>WiSARD Dict WiSARD Bloom WiSARD</td><td>0.722 0.721 0.718</td><td>4.414 1.947 1.932</td><td>1.05 1.188 1.166</td><td>8,978,432.000 383.535 80.173</td></tr><tr><td>Australian</td><td>WiSARD Dict WiSARD Bloom WiSARD</td><td>0.843 0.841 0.834</td><td>0.002 0.002 0.002</td><td>0.001 0.001 0.001</td><td>4,096.000 11.299 8.613</td></tr><tr><td>Banana</td><td>WiSARD Dict WiSARD Bloom WiSARD</td><td>0.87 0.871 0.864</td><td>0.052 0.054 0.058</td><td>0.028 0.033 0.036</td><td>13,312.000 23.428 3.047</td></tr><tr><td>Diabetes</td><td>WiSARD Dict WiSARD Bloom WiSARD</td><td>0.698 0.689 0.69</td><td>0.001 0.001 0.001</td><td>0.0007 0.0008 0.0008</td><td>2,048.000 6.553 4.793</td></tr><tr><td>Liver</td><td>WiSARD DictWiSARD Bloom WiSARD</td><td>0.593 0.587 0.591</td><td>0.001 0.001 0.001</td><td>0.0007 0.0008 0.0009</td><td>5,120.000 6.387 2.344</td></tr><tr><td>Mushroom</td><td>WiSARD DictWiSARD Bloom WiSARD</td><td>1 1 1</td><td>0.051 0.054 0.057</td><td>0.028 0.033 0.035</td><td>8,192.000 19.209 3.750</td></tr></table></body></html>  

# Results for Multiclass Classification - Part 1  

<html><body><table><tr><td>Dataset</td><td>WNN</td><td>Acc</td><td>Train</td><td>Test</td><td>Memory (KB)</td></tr><tr><td rowspan="3">Ecoli</td><td>WiSARD</td><td>0.793</td><td>0.0005</td><td>0.0005</td><td>7,168.000</td></tr><tr><td>DictWiSARD</td><td>0.799</td><td>0.0005</td><td>0.0005</td><td>5.664</td></tr><tr><td>Bloom WiSARD</td><td>0.799</td><td>0.0005</td><td>0.0007</td><td>3.281</td></tr><tr><td rowspan="3">Glass</td><td>WiSARD</td><td>0.72</td><td>0.003</td><td>0.003</td><td>51,968.000</td></tr><tr><td>DictWiSARD</td><td>0.73</td><td>0.003</td><td>0.003</td><td>20.884</td></tr><tr><td>Bloom WiSARD</td><td>0.726</td><td>0.003</td><td>0.003</td><td>23.789</td></tr><tr><td rowspan="3">Iris</td><td>WiSARD Dict WiSARD</td><td>0.985</td><td>0.0001</td><td>0.000009</td><td>1,536.000</td></tr><tr><td></td><td>0.977</td><td>0.0001</td><td>0.000008</td><td>0.747</td></tr><tr><td>Bloom WiSARD</td><td>0.976</td><td>0.0001</td><td>0.0001</td><td>0.703</td></tr><tr><td rowspan="3">Letter</td><td>WiSARD</td><td>0.845</td><td>1.483</td><td>0.16</td><td>10,223,616.000</td></tr><tr><td>Dict WiSARD</td><td>0.846</td><td>0.0717</td><td>0.22</td><td>121.748</td></tr><tr><td>Bloom WiSARD</td><td>0.848</td><td>0.07</td><td>0.208</td><td>91.292</td></tr><tr><td rowspan="3">MNIST</td><td>WiSARD</td><td>0.917</td><td>4.317</td><td>0.33</td><td>9,175,040.000</td></tr><tr><td>Dict WiSARD</td><td>0.916</td><td>0.811</td><td>0.475</td><td>1,368.457</td></tr><tr><td>Bloom WiSARD</td><td>0.915</td><td>0.77</td><td>0.369</td><td>819.049</td></tr></table></body></html>  

# Results for Multiclass Classification - Part 2  

<html><body><table><tr><td>Dataset</td><td>WNN</td><td>Acc</td><td>Train</td><td>Test</td><td>Memory (KB)</td></tr><tr><td rowspan="2">Satimage</td><td>WiSARD</td><td>0.851</td><td>0.048</td><td>0.034</td><td>27,648.000</td></tr><tr><td>Dict WiSARD</td><td>0.853</td><td>0.05</td><td>0.049</td><td>69.141</td></tr><tr><td rowspan="2">Segment</td><td>Bloom WiSARD WiSARD</td><td>0.851 0.935</td><td>0.053 0.009</td><td>0.05 0.007</td><td>12.656 17,024.000</td></tr><tr><td>Dict WiSARD</td><td>0.934</td><td>0.009</td><td>0.01</td><td>7.724</td></tr><tr><td rowspan="2"></td><td>Bloom WiSARD</td><td>0.933</td><td>0.01</td><td>0.011</td><td>7.793</td></tr><tr><td>WiSARD</td><td>0.87</td><td>0.119</td><td>0.064</td><td>8,064.000</td></tr><tr><td rowspan="2">Shuttle</td><td>Dict WiSARD</td><td>0.869</td><td>0.12</td><td>0.078</td><td>4.956</td></tr><tr><td>Bloom WiSARD WiSARD</td><td>0.868 0.67</td><td>0.132</td><td>0.103</td><td>3.691</td></tr><tr><td rowspan="2">Vehicle</td><td>Dict WiSARD</td><td>0.672</td><td>0.003 0.003</td><td>0.0021 0.0026</td><td>9,216.000</td></tr><tr><td>Bloom WiSARD</td><td>0.662</td><td>0.003</td><td>0.0028</td><td>17.617 4.219</td></tr><tr><td rowspan="2">Vowel</td><td>WiSARD</td><td>0.876</td><td>0.0023</td><td>0.0025</td><td>14,080.000</td></tr><tr><td>DictWiSARD</td><td>0.876</td><td>0.0023</td><td>0.0032</td><td>16.221</td></tr><tr><td rowspan="4">Wine</td><td>Bloom WiSARD</td><td>0.879</td><td>0.0022</td><td>0.0036</td><td>6.445</td></tr><tr><td>WiSARD</td><td>0.932</td><td>0.0006</td><td>0.0003</td><td>4,992.000</td></tr><tr><td>Dict WiSARD</td><td>0.924</td><td>0.0005</td><td>0.0003</td><td>4.248</td></tr><tr><td>Bloom WiSARD</td><td>0.926</td><td>0.0005</td><td>0.0004</td><td>2.285</td></tr></table></body></html>  

# Memory x Accuracy x Probability of False Positive - Part 1  

![](images/d5f03a723f83737eac2e7a9b74acd88ff44c6fcbcc3055ab68477c15c3a318e4.jpg)  

# Memory x Accuracy x Probability of False Positive - Part 2  

![](images/8f15242dbf48ec4eba74867a9d9bbf26ede482ae47dfe31cb85dfe9d03aa4cd4.jpg)  

# Memory x Accuracy x Probability of False Positive - Part 3  

![](images/774f591a14a31d84be0c148dfe762159f98bb010c0a0b44923be6df1b023f934.jpg)  

# Memory x Accuracy x Probability of False Positive - Part 4  

![](images/a6eb12441ebcdc85ddf6497714e31be896ef06c8b11903a45ece459fe3b436db.jpg)  
Bloom WiSARD Memory × Accuracy× Propability of False Positive  

# Conclusão  

Bloom filter economiza mais recursos de memória do que o Dicionário  

Bloom filter com 10% de Falso Positivo mantêm boa precisão e reduz a memória em cerca de 7,7 vezes em comparação com a implementação de dicionários e até 6 ordens de magnitude quando comparados à implementação padrão do WiSARD  

Bloom filter com 50% de Falso Positivo também mantêm boa precisão e reduz a memória em cerca de 3,3 vezes em comparação com 10% da configuração de Falso Positivo  

# BTHOWeN  

# Motivação  

Crescimento da área de Edge Computing TinyML  

Evolução de aceleradores para Deep Learning GP-GPU: CUDA TPU: Tensor Processing Unit  

# Tipos de Hardware para AI  

![](images/24c90dfd8c48c11b0defeaec8ade5b20747ba1d258100e2773bc33d761cc7cdc.jpg)  

# Reconfigurable Computing - FPGA  

![](images/5daf3c33d6eb3b3fa503e01bcd253b847d13eb7235fd42f3b4f03f443665530c.jpg)  

![](images/7f5751ef7b59de63320803921377f176bec4bc3f354440ad17af9479a0ee63c7.jpg)  

# BTHOWeN  

BTHOWeN = Bleached Thermometer-encoded Hashed-input Optimized Weightless Neural Network  

Implementação de hardware para edge inference FPGA - System Verilog  

Técnicas utilizadas: Counting Bloom Filter Gaussian non-linear thermometer ◦ Bleaching (B-bleaching)  

# Counting Bloom Filter  

Bloom filter Inserção ◦ Query  

Counting Bloom filter Deleção  

# Counting Bloom Filter - Insertion Operation  

![](images/cf1e568c6d5a76dec72715acb67c7ad1d078f70f2e4f984fe78b9facfb646659.jpg)  

# Counting Bloom Filter - Insertion Operation  

![](images/395382d0b09d028409f5d0b96edab4032422b06474830719a7b5adbc97ce5282.jpg)  

# Counting Bloom Filter - Insertion Operation  

![](images/c30eb7e027fdf4cd02df6c6c7f48809421abac5701df569507765e5ef8396e67.jpg)  

# Counting Bloom Filter - Insertion Operation  

![](images/76991a0f8589c1739c6ef6355020ede5b69b9f7f2c3c04406c894101027d2050.jpg)  

# Counting Bloom Filter - Query Operation  

![](images/fbcc1294bb624f8d3d798ffa799e33658d18962b407607039b3277f1465d16d5.jpg)  

# Counting Bloom Filter - Query Operation  

![](images/0f93efcd3c3d4e7d856e4cf1ae9e3090e2db7af543ad9819b94ed2d9f4081a33.jpg)  

# Counting Bloom Filter - Query Operation  

![](images/546044f174278cd66834287afb65b5ebba1e5114718154d105896d8863473d5a.jpg)  

# Counting Bloom Filter - Deletion Operation  

![](images/e9e65252b72481443fbb27babfa8295aa7d67e1da7c438f3be438f9740906fdd.jpg)  

# Counting Bloom Filter - Deletion Operation  

![](images/cad2503da6dff17e0eb56342117423ea2d91dcd2729afbc5568604f171d02b3f.jpg)  

# Counting Bloom Filter - Deletion Operation  

![](images/6f8e0016bf206626b6c4253e26adba7c60e30c52ff88c025bb31df6b7a68966e.jpg)  

# Counting Bloom Filter - Deletion Operation  

![](images/a4a3c961b1f2129d7d5e9ab94e03d5842b5ebb072be0c3457e683b32732fde28.jpg)  

# Bloom Filter Design  

Discriminadores = Conjunto de Counting Bloom Filters  

Função hash: H3 para mapear de n m bits Dimensão: x (1 x n), p (n x m)  

$$
h ( x ) = x [ 0 ] p _ { 0 } \oplus x [ 1 ] p _ { 1 } \oplus . . . \oplus x [ n - 1 ] p _ { n - 1 }
$$  

# Arquitetura do BTHOWeN  

![](images/477b6933f79009f6deb992bac5d253090a4318c0aafe42d5a93477a9773783be.jpg)  

# BTHOWeN - Treinamento  

![](images/41628c1e3fd1f4b630bbc48fe5b0b1a414077b44f38201ff4a9797cdc0738fe5.jpg)  

# Experimental Setup  

Modelos: BTHOWeN O MLP C CNN - LeNet-1  

TensorFlow Lite (quantização com 8-bit) e hls4ml tool Total de 9 datasets  

# Acurácia dos Modelos do BTHOWeN  

<html><body><table><tr><td>Model Name</td><td>Bits /Input</td><td>Bits /Filter</td><td>Entries /Filter</td><td>Hashes /Filter</td><td>Size (KiB)</td><td>Test Acc.</td></tr><tr><td>MNIST-Small</td><td>2</td><td>28</td><td>1024</td><td>2</td><td>70.0</td><td>0.934</td></tr><tr><td>MNIST-Medium</td><td>3</td><td>28</td><td>2048</td><td>2</td><td>210</td><td>0.943</td></tr><tr><td>MNIST-Large</td><td>6</td><td>49</td><td>8192</td><td>4</td><td>960</td><td>0.952</td></tr><tr><td>Ecoli</td><td>10</td><td>10</td><td>128</td><td>2</td><td>0.875</td><td>0.875</td></tr><tr><td>Iris</td><td>3</td><td>2</td><td>128</td><td>1</td><td>0.281</td><td>0.980</td></tr><tr><td>Letter</td><td>15</td><td>20</td><td>2048</td><td>4</td><td>78.0</td><td>0.900</td></tr><tr><td>Satimage</td><td>8</td><td>12</td><td>512</td><td>4</td><td>9.00</td><td>0.880</td></tr><tr><td>Shuttle</td><td>9</td><td>27</td><td>1024</td><td>2</td><td>2.63</td><td>0.999</td></tr><tr><td>Vehicle</td><td>16</td><td>16</td><td>256</td><td>3</td><td>2.25</td><td>0.762</td></tr><tr><td>Vowel</td><td>15</td><td>15</td><td>256</td><td>4</td><td>3.44</td><td>0.900</td></tr><tr><td>Wine</td><td>9</td><td>13</td><td>128</td><td>3</td><td>0.422</td><td>0.983</td></tr></table></body></html>  

# Comparação BTHOWeN e DNN Quantizadas - Parte 1  

<html><body><table><tr><td>Dataset</td><td>Model</td><td>Bus Width</td><td>Cycles per Inf.</td><td>Hash Units</td><td>Dyn. Power (Tot. Power) (W)</td><td>Dyn. Energy (Tot. Energy) (nJ/Inf.)</td><td>LUTs</td><td>FFs</td><td>BRAMsDSPs (36Kb)</td><td></td><td> Accuracy</td></tr><tr><td rowspan="8">MNIST</td><td>BTHOWeN-Small</td><td>64</td><td>25</td><td>5</td><td>0.195 (0.303)</td><td>48.75 (75.8)</td><td>15756</td><td>3522</td><td>0</td><td>0</td><td>0.934</td></tr><tr><td>BTHOWeN-Medium</td><td>64</td><td>37</td><td>5</td><td>0.386 (0.497)</td><td>142.8 (183.9)</td><td>38912</td><td>6577</td><td>0</td><td>0</td><td>0.943</td></tr><tr><td>BTHOWeN-Large</td><td>64</td><td>74</td><td>6</td><td>3.007 (3.509)</td><td>2225 (2597)</td><td>151704</td><td>18796</td><td>0</td><td>0</td><td>0.952</td></tr><tr><td>BTHOWeN-Large*</td><td>256</td><td>19</td><td>24</td><td>3.158 (3.661)</td><td>600.0 (695.6)</td><td>158367</td><td>25905</td><td>0</td><td>0</td><td>0.952</td></tr><tr><td>MLP784-16-10</td><td>64</td><td>846</td><td>-</td><td>0.029 (0.134)</td><td>245 (1133)</td><td>2163</td><td>3007</td><td>8</td><td>28</td><td>0.946</td></tr><tr><td>CNN1 (LeNet1) [34]</td><td>64</td><td>33615</td><td>-</td><td>0.058 (0.163)</td><td>19497 (54792)</td><td>5753</td><td>3115</td><td>7</td><td>18</td><td>0.947</td></tr><tr><td>CNN 2 (LeNet1) [34]</td><td>64</td><td>33555</td><td>-</td><td>0.043 (0.148)</td><td>14429 (49661)</td><td>3718</td><td>2208</td><td>5</td><td>10</td><td>0.920</td></tr><tr><td>HashedWNN[19]</td><td>32</td><td>28</td><td>-</td><td>0.423 (0.528)</td><td>118.4 (147.8)</td><td>9636</td><td>4568</td><td>128.5</td><td>5</td><td>0.907</td></tr><tr><td rowspan="2">Ecoli</td><td>BTHOWeN</td><td>64</td><td>2</td><td>7</td><td>0.012 (0.117)</td><td>0.24 (2.34)</td><td>353</td><td>223</td><td>0</td><td>0</td><td>0.875</td></tr><tr><td>MLP7-8-8</td><td>64</td><td>14</td><td></td><td>0.03 (0.135)</td><td>4.2 (18.9)</td><td>1596</td><td>1615</td><td>0</td><td>0</td><td>0.875</td></tr><tr><td rowspan="2">Iris</td><td>BTHOWeN</td><td>64</td><td>1</td><td>6</td><td>0.005 (0.109)</td><td>0.05 (1.09)</td><td>57</td><td>90</td><td>0</td><td>0</td><td>0.980</td></tr><tr><td>MLP 4-4-3</td><td>64</td><td>10</td><td>-</td><td>0.008 (0.112)</td><td>0.8 (11.2)</td><td>427</td><td>488</td><td>0</td><td>0</td><td>0.980</td></tr></table></body></html>  

# Comparação BTHOWeN e DNN Quantizadas - Parte 2  

<html><body><table><tr><td rowspan="2">Dataset</td><td rowspan="2">Model</td><td rowspan="2">Bus Width</td><td rowspan="2">Cycles per</td><td rowspan="2">Hash Units</td><td rowspan="2">Dyn. Power (Tot.</td><td rowspan="2">Dyn. Energy (Tot. Energy)</td><td rowspan="2">LUTs</td><td rowspan="2">FFs</td><td rowspan="2">BRAMs (36Kb)</td><td rowspan="2">DSPs</td><td rowspan="2">Accuracy</td></tr><tr><td>(nJ/Inf.)</td></tr><tr><td rowspan="2">Letter</td><td>BTHOWeN</td><td>64</td><td>Inf. 4</td><td>12</td><td>Power) (W) 0.623 (0.738)</td><td>24.92 (29.52)</td><td>21603</td><td>2715</td><td>0</td><td>0</td><td>0.900</td></tr><tr><td>MLP 16-40-26</td><td>64</td><td>26</td><td>-</td><td>0.109 (0.259)</td><td>39.52 (67.34)</td><td>17305</td><td>15738</td><td>0</td><td>0</td><td>0.904</td></tr><tr><td rowspan="2"> Satimage</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>MBTHOWeN6</td><td>6464</td><td>525</td><td>24</td><td>0.084 (0.140)</td><td>4.2(9.5)</td><td>3771</td><td>1131</td><td>00</td><td>00</td><td>0.880</td></tr><tr><td rowspan="2">Shuttle</td><td></td><td>6464</td><td></td><td></td><td></td><td></td><td>593</td><td></td><td></td><td></td><td></td></tr><tr><td>BTHOWeN</td><td></td><td>24</td><td>3-</td><td>0.018 (0.12)</td><td>0.36 (2.462)</td><td></td><td>121</td><td>00</td><td>00</td><td>0.999</td></tr><tr><td rowspan="2">Vehicle</td><td>BTHOWeN</td><td>64</td><td>5</td><td>18</td><td>0.038 (0.143)</td><td>1.9 (7.15)</td><td>1781</td><td>597</td><td>0</td><td>0</td><td>0.762</td></tr><tr><td>MLP 18-16-4</td><td>64</td><td>15</td><td>-</td><td>0.024 (0.128)</td><td>3.6 (19.2)</td><td>2824</td><td>3035</td><td>0</td><td>0</td><td>0.766</td></tr><tr><td rowspan="2">Vowel</td><td>BTHOWeN</td><td>64</td><td>2</td><td>12</td><td>0.040 (0.145)</td><td>0.8 (2.9)</td><td>1559</td><td>756</td><td>0</td><td>0</td><td>0.900</td></tr><tr><td>MLP 10-18-11</td><td>64</td><td>18</td><td>-</td><td>0.070 (0.175)</td><td>12.6 (31.5)</td><td>5743</td><td>4663</td><td>0</td><td>0</td><td>0.903</td></tr><tr><td rowspan="2">Wine</td><td></td><td>6464</td><td>34</td><td>9-</td><td>0.01 (0.117)</td><td>0.36 (.1)</td><td>585</td><td>239</td><td>00</td><td>00</td><td></td></tr><tr><td>MLP13-103</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>0.983</td></tr></table></body></html>  

# BTHOWeN vs DNN - Latência e Energia Relativa  

![](images/08d299aa86d8afcb2a7281ced8da1e2d869b9d8ba8a425e202111943be71cb90.jpg)  

# Conclusão  

Função H3 e Termômetro Gaussiano melhorou a acurácia do MNIST de 91.5% para 95.2%  

BTHOWeN é competitivo com outras arquiteturas: WNN, MLP e CNN  

Em relação a outras arquitetura de WNN, BTHOWeN reduz erro por 41% e o tamanho do modelo by 51% considerando os 9 datasets  

Comparado aos modelos MLP, BTHOWeN consome 20% da energia total enquanto reduz a latência em 85%  

As melhorias de energia/latência em relação às CNNs são ainda maiores, embora as CNNs tenham maior precisão  

# ULEEN  

# ULEEN  

ULEEN = Ultra-low-energy Edge Networks  

$$ Implementação de hardware para edge inference - FPGA  

Técnicas adicionais: Continuos Bloom Filter Ensemble Pruning Multi-pass training  

# Visão Geral da ULEEN  

![](images/e296cb9e37afa8b398148897e01f8eff98c5cd95d219c424d893747a8d1ee44a.jpg)  

# Continuos Bloom Filter  

→ Floating-point array → [-1, 1]  

Gradiente $\begin{array} { r } { s i g n ( x ) = \left\{ \begin{array} { l l } { - 1 \quad } & { x < 0 } \\ { 1 \quad } & { x \geq 0 } \end{array} \right. } \end{array}$  

$$
s i g n ^ { \prime } ( x ) = \left\{ \begin{array} { l l } { { + \infty } } & { { \quad x = 0 } } \\ { { 0 } } & { { \quad x \neq 0 } } \end{array} \right.
$$  

# Continuos Bloom Filter  

Straight-through estimator (STE) function  

# Gradiente  

$$
S T E ( x ) = \left\{ { \begin{array} { l l } { - 1 } & { \quad x < 0 } \\ { 1 } & { \quad x \geq 0 } \end{array} } \right.
$$  

$$
S T E ^ { \prime } ( x ) = { \left\{ \begin{array} { l l } { 1 } & { \quad | x | \leq 1 } \\ { 0 } & { \quad | x | > 1 } \end{array} \right. }
$$  

# Continuos Bloom Filter  

![](images/b051d74654542182745ba5970d950018878c4201f8298101624779db2230546a.jpg)  

# Ensemble  

![](images/d73330eb7bdd614684b8bcbc72d7a8722b188982b1633637a6480bdbf1d9510e.jpg)  

# Pruning  

Discriminadores são testados com seus exemplos de treinamento. É analisado quais neurônios retornam 1  

Utility score (US) baseado nas taxas de falso (e verdadeiro) positivo (e negativo)  

$$
U S = ( M - 1 ) ( T P R - F N R ) + ( T N R - F P R )
$$  

Fração fixa de neurônios em cada discriminador com US mais baixas são removidos do modelo  

bias inteiro adicionado nas respostas dos neurônios que é aprendido  

# Pruning  

# Uniform Pruning  

![](images/8d6fe8a8cba92587d596aa6bf0e80ea68cfcc7fc1dbf194dc5e63580c264c15a.jpg)  
No Pruning  

![](images/f3cfd9bb207a9f67560c5ff6e9a86987135dffbcf084ed6c64e38fe3c5a7a196.jpg)  
Non-Uniform Pruning  

![](images/bb75f02bc33d92ce5afc58d262907f1d37a3cf4f3a5b33ec8c45fdb654cccb6f.jpg)  

# Multi-Pass Training  

Continuous Bloom filter são inicializados com valores aleatórios [-1, 1]  

Backpropation com STE  

Função de custo: Cross-entropy loss  

Dropout = 0.5, Adam optimizer, learning rate = 10−3  

Saída neurônio: {-1, 0, 1}  

# Multi-Pass Training  

![](images/540453dcf64b4167ac87d3418a334fc9f759eaf3442980e75a72d1c61c092cce.jpg)  

# Inferência  

Countinuos Bloom filter é convertido em Bloom filter usando unit step function  

Saída neurônio: {-1, 1}  

# Arquitetura ULEEN  

![](images/52184a1640b2ac8c819106e7d8cb067125e6dcb0de63a4bcd086b86bc57c4dba.jpg)  

# Implementação do Modelo  

![](images/e85e4aaa0cfe9ecf370a829fda9b74c6f437e2fdab37aba37473c158bb36b095.jpg)  

# Experimental Setup  

Modelos: O ULEEN Xilinx FINN - (3 hidden layers) ? BTHOWeN  

Total de 5 datasets: MNIST + 4 MLPerf Tiny dataset  

# Acurácia: ULEEN vs FINN  

<html><body><table><tr><td rowspan="2">Dataset</td><td colspan="2">ULEEN</td><td colspan="3">FINN BNN Iso-accuracy</td><td colspan="3">FINN BNN Iso-size</td></tr><tr><td>Size (KiB)</td><td>Test Acc.%</td><td>Size (KiB)</td><td>Test Acc.%</td><td>Hidden layer size</td><td>Size (KiB)</td><td>Test Acc.%</td><td>Hidden layer size</td></tr><tr><td>MNIST-S</td><td>16.9</td><td>96.2</td><td>40.8</td><td>95.8</td><td>256 × 3 (SFC)</td><td>16.4</td><td>95.2</td><td>128 ×3</td></tr><tr><td>MNIST-M</td><td>101</td><td>97.8</td><td>114</td><td>97.7</td><td>512 ×3 (MFC)</td><td>103</td><td>97.7</td><td>480 ×3</td></tr><tr><td>MNIST-L</td><td>262</td><td>98.5</td><td>355</td><td>98.4</td><td>1,024 × 3 (LFC)</td><td>283</td><td>98.0</td><td>896×3</td></tr><tr><td>KWS</td><td>101</td><td>70.3</td><td>324</td><td>70.6</td><td>1,024 × 3</td><td>101</td><td>67.0</td><td>524×3</td></tr><tr><td>ToyADMOS</td><td>16.6</td><td>86.3</td><td>36.1</td><td>86.1</td><td>256×3</td><td>16.4</td><td>85.5</td><td>144×3</td></tr><tr><td>VWW</td><td>251</td><td>61.8</td><td>3,329*</td><td>57.1*</td><td>2,048 × 3*</td><td>264</td><td>55.7</td><td>224×3</td></tr><tr><td>CIFAR-10</td><td>1,379</td><td>54.2</td><td>19,466*</td><td>45.7*</td><td>8,192 × 3*</td><td>1,345</td><td>44.4</td><td>1,700 × 3</td></tr></table></body></html>  

# Detalhes Modelos ULEEN  

<html><body><table><tr><td>Dataset</td><td>Model</td><td>Submodel</td><td>Bits/Inp</td><td>Inputs/Filter</td><td>Entries/Filter</td><td>Size (KiB)</td><td>Test Acc.%</td></tr><tr><td rowspan="9">MNIST</td><td rowspan="3">ULN-S</td><td>Ensemble</td><td>2</td><td>1</td><td>1</td><td>16.9</td><td>96.20</td></tr><tr><td>SM0</td><td>2</td><td>12</td><td>64</td><td>7.19</td><td>92.91</td></tr><tr><td>SM1 SM2</td><td>2 2</td><td>16 20</td><td>64 64</td><td>5.39 4.38</td><td>90.25 86.16</td></tr><tr><td rowspan="2">ULN-M</td><td>Ensemble SM0</td><td>3 3</td><td>二 12</td><td>二 64</td><td>101 10.9</td><td>97.79 83.54</td></tr><tr><td>SM1 SM2 SM3 SM4</td><td>3 3 3 3</td><td>16 20 28 36</td><td>128 256 256</td><td>16.0 26.0 18.44</td><td>90.93 92.92 87.05</td></tr><tr><td rowspan="2">ULN-L</td><td>Ensemble SM0 SM1</td><td>7 7 7</td><td>二 12 16</td><td>512 二 64</td><td>29.38 262 25.0</td><td>80.93 98.46 88.78</td></tr><tr><td>SM2 SM3 SM4 SM5</td><td>7 7 7</td><td>20 24 28</td><td>128 128 256 256</td><td>37.7 30.2 50.3 43.1</td><td>93.24 92.44 93.92</td></tr></table></body></html>  

# Detalhes Modelos ULEEN  

<html><body><table><tr><td>Dataset</td><td>Model</td><td>Submodel</td><td>Bits/Inp</td><td>Inputs/Filter</td><td>Entries/Filter</td><td>Size (KiB)</td><td>Test Acc.%</td></tr><tr><td colspan="2" rowspan="3">KWS</td><td>Ensemble</td><td>12</td><td>二</td><td>二</td><td>101</td><td>70.34</td></tr><tr><td>SM0</td><td>12</td><td>5</td><td>8</td><td>9.62</td><td>56.93</td></tr><tr><td>SM1 SM2</td><td>12 12</td><td>6 7</td><td>16 32</td><td>16.1 27.5</td><td>59.32 59.94</td></tr><tr><td rowspan="5">ToyADMOS</td><td>SM3</td><td>12</td><td>8</td><td>64</td><td>48.12</td><td>61.01</td></tr><tr><td>Ensemble</td><td>6</td><td>二</td><td></td><td>16.6</td><td>86.33</td></tr><tr><td>SM0</td><td>6</td><td>7</td><td>64</td><td>6.88</td><td>83.61</td></tr><tr><td>SM1</td><td>6</td><td>9</td><td>64</td><td>5.34</td><td>82.32</td></tr><tr><td>SM2</td><td>6</td><td>11</td><td>64</td><td>4.38</td><td>79.85</td></tr><tr><td rowspan="5">VWW</td><td>Ensemble</td><td>12</td><td>二</td><td>二</td><td>251</td><td>61.76</td></tr><tr><td>SM0</td><td>12</td><td>5</td><td>8</td><td>30.2</td><td>59.07</td></tr><tr><td>SM1</td><td>12</td><td>7</td><td>16</td><td>43.2</td><td>57.78</td></tr><tr><td>SM2</td><td>12</td><td>9</td><td>32</td><td>67.2</td><td>59.20</td></tr><tr><td>SM3</td><td>12</td><td>11</td><td>64</td><td>110</td><td>58.96</td></tr><tr><td rowspan="5">CIFAR-10</td><td>Ensemble</td><td>8</td><td>二</td><td>二</td><td>1379</td><td>54.21</td></tr><tr><td>SM0</td><td>8</td><td>6</td><td>32</td><td>112</td><td>49.12</td></tr><tr><td>SM1</td><td>8</td><td>8</td><td>64</td><td>168</td><td>49.53</td></tr><tr><td>SM2</td><td>8</td><td>12</td><td>128</td><td>224</td><td>46.39</td></tr><tr><td>SM3</td><td>8</td><td>16</td><td>256</td><td>336</td><td>42.23</td></tr><tr><td></td><td>SM4</td><td>8</td><td>20</td><td>512</td><td>538</td><td>38.27</td></tr></table></body></html>  

# Eficiência Energética e de Área  

![](images/15f88ba126040a438816585529908bba8a0ee019acd7a9590c15a5b06199e0fe.jpg)  

![](images/e54462256419f94d108a64542b67b48c301023df6c4a86650c67ae068e0a0cbf.jpg)  

# Conclusão  

Ensemble, pruning e multi-pass training melhoram a acurácia do modelo e reduz o tamanho dos parâmetros  

Comparação com FINN: melhora na eficiência energética em estado estacionário em 3,8–9,1x e area-delay product em 1,7–7,8x nos datasets MNIST, KWS e ToyADMOS/car  

ULEEN avança significativamente o estado da arte em precisão de WNN, demonstrando que WNNs podem superar até mesmo BNNs altamente otimizados para algumas aplicações  

# Conclusão  

# Comparação dos Modelos WNN  

<html><body><table><tr><td>Model</td><td>Bloom filters</td><td>Thermometer encoding</td><td>Submodel ensembles</td><td>Bleaching</td><td>Multi-pass training</td><td> Pruning</td></tr><tr><td>WiSARD [4]</td><td>X</td><td>X</td><td>X</td><td>X</td><td>X</td><td>X</td></tr><tr><td>Bloom WiSARD[55]</td><td>√</td><td>X</td><td>X</td><td>X</td><td>X</td><td>X</td></tr><tr><td>WiSARD Encodings [31]</td><td>X</td><td>√</td><td>X</td><td>√</td><td>X</td><td>X</td></tr><tr><td>Regression WiSARD [44]</td><td>X</td><td>√</td><td>√</td><td>√</td><td>X</td><td>X</td></tr><tr><td>BTHOWeN[57]</td><td>√</td><td>√</td><td>X</td><td>√</td><td>X</td><td>X</td></tr><tr><td>ULEEN (this work)</td><td>√</td><td>√</td><td>√</td><td>X*</td><td>√</td><td>√</td></tr></table></body></html>  

# WiSARD - Outros Tipos de Memória  

Ainda é possível reduzir os espaço de memória? Estrutura de Dados Esparsas - ex: SDM, SDR Redução da Tabela Verdade - Mapa de Karnaugh Gerar uma função que substitua memória - Problema XOR Outras representações de sistemas numéricos - RNS  

# Tópicos de Pesquisas  

Segurança-Assistida por Hardware Computação de Alto Desempenho Computação Reconfigurável $$ Inteligência Artificial Verde Redes Neurais sem Peso ◦ HyperDimensional Computing  

# Tópicos de Pesquisas  

HyperDimensional Computing Série Temporal - Forecasting e Nowcasting   
Processamento de Linguagem Natural Aprendizado Federado   
◦ TinyML  

# Redes Sem Peso WiSARD - Soluções para Memória Eficiente  

Leandro Santiago  

# leandro@ic.uff.br  