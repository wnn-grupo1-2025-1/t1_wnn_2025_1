**Avaliação do Modelo BTHOWeN em Datasets MultiClasses**

**Integrantes do grupo:** Breno Tostes, Eduardo Naslausky, Felipe Barrocas, Giordano Souza, Maria Bianca Irace, Miguel Sousa, Rafael Paladini

1. Introdução  
2. Proposta  
   1. Metodologia  
   2. Hiperparâmetros
   3. Datasets  
3. Resultados  
4. Conclusão  
5. Referências

1. **Introdução**

BTHOWeN vem de Bleached Thermometer-encoded Hashed-input Optimized Weightless Neural Network, e como o nome indica, é uma arquitetura de rede neural sem peso que se diferencia do uso do modelo WiSARD por incorporar counting Bloom filters para reduzir o tamanho do modelo e permitir bleaching, usar função hash que não requer operações aritméticas e fazer encoding com termômetro não-linear de distribuição normal para melhorar a acurácia do modelo. \[1\]

2. **Proposta**  
   **a. Metodologia**

Neste trabalho iremos replicar o BTHOWeN utilizando o BloomWisard como cerne, com os diferentes datasets multiclasses inclusos no artigo original. A proporção entre a massa de treinamento e a massa de teste foi mantida idêntica às respectivas proporções originais, porém os hiperparâmetros foram configurados para verificar o impacto de cada um deles na acurácia final. Estes hiperparâmetros configuráveis são os de tamanho do endereço (ou tamanho da tupla), número de discriminadores, número de funções hash, número de filtros de bloom, e o fator do bleaching. Os experimentos são feitos alterando um hiperparâmetro de cada vez, de maneira gulosa. Ao atingir um valor máximo variando apenas um hiperparâmetro, este tem seu valor mantido pelo resto do experimento, e o próximo hiperparâmetro passa a ser o variável. Ao realizar este procedimento com todos os hiperparâmetros, identificamos o melhor resultado. Tomamos o maior valor de acurácia de todos os experimentos e sua configuração de hiperparâmetros como melhor valor obtido e o comparamos com a acurácia obtida no artigo original.

**b. Hiperparâmetros**

**Tamanho do Endereço (Tamanho da Tupla)**

O tamanho do endereço, também conhecido como tamanho da tupla, é um parâmetro fundamental em redes WiSARD e determina a quantidade de bits de entrada que são agrupados para formar um endereço para cada RAM ou filtro de Bloom no discriminador. 

No contexto do BTHOWeN:
- Define quantos bits são agrupados para endereçar cada filtro de Bloom
- Afeta o número total de filtros necessários (número total de entradas dividido pelo tamanho da tupla)
- Determina o espaço de endereçamento para cada filtro (2^tamanho da tupla possíveis endereços em WiSARD tradicional)

Tuplas menores resultam em mais filtros e melhor generalização, enquanto tuplas maiores reduzem o número de filtros, mas podem afetar a capacidade de generalização. Nos experimentos, este parâmetro varia conforme o dataset, desde valores pequenos (2 para Iris) até valores maiores (28 para MNIST).

**Número de Discriminadores (OWeN)**

O número de discriminadores no BTHOWeN está diretamente relacionado à estrutura de entrada do modelo. Cada discriminador é responsável por processar uma parte específica do vetor de entrada, através da segmentação dos dados de entrada em subconjuntos de bits. 

Concretamente, o número de discriminadores para cada classe é determinado pela divisão da dimensão total da entrada pelo número de bits por filtro. Por exemplo, em um modelo para MNIST com entrada de 784 pixels (28x28) e 28 bits por filtro, teríamos 28 discriminadores por classe.

O valor do parâmetro OWeN representa o número de bits de entrada que cada discriminador recebe. Quanto maior este valor, mais granular é a segmentação dos dados, o que pode melhorar a capacidade de distinção, mas aumenta a complexidade computacional.

**Número de Funções Hash (FH)**

As funções hash são utilizadas nos filtros de Bloom para mapear os dados de entrada em posições de memória. O parâmetro FH determina quantas diferentes funções hash são utilizadas em cada filtro de Bloom.

BTHOWeN implementa a família de funções hash H3, conforme descrita por Carter e Wegman, que não requer operações aritméticas complexas, sendo ideal para implementações em hardware.

O número de funções hash tem uma relação complexa com a taxa de falsos positivos:
- Valores típicos variam de 1 a 4, dependendo do dataset
- Um maior número de funções hash pode aumentar o poder discriminativo do modelo, mas também eleva o custo computacional
- Os melhores resultados obtidos nos experimentos utilizaram de 1 a 4 funções hash, dependendo da complexidade da tarefa

**Número de Filtros de Bloom (FE - Filter Entries)**

O parâmetro FE define o número de entradas em cada filtro de Bloom, ou seja, o tamanho do array subjacente utilizado para armazenar os padrões aprendidos. Este valor deve ser uma potência de 2 (por exemplo, 128, 256, 512, 1024, 2048, etc.).

Aumentar o tamanho do filtro reduz a probabilidade de colisões e, consequentemente, a taxa de falsos positivos, mas também aumenta a memória necessária. A escolha deste parâmetro afeta diretamente o equilíbrio entre precisão e eficiência de memória do modelo.

Em hardware, este parâmetro se traduz na quantidade de memória alocada para cada filtro, influenciando o tamanho total do chip e o consumo de energia.

**Fator de Bleaching (b)**

O fator de bleaching é uma das principais inovações do BTHOWeN em relação a outras redes neurais sem peso. Em implementações tradicionais de filtros de Bloom, cada posição de memória armazena apenas um bit (0 ou 1), indicando se um padrão foi visto ou não. 

No BTHOWeN, cada posição armazena um contador, permitindo registrar quantas vezes um determinado padrão foi encontrado durante o treinamento. O fator de bleaching define o limiar mínimo de ocorrências para que um padrão seja considerado válido durante a inferência.

Funcionamento:
- Durante o treinamento, os contadores são incrementados cada vez que um padrão é encontrado
- Na inferência, um padrão é considerado presente apenas se seu contador for ≥ b
- Aumentar o valor de b pode melhorar a precisão ao reduzir falsos positivos
- Valores comuns de b variam de 1 a 10, sendo o valor ótimo determinado durante o treinamento

A técnica de bleaching permite ao modelo distinguir entre padrões frequentes (relevantes) e ocorrências aleatórias (ruído), melhorando significativamente a generalização e robustez do modelo.

**c. Datasets**

* **MNIST**

O dataset MNIST (Modified National Institute of Standards and Technology) é uma coleção de dígitos manuscritos. Ele inclui 60k imagens de treinamento e 10k imagens de teste, todas em escala de cinza e com tamanho de 28×28 pixels.

* **Ecoli**

O dataset Ecoli é usado para prever onde proteínas celulares se localizam com base em suas sequências de aminoácidos. Contém dados de 336 proteínas, cada uma descrita por sete atributos numéricos derivados da sequência. As proteínas são classificadas em oito possíveis locais celulares.

* **Iris**

O dataset Iris contém 150 observações de flores de íris, cada uma descrita por quatro características: comprimento da sépala, largura da sépala, comprimento da pétala e largura da pétala. A classificação é feita em uma de três espécies: Iris Setosa, Versicolor ou Virginica.

* **Glass**

O dataset Glass contém 214 instâncias de fragmentos de vidro, cada uma descrita por 10 atributos. Com ele conseguimos prever o tipo de vidro com base em sua composição química e índice de refração.

* **Letter**

É um dataset de letras manuscritas. As imagens dos caracteres foram baseadas em 20 fontes diferentes, e cada letra dentro dessas 20 fontes foi distorcida aleatoriamente para produzir um arquivo de 20k entradas, onde 16k foram usadas para treinamento e 4k pra teste.

* **Wine**

O dataset Wine reúne 178 amostras de vinho produzidas na região de Piemonte, Itália. Cada amostra é caracterizada por 13 medidas físico-químicas obtidas por espectrometria, como teor de álcool, magnésio e intensidade de cor. O objetivo é identificar a cepa da uva usada, escolhendo entre três cultivares.

* **Segment**

O dataset Segment consiste em regiões recortadas de imagens coloridas, totalizando 2310 exemplos. Cada região é descrita por 19 atributos que capturam cor, textura e forma. A tarefa é reconhecer o tipo de superfície ou objeto entre sete categorias, como céu, grama ou tijolo.

* **Shuttle**

O dataset Shuttle foi coletado da telemetria de um ônibus espacial e contém 58k registros de treinamento e 14,5k de teste. Cada registro possui nove atributos numéricos correspondentes a leituras de sensores a bordo. O objetivo é classificar o estado operacional em uma de sete classes.

* **Vehicle**

O dataset Vehicle apresenta 846 silhuetas de veículos vistas de diferentes ângulos. Cada silhueta é representada por 18 atributos derivados de momentos de contorno: circularidade, assimetria, curtose, entre outros. A missão é distinguir entre quatro tipos de automóvel: bus, opel, saab e van.

* **Vowel**

O dataset Vowel contém 990 gravações de vogais pronunciadas por 15 falantes do inglês britânico. Cada gravação é descrita por dez coeficientes acústicos extraídos do espectro, mais identificadores de locutor e gênero. A categoria alvo corresponde a uma de onze vogais da língua inglesa.

* **MNIST**

O dataset Vowel contém 990 gravações de vogais pronunciadas por 15 falantes do inglês britânico. Cada gravação é descrita por dez coeficientes acústicos extraídos do espectro, mais identificadores de locutor e gênero. A categoria alvo corresponde a uma de onze vogais da língua inglesa.

3. **Resultados**  
   1. **Resultados por dataset**  
      1. **Tabelas**  
           
           
         

| Dataset | Config Type | b | OWeN | FE | FH | Acurácia | Empates (%) | Melhor Bleaching | Execução |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Iris | Referência Bloom WiSARD | N/A | N/A | N/A | N/A | 960 | 2.0 | 3 | 1 |
| Iris | BTHOWeN Base Estudo | 3 | 2 | 128 | 1 | 980 | 12.0 | 2 | 3 |
| Iris | BTHOWeN Variação 1 | 4 | 2 | 128 | 1 | 920 | 16.0 | 1 | 1 |
| Iris | BTHOWeN Variação 2 | 3 | 2 | 256 | 1 | 980 | 14.0 | 2 | 1 |
| Iris | BTHOWeN Variação 3 | 3 | 2 | 128 | 2 | 980 | 8.0 | 2 | 1 |
| Iris | BTHOWeN Variação 4 | 4 | 2 | 128 | 2 | 860 | 4.0 | 9 | 1 |
| Iris | BTHOWeN Variação 5 | 3 | 2 | 256 | 2 | 980 | 0.0 | 2 | 1 |
| Iris | BTHOWeN Variação 6 | 4 | 2 | 256 | 2 | 900 | 12.0 | 3 | 1 |

         

         

         

| Dataset | Config Type | b | OWeN | FE | FH | Acurácia | Empates (%) | Melhor Bleaching | Execução |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Ecoli | Referência Bloom WiSARD | N/A | N/A | N/A | N/A | 799 | N/A | N/A | \- |
| Ecoli | BTHOWeN Base Estudo | 10 | 128 | 2 | 10 | 786 | 8.9 | 7 | 1 |
| Ecoli | BTHOWeN Variação 1 | 4 | 128 | 2 | 11 | 821 | 10.7 | 1 | 1 |
| Ecoli | BTHOWeN Variação 2 | 3 | 256 | 2 | 10 | 813 | 19.6 | 1 | 1 |
| Ecoli | BTHOWeN Variação 3 | 3 | 128 | 3 | 10 | 786 | 15.2 | 7 | 1 |
| Ecoli | BTHOWeN Variação 4 | 4 | 128 | 3 | 11 | 839 | 17.9 | 1 | 1 |
| Ecoli | BTHOWeN Variação 5 | 3 | 256 | 3 | 10 | 848 | 10.7 | 1 | 1 |
| Ecoli | BTHOWeN Variação 6 | 4 | 256 | 4 | 10 | 830 | 13.4 | 1 | 1 |

         Tabela 1: Parâmetros e métricas do dataset Ecoli

         

         

         

| Dataset | Config Type | b | OWeN | FE | FH | Acurácia | Empates (%) | Melhor Bleaching | Execução |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Glass | Referência Bloom WiSARD | N/A | N/A | N/A | N/A | 726 | N/A | N/A | \- |
| Glass | BTHOWeN Base Estudo | 3 | 128 | 3 | 9 | 577 | 39.4 | 1 | 1 |
| Glass | BTHOWeN Variação 1 | 4 | 128 | 3 | 10 | 563 | 38.0 | 1 | 1 |
| Glass | BTHOWeN Variação 2 | 3 | 256 | 3 | 9 | 493 | 33.8 | 1 | 1 |
| Glass | BTHOWeN Variação 3 | 3 | 128 | 4 | 9 | 549 | 19.7 | 4 | 1 |
| Glass | BTHOWeN Variação 4 | 4 | 128 | 4 | 10 | 592 | 40.8 | 1 | 1 |
| Glass | BTHOWeN Variação 5 | 3 | 256 | 4 | 9 | 676 | 29.6 | 1 | 1 |
| Glass | BTHOWeN Variação 6 | 4 | 256 | 4 | 10 | 676 | 28.2 | 1 | 1 |

         

         

         

| Dataset | Config Type | b | OWeN | FE | FH | Acurácia | Empates (%) | Melhor Bleaching | Execução |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Letter | Referência Bloom WiSARD | N/A | N/A | N/A | N/A | 848 | N/A | N/A | \- |
| Letter | BTHOWeN Base Estudo | 3 | 128 | 3 | 9 | 734 | 18.6 | 6 | 1 |
| Letter | BTHOWeN Variação 1 | 4 | 128 | 3 | 10 | 736 | 17.4 | 5 | 1 |
| Letter | BTHOWeN Variação 2 | 3 | 256 | 3 | 9 | 789 | 15.2 | 3 | 1 |
| Letter | BTHOWeN Variação 3 | 3 | 128 | 4 | 9 | 707 | 19.2 | 5 | 1 |
| Letter | BTHOWeN Variação 4 | 4 | 128 | 4 | 10 | 719 | 18.7 | 6 | 1 |
| Letter | BTHOWeN Variação 5 | 3 | 256 | 4 | 9 | 775 | 15.6 | 4 | 1 |
| Letter | BTHOWeN Variação 6 | 4 | 256 | 5 | 12 | 811 | 11.3 | 4 | 1 |
| Letter | BTHOWeN Variação 7 | 11 | 256 | 5 | 18 | 840 | 7.6 | 3 | 1 |
| Letter | BTHOWeN Variação 8 | 15 | 256 | 5 | 35 | 884 | 3.9 | 3 | 1 |

         

         

         

| Dataset | Config Type | b | OWeN | FE | FH | Acurácia | Empates (%) | Melhor Bleaching | Execução |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Wine | Referência Bloom WiSARD | N/A | N/A | N/A | N/A | 926 | N/A | N/A | \- |
| Wine | Referência BTHOWeN | 9 | 13 | 128 | 3 | 983 | N/A | N/A | \- |
| Wine | BTHOWeN Base Estudo | 9 | 13 | 128 | 3 | 983 | N/A | 1 | \- |
| Wine | BTHOWeN Variação 1.1 | 10 | 13 | 256 | 4 | 1.000 | 1.7 | 1 | \- |
| Wine | BTHOWeN Variação 1.2 | 10 | 13 | 256 | 4 | 949 | 3.39 | 1 | \- |
| Wine | BTHOWeN Variação 2 | 11 | 9 | 256 | 4 | 983 | 1.69 | 1 | \- |
| Wine | BTHOWeN Variação 3 | 11 | 13 | 256 | 2 | 966 | 3.38 | 1 | \- |
| Wine | BTHOWeN Variação 4 | 9 | 9 | 256 | 4 | 966 | 1.69 | 1 | \- |
| Wine | BTHOWeN Variação 5 | 10 | 17 | 256 | 4 | 966 | 11.8 | 1 | \- |
| Wine | BTHOWeN Variação 6 | 11 | 15 | 256 | 2 | 966 | N/A | N/A | \- |
| Wine | BTHOWeN Variação 7 | 7 | 9 | 256 | 4 | 932 | 5.08 | 1 | \- |

         

         

         

         

| Dataset | Config Type | b | OWeN | FE | FH | Acurácia | Empates (%) | Melhor Bleaching | Execução |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Segment | Referência Bloom WiSARD | N/A | N/A | N/A | N/A | N/A | N/A | N/A | \- |
| Segment | Referência BTHOWeN | 8 | 12 | 512 | 4 | 880 | N/A | \- | \- |
| Segment | BTHOWeN Base Estudo | 9 | 27 | 1024 | 2 | 925 | N/A | 1 | \- |
| Segment | BTHOWeN Variação 1 | 10 | 16 | 256 | 4 | 938 | 9.35 | 1 | \- |
| Segment | BTHOWeN Variação 2 | 8 | 20 | 512 | 3 | 924 | 9.24 | 1 | \- |
| Segment | BTHOWeN Variação 3.1 | 10 | 18 | 1024 | 3 | 942 | 8.57 | 2 | \- |
| Segment | BTHOWeN Variação 3.2 | 10 | 18 | 1024 | 3 | 944 | 6.49 | 2 | \- |
| Segment | BTHOWeN Variação 4 | 10 | 20 | 512 | 2 | 944 | 8.96 | 1 | \- |
| Segment | BTHOWeN Variação 5 | 16 | 16 | 256 | 3 | 941 | 10.8 | 8 | \- |
| Segment | BTHOWeN Variação 6 | 10 | 14 | 512 | 4 | 939 | 8.10 | 2 | \- |
| Segment | BTHOWeN Variação 7 | 9 | 20 | 1024 | 2 | 937 | 23.7 | 1 | \- |
| Segment | BTHOWeN Variação 8 | 15 | 15 | 256 | 4 | 936 | 9.8 | 1 | \- |
| Segment | BTHOWeN Variação 9 | 9 | 32 | 2048 | 4 | 936 | 34.9 | 1 | \- |

         

         

         

| Dataset | Config Type | b | OWeN | FE | FH | Acurácia | Empates (%) | Melhor Bleaching | Execução |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Shuttle | Referência Bloom WiSARD | N/A | N/A | N/A | N/A | 868 | N/A | N/A | \- |
| Shuttle | Referência BTHOWeN README | 9 | 27 | 1024 | 2 | 868 | N/A | N/A | \- |
| Shuttle | BTHOWeN Base Estudo | 9 | 27 | 1024 | 2 | 0 | 0 | 0 | 0 |
| Shuttle | BTHOWeN Variação 1.1 | 11 | 29 | 1024 | 2 | 999 | 0.11 | 1 | \- |
| Shuttle | BTHOWeN Variação 1.2 | 11 | 29 | 1024 | 2 | 998 | 0.17 | 1 | \- |
| Shuttle | BTHOWeN Variação 2 | 11 | 25 | 1024 | 3 | 999 | 0.10 | 1 | \- |
| Shuttle | BTHOWeN Variação 3 | 8 | 27 | 1024 | 1 | 998 | 0.21 | 4 | \- |
| Shuttle | BTHOWeN Variação 4 | 9 | 23 | 512 | 3 | 998 | 0.21 | 8 | \- |
| Shuttle | BTHOWeN Variação 5 | 8 | 23 | 2048 | 1 | 998 | 0.70 | 1 | \- |
| Shuttle | BTHOWeN Variação 6 | 7 | 27 | 1024 | 2 | 989 | 2.55 | 5 | \- |
| Shuttle | BTHOWeN Variação 8 | 11 | 27 | 1024 | 2 | 976 | 4.99 | 276 | \- |

         

         

         

| Dataset | Config Type | b | OWeN | FE | FH | Acurácia | Empates (%) | Melhor Bleaching | Execução |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Wine | Referência Bloom WiSARD | N/A | N/A | N/A | N/A | 926 | N/A | N/A | \- |
| Vehicle | Referência BTHOWeN README | 16 | 16 | 256 | 3 | 662 | N/A | N/A | \- |
| Vehicle | BTHOWeN Base Estudo | 16 | 16 | 256 | 3 | N/A | N/A | N/A | N/A |
| Vehicle | BTHOWeN Variação 1 | 14 | 14 | 512 | 4 | 755 | 32.1 | 1 | \- |
| Vehicle | BTHOWeN Variação 11 | 15 | 12 | 256 | 2 | 755 | 20.2 | 1 | \- |
| Vehicle | BTHOWeN Variação 19 | 18 | 16 | 512 | 2 | 748 | 30.3 | 1 | \- |
| Vehicle | BTHOWeN Variação 9 | 18 | 18 | 512 | 3 | 737 | 25.5 | 1 | \- |
| Vehicle | BTHOWeN Variação 18 | 16 | 14 | 512 | 2 | 734 | 20.2 | 1 | \- |
| Vehicle | BTHOWeN Variação 10 | 15 | 16 | 512 | 3 | 726 | 32.8 | 1 | \- |
| Vehicle | BTHOWeN Variação 15 | 14 | 12 | 512 | 3 | 726 | 27.5 | 1 | \- |

         

         

         

| Dataset | Config Type | b | OWeN | FE | FH | Acurácia | Empates (%) | Melhor Bleaching | Execução |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Vowel | Referência Bloom WiSARD | N/A | N/A | N/A | N/A | 876 | N/A | N/A | \- |
| Vowel | Referência BTHOWeN | 15 | 15 | 256 | 4 | 876 | N/A | N/A | \- |
| Vowel | BTHOWeN Base Estudo | 15 | 15 | 256 | 4 | 0 | 0 | 0 | 0 |
| Vowel | BTHOWeN Variação 16 | 15 | 13 | 512 | 5 | 924 | 24.4 | 1 | \- |
| Vowel | BTHOWeN Variação 8 | 16 | 11 | 256 | 3 | 918 | 21.8 | 1 | \- |
| Vowel | BTHOWeN Variação 9 | 16 | 11 | 256 | 5 | 918 | 0 | 0 | \- |
| Vowel | BTHOWeN Variação 19 | 17 | 11 | 512 | 5 | 918 | 23.0 | 1 | \- |
| Vowel | BTHOWeN Variação 18 | 14 | 11 | 128 | 3 | 912 | 28.8 | 1 | \- |
| Vowel | BTHOWeN Variação 17 | 16 | 13 | 512 | 5 | 909 | 0 | 0 | \- |
| Vowel | BTHOWeN Variação 5 | 16 | 17 | 256 | 5 | 906 | 0 | 0 | \- |

         

         

| Dataset | Config Type | b | OWeN | FE | FH | Acurácia | Empates (%) | Melhor Bleaching | Execução | Encoding bits | Dropout |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | ----- | ----- |
| MNIST | ULEEN Base Estudo | 6 | 49 | 8192 | 4 | 952 | N/A | N/A | N/A |  |  |
| MNIST | Referência BTHOWeN | 2 | 28 | 1024 | 2 | 934 | N/A | N/A | N/A |  |  |
| MNIST | BTHOWeN Base Estudo | 2 | 28 | 1024 | 2 | 929 | 1.61 | 8 | \- |  |  |
| MNIST | BTHOWeN Variação 1 | 16 | 16 | 256 | 3 | 915 | N/A | N/A | \- |  |  |
| MNIST | BTHOWeN Variação 2 | 15 | 15 | 256 | 4 | 913 | N/A | N/A | \- |  |  |
| MNIST | BTHOWeN Variação 3 | 9 | 27 | 1024 | 2 | 933 | N/A | N/A | \- |  |  |
| MNIST | BTHOWeN Variação 4 | 4 | 16 | 512 | 2 | 918 | 2.1 | 16 | \- |  |  |
| MNIST | BTHOWeN Variação 5 | 8 | 20 | 512 | 3 | 921 | N/A | N/A | \- |  |  |
| MNIST | BTHOWeN Variação 6 | 4 | 24 | 256 | 2 | 916 | N/A | N/A | \- |  |  |
| MNIST | BTHOWeN Variação 7 | 8 | 32 | 2048 | 4 | 943 | 0.36 | 6 | \- |  |  |

         

         

         

   2. **Resultados agregados**

…..

4. **Conclusão**

….

5. **Referências**

\[1\] Zachary Susskind, Aman Arora, Igor D. S. Miranda, Luis A. Q. Villon, Rafael F. Katopodis, Leandro S. de Araújo, Diego L. C. Dutra, Priscila M. V. Lima, Felipe M. G. França, Mauricio Breternitz Jr., and Lizy K. John. 2022\. Weightless Neural Networks for Efficient Edge Inference. In *PACT ’22: International Conference on Parallel Architectures and Compilation Techniques (PACT), October 10–12, 2022, Chicago, IL*. ACM, New York, NY, USA, 12 pages. https://doi.org/10.1145/3559009.3569680