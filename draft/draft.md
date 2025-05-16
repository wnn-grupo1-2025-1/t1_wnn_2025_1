# Hiperparâmetros do BTHOWeN

Esta seção descreve os principais hiperparâmetros do modelo BTHOWeN (Bleached Thermometer-encoded Hashed-input Optimized Weightless Neural Network) e seu significado para o funcionamento e desempenho do modelo.

## 2.2 Hiperparâmetros

### Número de Discriminadores (OWeN)

O número de discriminadores no BTHOWeN está diretamente relacionado à estrutura de entrada do modelo. Cada discriminador é responsável por processar uma parte específica do vetor de entrada, através da segmentação dos dados de entrada em subconjuntos de bits. 

Concretamente, o número de discriminadores para cada classe é determinado pela divisão da dimensão total da entrada pelo número de bits por filtro. Por exemplo, em um modelo para MNIST com entrada de 784 pixels (28x28) e 28 bits por filtro, teríamos 28 discriminadores por classe.

O valor do parâmetro OWeN representa o número de bits de entrada que cada discriminador recebe. Quanto maior este valor, mais granular é a segmentação dos dados, o que pode melhorar a capacidade de distinção, mas aumenta a complexidade computacional.

### Número de Funções Hash (FH)

As funções hash são utilizadas nos filtros de Bloom para mapear os dados de entrada em posições de memória. O parâmetro FH determina quantas diferentes funções hash são utilizadas em cada filtro de Bloom.

BTHOWeN implementa a família de funções hash H3, conforme descrita por Carter e Wegman, que não requer operações aritméticas complexas, sendo ideal para implementações em hardware.

O número de funções hash tem uma relação complexa com a taxa de falsos positivos:
- Valores típicos variam de 1 a 4, dependendo do dataset
- Um maior número de funções hash pode aumentar o poder discriminativo do modelo, mas também eleva o custo computacional
- Os melhores resultados obtidos nos experimentos utilizaram de 1 a 4 funções hash, dependendo da complexidade da tarefa

### Número de Filtros de Bloom (FE - Filter Entries)

O parâmetro FE define o número de entradas em cada filtro de Bloom, ou seja, o tamanho do array subjacente utilizado para armazenar os padrões aprendidos. Este valor deve ser uma potência de 2 (por exemplo, 128, 256, 512, 1024, 2048, etc.).

Aumentar o tamanho do filtro reduz a probabilidade de colisões e, consequentemente, a taxa de falsos positivos, mas também aumenta a memória necessária. A escolha deste parâmetro afeta diretamente o equilíbrio entre precisão e eficiência de memória do modelo.

Em hardware, este parâmetro se traduz na quantidade de memória alocada para cada filtro, influenciando o tamanho total do chip e o consumo de energia.

### Fator de Bleaching (b)

O fator de bleaching é uma das principais inovações do BTHOWeN em relação a outras redes neurais sem peso. Em implementações tradicionais de filtros de Bloom, cada posição de memória armazena apenas um bit (0 ou 1), indicando se um padrão foi visto ou não. 

No BTHOWeN, cada posição armazena um contador, permitindo registrar quantas vezes um determinado padrão foi encontrado durante o treinamento. O fator de bleaching define o limiar mínimo de ocorrências para que um padrão seja considerado válido durante a inferência.

Funcionamento:
- Durante o treinamento, os contadores são incrementados cada vez que um padrão é encontrado
- Na inferência, um padrão é considerado presente apenas se seu contador for ≥ b
- Aumentar o valor de b pode melhorar a precisão ao reduzir falsos positivos
- Valores comuns de b variam de 1 a 10, sendo o valor ótimo determinado durante o treinamento

A técnica de bleaching permite ao modelo distinguir entre padrões frequentes (relevantes) e ocorrências aleatórias (ruído), melhorando significativamente a generalização e robustez do modelo.

### Interação entre Hiperparâmetros

A configuração ideal destes hiperparâmetros depende do dataset específico e da tarefa de classificação. Devido às interações complexas entre esses parâmetros, uma abordagem de otimização gulosa, variando um parâmetro de cada vez, foi adotada neste trabalho para encontrar as configurações que maximizam a acurácia.

Existem trade-offs importantes a considerar:
- Maior número de funções hash e entradas por filtro → Melhor acurácia, maior consumo de recursos
- Maior fator de bleaching → Maior precisão, mas pode reduzir a capacidade de generalização se for muito alto
- Número ideal de bits por entrada (OWeN) → Varia conforme a natureza e dimensionalidade dos dados

A tabela a seguir mostra exemplos de configurações de hiperparâmetros utilizadas em diferentes datasets no artigo original do BTHOWeN:

| Dataset | b | OWeN | FE | FH | Acurácia |
|---------|---|------|----|----|----------|
| MNIST-Small | 2 | 28 | 1024 | 2 | 93.4% |
| MNIST-Medium | 3 | 28 | 2048 | 2 | 94.3% |
| MNIST-Large | 6 | 49 | 8192 | 4 | 95.2% |
| Iris | 3 | 2 | 128 | 1 | 98.0% |
| Letter | 15 | 20 | 2048 | 4 | 90.0% |
| Vowel | 15 | 15 | 256 | 4 | 90.0% |