# Desafio do Kaggle: Digit Recognizer

## Sobre o desafio

O desafio Digit Recognizer consiste em classificar dígitos manuscritos da base MNIST.

A competição disponibiliza um conjunto de treinamento com 42 mil imagens rotuladas e um conjunto de teste com 28 mil imagens não rotuladas, que deve ser utilizado para realizar a submissão.

A métrica de avalição é a acurácia, ou seja, a porcentagem de imagens do conjunto de teste classificadas corretamente.

## Sobre os dados disponibilizados pela competição

É necessário participar da competição para poder baixar o dataset. De acordo com os termos de uso, não posso disponibibilizá-los aqui. Além dos dados disponibilizados pela competição, também utilizei a base MNSIT disponibilizada pela biblioteca torchvision. Para esse último caso, é necessário apenas ter a biblioteca instalada.

Link da competição: https://www.kaggle.com/competitions/digit-recognizer

## Ideia de solução

Acredito que a melhor abordagem para essa competição é a utilização de redes neurais convolucionais (CNNs), que são consideradas estado da arte para tarefas de visão computacional.

As camadas convolucionais são utilizadas para extrair as características mais relevantes de cada imagem e, em seguida, os dados são passados para uma rede Multi Layer Perceptron (MLP) para realizar a classificação com base nas características extraídas.

## Arquitetura utilizada

As imagens da base MNIST não são complexas, então, com uma arquitetura relativamente simples, é possível obter um bom resultado. Optei por utilizar duas camadas convolucionais, seguidas de três camadas fully connected. Além disso, utilizei dropout nas duas primeiras camadas fully connected para evitar overfitting. Para melhorar a convergência do treinamento, utilizei a inicialização de He nas camadas convolucionais e fully connected.

Resumo detalhado da arquitetura com torchsummary:

    Layer (type)         Output Shape              Param

    Conv2d-1             [-1, 32, 26, 26]          320
    ReLU-2               [-1, 32, 26, 26]          0
    Conv2d-3             [-1, 64, 24, 24]          18,496
    ReLU-4               [-1, 64, 24, 24]          0
    MaxPool2d-5          [-1, 64, 12, 12]          0
    Linear-6             [-1, 64]                  589,888
    ReLU-7               [-1, 64]                  0
    Dropout-8            [-1, 64]                  0
    Linear-9             [-1, 32]                  2,080
    ReLU-10              [-1, 32]                  0
    Dropout-11           [-1, 32]                  0
    Linear-12            [-1, 10]                  330

    Total params: 611,114
    Trainable params: 611,114
    Non-trainable params: 0

## Treinamento

Utilizei o otimizador Adam para ajuste dos pesos com 0.001 de learning rate. A função de perda utilizada foi a CrossEntropyLoss, que aplica internamente a função softmax, sendo esse o motivo de eu não ter aplicado a softmax na última camada da rede MLP. Além disso, utilizei 40 épocas de treinamento, com batch size de 64 e shuffle=True no DataLoader.

## Resultados e discussão

### Primeira submissão

Na primeira submissão utilizei uma arquitetura mais simples do que a arquitetura final discutida mais acima. Não tinha dropout e não tinha inicialização dos pesos. Além disso, utilizei o conjunto de dados da competição com 42 mil imagens sem uso de data augmentation. O resultado foi bom, mas não excelente.

A acurácia obtida foi 98.864%.

### Submissão final

As imagens do conjunto de teste da competição não são tão comportadas como as imagens de treino, os números possuem leves rotações e translações, o que acaba dificultando a generalização do modelo. Logo, a utilização de data augmentation é muito importante para obter uma acurácia maior.

Utilizei o conjunto de dados de 60 mil imagens da base MNIST disponibilizado pela torchvision. Além disso, utilizei data augmentation, aplicando rotações aleatórias dentro do intervalo de -10 a 10° e translações horizontais e verticais.

Além disso, aprimorei o modelo CNN, aplicando dropout e inicialização dos pesos com He, como foi discutido anteriormente.

Aplicando essas estratégias, foi possível chegar a 99.4% de acurácia. Ainda dá para melhorar, porém, a melhoria não seria tão significativa.