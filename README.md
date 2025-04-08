## Equipe: Elias da Costa Rodrigues

## Requisitos para rodar o código

- Python 3+ 
- bibliotecas torch, torchvision e pandas
- dataset da competição

## Dataset da competição

De acordo com os termos de uso da competição, não posso disponibilizar o dataset da competição aqui. Para ter acesso ao dataset, é necessário participar da competição e baixar diretamente pelo Kaggle. Link da competição: https://www.kaggle.com/competitions/digit-recognizer

O código espera os dados na forma: ./dados_competicao/train.csv e ./dados_competicao/test.csv.

## Qual dataset utilizar

O melhor resultado que obtive, aplicando data augmentation, foi com o dataset disponibilizado pela torchvision. O trainloader que utiliza esse conjunto de dados está nomeado como 'trainloader_MNIST'.

O trainloader que utiliza os dados disponibilizados pela competição está nomeado como 'trainloader'. Para utilizar um ou outro basta passar como argumento para a função train(model, trainloader, epochs).

O 'trainloader_MNIST' é mais rico, possui mais imagens e utiliza data augmentation. Com certeza o melhor resultado será obtido com ele.

## Rodando o código

Para rodar o código basta executar main.py. Serão exibidas acurácias no terminal, sendo elas a partir do próprio conjunto de treino e a partir do conjunto de teste de 10 mil imagens disponibilizado pela torchvision. No fim, o arquivo de submissão será criado, rotulando as 28 mil imagens de teste que o desafio pede.