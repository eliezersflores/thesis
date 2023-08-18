Adaptação da implementação do método ARL em https://github.com/Vipermdl/ARL para testá-lo em imagens macroscópicas. 

O âmbiente Python dado em "requirements.txt" não funcionou bem para mim. Para reproduzir o âmbiente que eu efetivamente utilizei, basta executar:

conda env create -f environment_ARL.yml

O script "predict_macroscopy.py" é uma adaptação do script "predict2017_mel.py", só que para testar o método ARL em uma determinada base de imagens macroscópicas, ao invés de testar no ISIC 2017 (todos os arquivos da base em questão, padronizados como se fosse a base ISIC 2017, ficarão na pasta "tmp"). Para executar tal script com uma determinada base de imagens macroscópicas ('DermNet', 'DermQuest' ou 'MED-NODE'), basta enviar o nome da base subsequentemente ao nome do script. Por exemplo (no Ipython):

%run predict_macroscopy.py DermNet

Os resultados ficarão armazenados na pasta "results".

Obs.: Testes realizados no servidor Oslo. 

-----

1) Criei uma pasta chamada "ARL" e dentro dela, coloquei o arquivo "requirements.txt", de acordo com:

https://github.com/Vipermdl/ARL/blob/master/requirements.txt

2) Também dentro da pasta "ARL", coloquei um arquivo denominado "environment.yml", com o seguinte conteúdo:

name: test-env
dependencies:
  - python>=3.5
  - anaconda
  - pip
  - pip:
    - -r file:requirements.txt
    
3) Criei um ambiente com o comando (https://github.com/kaust-vislab/python-data-science-project):

ENV_PREFIX=$PWD/env conda env create --prefix $ENV_PREFIX --file environment.yml --force

4) Ativei o ambiente em questão:

conda activate $ENV_PREFIX


