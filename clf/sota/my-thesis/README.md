Esse repositório corresponde à tese de doutorado do André Pacheco (UFES). 
Tal repositório foi disponibilizado no github (https://github.com/paaatcha/my-thesis).
Todas as execuções foram realizadas na máquina Oslo da UFRGS. 

Obs.: por conveniência, o diretório em que serão armazenadas as bases de imagens ('/mnt/EES-Babylon/eliezerflores_doutorado') será referido como '..', de modo que o subdiretório onde ficará o repositório 'my-thesis' será referido como '.'.

-------------------------------------
INSTALAÇÕES E PREPARAÇÃO DO ÂMBIENTE:
-------------------------------------

1) No diretório '..', baixar o repositório principal usando o seguinte comando:

git clone https://github.com/paaatcha/my-thesis.git

2) Também no diretório '..', baixar o respositório MetaBlock, usando o seguinte comando:

git clone https://github.com/paaatcha/MetaBlock.git

3) Dentro de '.', baixar o repositório 'RAUG' com:

git clone https://github.com/paaatcha/raug.git

4) Baixar a imagem Docker com o Pytorch instalado e configurado para ser usado com a GPU:

sudo docker pull nvcr.io/nvidia/pytorch:20.02-py3t 

5) Criar um container Docker com:

sudo docker run --ipc=host --shm-size 8G -it --gpus all -v $(pwd):/workspace/ 0ce1fb47085a bash

Obs.: esse comando foi executado dentro do diretório '..' (0ce1fb47085a é o IMAGE ID da imagem baixada na etapa acima).
Obs.: dentro do container, o diretório '..' ficará em '/workspace' e o diretório '.' em '/workspace/my-thesis'.

6) Instalei as dependencias no container com:

pip install -r requirements

---

Ocorreu o seguinte erro:

ValueError: At least one stride in the given numpy array is negative, and tensors with negative strides are not currently supported. (You can probably work around this by making a copy of your array  with array.copy().)

Para corrigir, substituir

img = torch.from_numpy(pic.transpose((2, 0, 1))).contiguous()

por 

img = torch.from_numpy(pic.transpose((2, 0, 1)).copy()).contiguous()

na linha 114 do arquivo "functional.py" em /usr/local/lib/python3.6/dist-packages/torchvision/transforms
 
RuntimeError: view size is not compatible with input tensor's size and stride (at least one dimension spans across two contiguous subspaces). Use .reshape(...) instead.
 
Alterar a linha 460 do arquivo /workspace/raug/raug/metrics.py de 

correct_k = correct[:k].view(-1).float().sum(0)

para

correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)

---

Obs.: Daqui em diante, todas as execuções foram realizadas no container 593da00ca767.

-------------------------------------------------------------
ORGANIZAÇÃO DOS DADOS PARA O TREINAMENTO COM A BASE ISIC 2019
-------------------------------------------------------------

1) Baixar a base de imagens de treinamento do ISIC, disponibilizada em https://challenge.isic-archive.com/data, no diretório '../ISIC2019_with_macroscopic_data'. 

Obs.: dentro deste diretório deverá ficar o subdiretório 'ISIC_2019_Training_Input', com as imagens em .jpg, bem como os arquivos 'ISIC_2019_Training_Metadata.csv' e 'ISIC_2019_Training_GroundTruth.csv' (o comando unzip pode ser utilizado para realizar a descompactação).

2) Copiar todas as imagens de teste dos seus diretórios originais para o diretório 'ISIC_2019_Training_Input' mencionado acima, no qual as mesmas ficarão juntas com as imagens de treinamento do ISIC 2019. Para isso, em cada diretório que contém imagens de teste, utilizar o seguinte comando:

cp -r *.jpg /workspace/ISIC2019_with_macroscopic_data/ISIC_2019_Training_Input

Obs.: o diretório em questão deve ficar com 25331 + x imagens, onde x é o total de imagens que serão utilizadas nos testes. 

3) Executar o script /workspace/MetaBlock/benchmarks/isic/preprocess/merge_csv.py com

python3 merge_csv.py

para produzir o arquivo ISIC2019.csv.

------------------------------------------------------------
ORGANIZAÇÃO DOS DADOS PARA O TREINAMENTO COM A BASE PAD-UFES
------------------------------------------------------------

1) Baixar a base de imagens PAD-UFES, disponibilizada em https://data.mendeley.com/datasets/zr7vgbcyr2/1, no diretório '../PAD-UFES-20'. 

Obs.: dentro deste diretório podem ser encontrados o subdiretório 'images', com todas as 2298 imagens do dataset (extensão '.png'), bem como o arquivo 'metadata.csv' com os metadados das imagens (cada imagem é representada por meio de 26 colunas, sendo essas as 21 features descritas na página 99 da tese do André mais as colunas 'patient_id', 'lesion_id', 'diagnostic', 'img_id' e 'biopsed'). 

2) Copiar todas as imagens de teste dos seus diretórios originais para o diretório 'images' mencionado acima, no qual as mesmas ficarão juntas com as imagens da base PAD-UFES. Para isso, em cada diretório que contém imagens de teste, utilizar o seguinte comando:

cp -r *.jpg /workspace/PAD-UFES-20/images

Obs.: o diretório em questão deve ficar com 2298 + x imagens, onde x é o total de imagens que serão utilizadas nos testes. 

--------------------------------------------------
TREINAMENTO, VALIDAÇÃO E TESTE COM A BASE PAD-UFES
--------------------------------------------------

Obs.: alguns scripts foram adicionados por mim para viabilizar estas etapas. 

1) Executar o script 

../my-thesis/benchmarks/pad/preprocess/prepare_train_data.py

Obs. as 21 features originais são mapeadas em 81 novas features usando a estratégia "one-hot encoding", conforme mostrado a seguir:

feature original 1:
smoke_False (nova feature 1)
smoke_True (nova feature 2)

feature original 2:
drink_False (nova feature 3)
drink_True (nova feature 4)

feature original 3:
background_father_POMERANIA (nova feature 5)
background_father_GERMANY (nova feature 6)
background_father_BRAZIL (nova feature 7)
background_father_NETHERLANDS (nova feature 8)
background_father_ITALY (nova feature 9)
background_father_POLAND (nova feature 10)
background_father_UNK (nova feature 11)
background_father_PORTUGAL (nova feature 12)
background_father_BRASIL (nova feature 13)
background_father_CZECH (nova feature 14)
background_father_AUSTRIA (nova feature 15)
background_father_SPAIN (nova feature 16)

feature original 4:
background_father_ISRAEL (nova feature 17)
background_mother_POMERANIA (nova feature 18)
background_mother_ITALY (nova feature 19)
background_mother_GERMANY (nova feature 20)
background_mother_BRAZIL (nova feature 21)
background_mother_UNK (nova feature 22)
background_mother_POLAND (nova feature 23)
background_mother_NORWAY (nova feature 24)
background_mother_PORTUGAL (nova feature 25)
background_mother_NETHERLANDS (nova feature 26)
background_mother_FRANCE (nova feature 27)
background_mother_SPAIN (nova feature 28)

feature original 5:
age (nova feature 29)

feature original 6:
pesticide_False (nova feature 30)
pesticide_True (nova feature 31)

feature original 7:
gender_FEMALE (nova feature 32)
gender_MALE (nova feature 33)

feature original 8:
skin_cancer_history_True (nova feature 34)
skin_cancer_history_False (nova feature 35)

feature original 9:
cancer_history_True (nova feature 36)
cancer_history_False (nova feature 37)

feature original 10:
has_piped_water_True (nova feature 38)
has_piped_water_False (nova feature 39)

feature original 11:
has_sewage_system_True (nova feature 40)
has_sewage_system_False (nova feature 41)

feature original 12:
fitspatrick_3.0 (nova feature 42)
fitspatrick_1.0 (nova feature 43)
fitspatrick_2.0 (nova feature 44)
fitspatrick_4.0 (nova feature 45)
fitspatrick_5.0 (nova feature 46)
fitspatrick_6.0 (nova feature 47)

feature original 13:
region_ARM (nova feature 48)
region_NECK (nova feature 49)
region_FACE (nova feature 50)
region_HAND (nova feature 51)
region_FOREARM (nova feature 52)
region_CHEST (nova feature 53)
region_NOSE (nova feature 54)
region_THIGH (nova feature 55)
region_SCALP (nova feature 56)
region_EAR (nova feature 57)
region_BACK (nova feature 58)
region_FOOT (nova feature 59)
region_ABDOMEN (nova feature 60)
region_LIP (nova feature 61)

feature original 14:
diameter_1 (nova feature 62)

feature original 15:
diameter_2 (nova feature 63)

feature original 16:
itch_False (nova feature 64)
itch_True (nova feature 65)
itch_UNK (nova feature 66)

feature original 17:
grew_False (nova feature 67)
grew_True (nova feature 68)
grew_UNK (nova feature 69)

feature original 18:
hurt_False (nova feature 70)
hurt_True (nova feature 71)
hurt_UNK (nova feature 72)

feature original 19:
changed_False (nova feature 73)
changed_True (nova feature 74)
changed_UNK (nova feature 75)

feature original 20:
bleed_False (nova feature 76)
bleed_True (nova feature 77)
bleed_UNK (nova feature 78)

feature original 21:
elevation_False (nova feature 79)
elevation_True (nova feature 80)
elevation_UNK (nova feature 81)

Note que cada imagem é de fato representada por meio de 88 colunas, as quais correspondem às 81 features descritas acima mais as colunas 'img_id', 'diagnostic', 'patient_id', 'lesion_id', 'biopsed' (que já existiam) e as colunas 'folder' e 'diagnostic_number' (adicionadas pelo script).

2) Executar o script 

../my-thesis/benchmarks/pad/preprocess/prepare_test_data.py

3) Executar o script 

../my-thesis/benchmarks/pad_bench.py

4) Executar o script ../my-thesis/LewDir/ensemble_script.py

Obs.: antes de executar, verificar se o script em questão está configurado para a base PAD-UFES

---------------------------------------------------
TREINAMENTO, VALIDAÇÃO E TESTE COM A BASE ISIC 2019
---------------------------------------------------

Os códigos estão lá, mas seria interessante organizar da mesma forma com que no caso da base PAD-UFES.
