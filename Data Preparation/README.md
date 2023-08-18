A fim de facilitar o uso dos métodos, deve-se assegurar que as bases de imagens macroscópicas estejam organizadas de maneira padronizada. Nos experimentos da tese de doutorado, cada base de imagens está em um diretório raiz contendo os subdiretórios "melanoma" e "notmelanoma". Dentro desses subdiretórios, as imagens originais possuem extensão ".jpg" e os ground-truths, se existirem, estão em arquivos com os mesmos nomes das respectivas imagens originais, mas com extensão ".png".

A base DermNet foi enviada para mim por e-mail pelo professor Pablo Cavalcanti (https://scholar.google.com.br/citations?user=4M8N9jcAAAAJ&hl=pt-BR&oi=ao). Os ground truths dessa base estão todos em uma pasta separada, possuem extensão ".tif" (e não ".png") e, em alguns casos, não têm as mesmas dimensões da imagem correspondente. Para produzir uma pasta contendo essa base de imagens organizada da mesma forma que as demais, basta utilizar o script "prepare_dermnet.py".

As bases DermIS e DermQuest podem ser baixadas em: https://uwaterloo.ca/vision-image-processing-lab/research-demos/skin-cancer-detection
As imagens e os ground-truths dessas bases estão misturados dentro de pastas intituladas "Skin Image Data Set-1" e "Skin Image Data Set-2", as quais contêm, respectivamente, as imagens de "melanoma" e "notmelanoma" de ambas as bases. Além disso, os ground-truths não estão em arquivos com os mesmos nomes das respectivas imagens originais (nomes de imagens terminam com "_orig.jpg" e nomes de ground-truths terminam com "_contour.png"). Para produzir uma pasta contendo uma dessas bases de imagens (DermIS ou DermQuest) organizada da mesma forma que as demais, basta utilizar o script "prepare_dermis_dermquest.py".

A base MedNode pode ser baixada em http://www.cs.rug.nl/~imaging/databases/melanoma_naevi/index.html
Essa base não possui ground-truths (as imagens contêm somente a região de interesse, previamente selecionada) e para organizá-la de maneira padronizada basta renomear manualmente os diretórios originais. 

A base PAD-UFES pode ser baixada em https://data.mendeley.com/datasets/zr7vgbcyr2/1
As imagens dessa base estão particionados em três pastas intituladas "imgs_part_1", "imgs_part_2" e "imgs_parts_3". Essa base também não possui ground-truths e, para organizá-la de maneira padronizada, basta juntar todas as imagens dentro de uma mesma pasta e, em seguida, executar o script
"prepare_padufes.py".

A base Derm7pt pode ser baixada em https://derm.cs.sfu.ca/Download.html
Essa base também não possui ground-truths e, para organizá-la de maneira padronizada, basta executar o script "prepare_derm7pt.py".

As características resultantes dos datasets são apresentadas na tabela abaixo.

+-----------+-----+---------+-----------+-------+
|           | GT  | Imagens | Melanomas | Nevos |
+-----------+-----+---------+-----------+-------+
| DermIS    | Sim | 69      | 43        | 26    |
+-----------+-----+---------+-----------+-------+
| DermQuest | Sim | 137     | 76        | 61    |
+-----------+-----+---------+-----------+-------+
| DermNet   | Sim | 152     | 107       | 45    |
+-----------+-----+---------+-----------+-------+
| MED-NODE  | Não | 170     | 70        | 100   |
+-----------+-----+---------+-----------+-------+
| PAD-UFES  | Não | 112     | 52        | 60    |
+-----------+-----+---------+-----------+-------+
| TOTAL     |     | 640     | 348       | 292   |
+-----------+-----+---------+-----------+-------+

Outros datasets podem ser encontrados em: https://workshop2021.isic-archive.com/

