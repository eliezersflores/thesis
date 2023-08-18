"script_bench" aplica as mesmas etapas de pós-processamento em todas as segmentações (originais) que estão em um dado diretório padronizado (i.e., com os subdiretórios 'melanoma' e 'notmelanoma'). Os resultados após o pós-processamento ficarão em um novo diretório padronizado, dentro de Postprocessing, intitulado na forma "nomedometodo_nomedabase", de um modo que "nomedometodo" identifica o método que foi utilizado para produzir as segmentações originais, sem espaços, underlines e caracteres especiais, e "nomedabase" identifica a base de imagens a partir da qual as segmentações originais foram produzidas, também sem espaços, underlines e caracteres especiais. 

Obs.: a única exceção é o diretório "dictK3A5_dermis", o qual contém os mesmos resultados que em "dictK3_dermis", porém, sem as etapas de pós-processamento.