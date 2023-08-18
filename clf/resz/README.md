O script 'script_bench.py' redimensiona todas as imagens de todos os datasets (assim como os resultados de segmentação e os ground-truths correspondentes) para diversos tamanhos distintos. Esses redimensionamentos fazem-se necessários para permitir alimentar de maneira adequada as DCNNs disponíveis no Tensorflow/Keras.

Para o redimensionamento, foi utilizada a opção 'lossy' do ImageMagick, a qual consiste em aplicar o método de Lanczos de um modo que, primeiramente, o menor lado fique com o tamanho de destino e, em seguida, os pixels não centrais sejam descartados, fazendo com que a imagem de destino fique com o tamanho especificado e também mantenha o aspect ratio original. 

