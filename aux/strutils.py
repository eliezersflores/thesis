import pandas as pd

def df2latex(df, fname):

    text = '\\begin{table}[H]\n'
    text += '\\centering'
    text += df.to_latex(column_format=(df.shape[1] + 1)*'c', escape=False, float_format='{:g}'.format, index=False)
    text += '\\end{table}\n'
    text = text.replace('.', ',')
    with open(fname, 'w') as f:
        f.writelines(text)

def lst2str(lst, q=2):

    # formata cada elemento de lst para ser uma string que apresente q dígitos após o ponto decimal. 

    fst = '{:.' + '{:d}f'.format(q) + '}'
    return [fst.format(el) for el in lst] 

def method2prepstr(cnn):

    cnn = cnn.capitalize()
    cnn = cnn.replace('121', '-121')
    cnn = cnn.replace('169', '-169')
    cnn = cnn.replace('201', '-201')
    cnn = cnn.replace('50', '-50')
    cnn = cnn.replace('101', '-101')
    cnn = cnn.replace('152', '-152')
    cnn = cnn.replace('16', '-16')
    cnn = cnn.replace('19', '-19')
    cnn = cnn.replace('net', 'Net')
    cnn = cnn.replace('b0', '-B0')
    cnn = cnn.replace('b1', '-B1')
    cnn = cnn.replace('b2', '-B2')
    cnn = cnn.replace('b3', '-B3')
    cnn = cnn.replace('b4', '-B4')
    cnn = cnn.replace('b5', '-B5')
    cnn = cnn.replace('b6', '-B6')
    cnn = cnn.replace('b7', '-B7')
    cnn = cnn.replace('v2', '-v2')
    cnn = cnn.replace('v3', '-v3')
    cnn = cnn.replace('large', '-large')
    cnn = cnn.replace('small', '-small')
    cnn = cnn.replace('mobile', '-mobile')
    cnn = cnn.replace('Inception', 'InceptionNet')
    cnn = cnn.replace('Nas', 'NAS')
    cnn = cnn.replace('res', 'Res')
    cnn = cnn.replace('Vgg', 'VGGNet')
    cnn = cnn.replace('Xception', 'XceptionNet')

    return cnn
