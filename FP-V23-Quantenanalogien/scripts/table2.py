﻿import numpy as np
from roundUp import *

def nachKomma(s):
    i=s.find(".")
    if (i+1<len(s)):
        return s[i+1]
    else:
        return 0

def makeTable(data, names, filename, formats=[], formats2=[]):
    TableFile = open('content/tables/'+filename+'.tex', 'w+')
    # TableFile.write( r'\begin{table}'+'\n\t'+r'\centering'+'\n\t'+r'\caption{'+name+r'}'+'\n\t')
    TableFile.write( r'\label{tab:'+filename+'}\n\t'+r'\sisetup{table-format=1.2}'+'\n\t'+r'\begin{tabular}{')
    for i in range(len(data)):
        if formats:
            TableFile.write(formats[i])
        else:
            TableFile.write('S ')
    TableFile.write('}\n\t\t')
    TableFile.write(r'\toprule'+'\n\t\t')

    TableFile.write(names)

    TableFile.write(r' \\'+'\n\t\t')
    TableFile.write(r'\midrule'+'\n\t\t')
    for i in range(len(data[0])):
        for b in range(len(data[0:-1])):
            if not np.isnan(data[b][i]):
                if formats2:
                    if (formats[b].find("@") != -1):
                        TableFile.write(formats2[b] % roundUp(data[b][i],10**(-int(nachKomma(formats2[b])))))
                    else:
                        TableFile.write(formats2[b] % (data[b][i]))
                else:
                    TableFile.write(str(data[b][i]))
            else:
                TableFile.write(r' {-} ')
            TableFile.write(r' & ')
        if not np.isnan(data[-1][i]):
            if formats2:
                if (formats[-1].find("@") != -1):
                    TableFile.write(formats2[-1] % roundUp(data[-1][i],10**(-int(nachKomma(formats2[-1])))))
                else:
                    TableFile.write(formats2[-1] % (data[-1][i]))
            else:
                TableFile.write(str(data[-1][i]))
        else:
            TableFile.write(r' {-} ')
        TableFile.write(r' \\')
        TableFile.write('\n\t\t')


    TableFile.write(r'\bottomrule'+'\n\t')
    TableFile.write(r'\end{tabular}'+'\n')
    # TableFile.write(r'\end{table}')

def makeNewTable(data, names, filename='test', formats=[], formats2=[], formats3=[]):
    TableFile = open('content/tables/'+filename+'.tex', 'w+')
    TableFile.write( r'\label{tab:'+filename+'}\n\t'+r'\sisetup{table-format=1.2}'+'\n\t'+r'\begin{tabular}{')
    for i in range(len(data)):
        if formats:
            TableFile.write(formats[i])
        else:
            TableFile.write('c ')
    TableFile.write('}\n\t\t')
    TableFile.write(r'\toprule'+'\n\t\t')

    TableFile.write(names)

    TableFile.write(r' \\'+'\n\t\t')
    TableFile.write(r'\midrule'+'\n\t\t')
    for i in range(len(data[0])):
        for b in range(len(data[0:-1])):
            if formats2:
                    TableFile.write(formats2[b].format(data[b][i]))
            else:
                if formats3:
                    TableFile.write(formats3[b][i].format(data[b][i]))
                else:
                    TableFile.write('{}'.format(data[b][i]))
            TableFile.write(r' & ')
        if formats2:
                TableFile.write(formats2[-1].format(data[-1][i]))
        else:
            if formats3:
                TableFile.write(formats3[-1][i].format(data[-1][i]))
            else:
                TableFile.write('{}'.format(data[-1][i]))
        TableFile.write(r' \\')
        TableFile.write('\n\t\t')


    TableFile.write(r'\bottomrule'+'\n\t')
    TableFile.write(r'\end{tabular}'+'\n')