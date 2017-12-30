import uncertainties.unumpy as unp
import numpy as np
from test_dim import test_dim


        


class floatFormat(object):
    def __init__(self, Number, SI="", format="", SI2=False):
        self.u=Number
        self.SI = SI
        self.p = format
        self.SI2 = SI2

    def __format__(self, format):
        temp3 = ''
        temp4 = ''
        temp5 = ''
        if(self.p==""):
            temp = (r'{:'+format+r'}').format(float(self.u))
        else:
            temp = (r'{:'+self.p+r'}').format(float(self.u))
        if self.SI2:
            temp4 = r'\SI{'
            temp5 = r'}'
            if self.SI!="":
                temp3 = r'{'+self.SI+r'}'
        else:
            if self.SI!="":
                temp3 = r'\,\si{'+self.SI+r'}'
        return temp4+temp+temp5+temp3

class unpFormat(object):
    def __init__(self, unpNumber, SI="", format="", SI2=False):
        self.u=unpNumber
        self.SI = SI
        self.p = format
        self.SI2 = SI2

    def __format__(self, format):
        temp3 = ''
        temp4 = ''
        temp5 = ''
        if(self.p==""):
            e=0
            if(unp.std_devs(self.u)==0):
                e=0
            else:
                e=np.log10(float(unp.std_devs(self.u)))
        
            if(e<0):
                p=-e+0.5
            else:
                p=0
            temp1 = (r'{:0.'+(r'{:1.0f}'.format(float(p)))+r'f}').format(float(unp.nominal_values(self.u)))
            temp2 = (r'\pm{:0.'+(r'{:1.0f}'.format(float(p)))+r'f}').format(float(unp.std_devs(self.u)))
        else:
            temp1 = (r'{:'+self.p+r'}').format(float(unp.nominal_values(self.u)))
            temp2 = (r'\pm{:'+self.p+r'}').format(float(unp.std_devs(self.u)))
        if(unp.std_devs(self.u)==0):
            temp2=''
        if self.SI2:
            temp4 = r'\SI{'
            temp5 = r'}'
            if self.SI!="":
                temp3 = r'{'+self.SI+r'}'
        else:
            if self.SI!="":
                temp3 = r'\,\si{'+self.SI+r'}'
        return temp4+temp1+temp2+temp5+temp3
      

class strFormat(object):
    def __init__(self, string):
        self.s=string

    def __format__(self, format):
        return (r'{}').format(self.s)


def convert(data, format1=floatFormat, arguments=[]):
    convertedData=[]
    i=0
    if(test_dim(data)>=1):
        for x in data:
            convertedData.append(convert(x, format1, arguments))
        return convertedData
    else:
        for x in data:
            if test_dim(arguments)==1:
                convertedData.append(format1(x,*arguments))
            else:
                if test_dim(arguments)==2:
                    convertedData.append(format1(x,*arguments[i]))
                else:
                    convertedData.append(format1(x))
            i=i+1
        return convertedData