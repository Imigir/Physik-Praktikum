from sympy import *
from uncertainties import unumpy as unp
from uncertainties import umath

unumpy = {"sqrt":unp.sqrt,"sin":umath.sin,"cos":unp.cos,"tan":unp.tan,"arcsin":unp.arcsin,"arccos":unp.arccos,"arctan":unp.arctan,"log":unp.log,"exp":unp.exp}

def error_to_tex(f, name='Name', var=None, all_vars=None,err_vars=None):
    from sympy import Symbol, latex
    s = 0
    latex_names = dict()
    var_names = dict()
    
    if all_vars == None:
        all_vars = f.free_symbols
        
    i=0
    if var != None:
        for v in all_vars:
            var_names[Symbol(v.name)]=unp.nominal_values(var[i])
            var_names[Symbol('latex_std_' + v.name)]=unp.std_devs(var[i])
            i+=1
        
    
    if err_vars == None:
        err_vars = all_vars
        
    for v in err_vars:
        err = Symbol('latex_std_' + v.name)
        s += f.diff(v)**2 * err**2
        latex_names[err] = '\\sigma_{' + latex(v) + '}'

    all_vars = s.free_symbols
    f2=lambdify(all_vars, s, unumpy)
    if var != None:
        allvarslist = []
        for v in all_vars:
            allvarslist += [var_names[v]]
        sigma=unp.sqrt(f2(*allvarslist))

    file = open('build/'+name+'.tex', 'w+')
    temp = '\\sigma_{' + name + '}='+latex(sqrt(s), symbol_names=latex_names) + r' = '+'{:>.2}'.format(float(sigma))
    file.write(temp)
    return temp

def scipy_to_unp(f,var):
    return lambdify(var, f, unumpy)




