import numpy
import os
from cern_pymad_domain_tfs import TfsTable, TfsSummary
import collections
    
def tfs(inputfile):
    table,params=tfsDict(inputfile)
    return TfsTable(table), TfsSummary(params)

def tfsDict(inputfile):
    '''
    .. py:function:: tfsDict(inputfile)

    Read a tfs table and returns table/summary info
    
    The function takes in a tfs file. It will add
    all parameters into one dictionary, and the table
    into another dictionary.

    :param string inputfile: tfs file, full path
    :raises ValueError: In case file path is not found
    :rtype: tuple containing dictionaries (tfs table , summary)

    See also: :mod:`pymad.domain.tfs`
    '''
#    params={}
    params = collections.OrderedDict()

    if not os.path.isfile(inputfile):
        if os.path.isfile(inputfile+'.tfs'):
            inputfile+='.tfs'
        elif os.path.isfile(inputfile+'.TFS'):
            inputfile+='.TFS'
        else:
            raise ValueError("ERROR: "+inputfile+" is not a valid file path")
    f=file(inputfile,'r')
    l=f.readline()
    while(l):
        if l.strip()[0]=='@':
            _addParameter(params,l)
        if l.strip()[0]=='*': # beginning of vector list...
            names=l.split()[1:]
            table=_read_table(f,names)
        l=f.readline()
    return table, params

##
# Add parameter to object
# 
# Any line starting with an @ is a parameter.
# If that is found, this function should be called and given the line
# 
# @param line The line from the file that should be added

def _addParameter(params,line):
    lname=line.split()[1].lower()
    if line.split()[2]=='%le':
        params[lname]=float(line.split()[3])
    if line.split()[2][-1]=='s':
        params[lname]=line.split('"')[1]
    if line.split()[2]=='%d':
        params[lname]=int(line.split()[3])

##
# Reads in a table in tfs format.
# Input the file stream at the location
# where the names of the columns have just been read.
def _read_table(fstream,names):
    l=fstream.readline()
    types=[]
#    table={}
    table=collections.OrderedDict() 
    for n in names:
        table[n.lower()]=[]
    while(l):
        if l.strip()[0]=='$':
            types=l.split()[1:]
        else:
            for n,el in zip(names,l.split()):
                table[n.lower()].append(el)
        l=fstream.readline()
    for n,typ in zip(names,types):
        if typ=='%le':
            table[n.lower()]=numpy.array(table[n.lower()],dtype=float)
        elif typ=='%d':
            table[n.lower()]=numpy.array(table[n.lower()],dtype=int)
        elif typ=='%s':
            for k in xrange(len(table[n.lower()])):
                table[n.lower()][k]=table[n.lower()][k].split('"')[1]
    return table
