# coding: utf-8
import pandas as pd
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import csv
import glob
import datetime
import collections
import time
import subprocess
import os
from scipy import optimize as opt
from scipy import constants as const
from StringIO import StringIO
from matplotlib import rc,rcParams
from matplotlib.patches import Rectangle
import itertools

# simdata
from pandas.tools.plotting import autocorrelation_plot
from pandas.tools.plotting import lag_plot
from pandas.tools.plotting import scatter_matrix

class LHCfill(object):
    # names of timberdata we want to extract
    timbervarFBCTB1    = 'LHC.BCTFR.A6R4.B1:BUNCH_INTENSITY'
    timbervarFBCTB2    = 'LHC.BCTFR.A6R4.B2:BUNCH_INTENSITY'

    timbervarBQMB1L    = 'LHC.BQM.B1:BUNCH_LENGTHS'
    timbervarBQMB1F    = 'LHC.BQM.B1:FILLED_BUCKETS'

    timbervarBQMB2L    = 'LHC.BQM.B2:BUNCH_LENGTHS'
    timbervarBQMB2F    = 'LHC.BQM.B2:FILLED_BUCKETS'

    timbervarBSRTB1H   = 'LHC.BSRT.5R4.B1:FIT_SIGMA_H'
    timbervarBSRTB1V   = 'LHC.BSRT.5R4.B1:FIT_SIGMA_V'
    timbervarBSRTB1GD  = 'LHC.BSRT.5R4.B1:GATE_DELAY'
    timbervarBSRTB1CH  = 'LHC.BSRT.5R4.B1:LSF_H'
    timbervarBSRTB1CV  = 'LHC.BSRT.5R4.B1:LSF_V'

    timbervarBSRTB2H   = 'LHC.BSRT.5L4.B2:FIT_SIGMA_H'
    timbervarBSRTB2V   = 'LHC.BSRT.5L4.B2:FIT_SIGMA_V'
    timbervarBSRTB2GD  = 'LHC.BSRT.5L4.B2:GATE_DELAY'
    timbervarBSRTB2CH  = 'LHC.BSRT.5L4.B2:LSF_H'
    timbervarBSRTB2CV  = 'LHC.BSRT.5L4.B2:LSF_V'

    timbervarLumiAtlas = "ATLAS:LUMI_TOT_INST"
    timbervarLumiAlice = "ALICE:LUMI_TOT_INST"
    timbervarLumiCMS   = "CMS:LUMI_TOT_INST"
    timbervarLumiLHCB  = "LHCB:LUMI_TOT_INST"
    
    timbervarhorbpm    = 'LHC.BOFSU:POSITIONS_H'

    # constants
    protonmass = const.physical_constants['proton mass energy equivalent in MeV'][0]/1000 # GeV
    ionA       = 208.
    ionZ       = 82.
    energy     = 6370
    gamma      = energy * ionZ / 193.7291748489224

    # beta's for the undulators and dipoles for the bsrt light
    betaUndH = [203.,200.]
    betaUndV = [318.,327.]
    betaDipH = [214., 205.]
    betaDipV = [328.,344.]
    
    #--------------------------------------------------------------------------------------------------
    # initialization :
    # ----------------
    # fillnumber                : int
    # basedir                   : string
    # summarydf                 : pandas dataframe
    # bunchlenb1df,bunchlenb1df : pandas dataframe
    # ex1df,ey1df,ex2df,ey2df   : pandas dataframe
    # I1df, I2df                : pandas dataframe
    # lumiatlasdf,lumicmsdf,lumialicedf,lumilhcbdf : pandas dataframe
    # bpmhdf                    : pandas dataframe
    # bpmhmask                  : pandas dataframe
    #--------------------------------------------------------------------------------------------------
    def __init__(self,fillnumber,basedir):
        self.fillnumber = fillnumber
        self.basedir    = basedir
        
        self.summarydf, self.summaryfile     = self.getsummary()
        self.bunchlenb1df, self.bunch2enb1df = self.getbunchlenghts('Fill' + str(self.fillnumber) + 'BLb1',
                                                                    'Fill' + str(self.fillnumber) + 'BLb2')
        
        self.ex1df,self.ey1df,self.ex2df,self.ey2df = self.getemitbsrt('Fill' + str(self.fillnumber) +'emit1',
                                                                       'Fill' + str(self.fillnumber)+'emit2') 
        
        self.I1df,self.I2df                         = self.getFBCT('Fill' + str(self.fillnumber) +'fbct1',
                                                                   'Fill' + str(self.fillnumber)+'fbct2')
        
        self.lumiatlasdf,self.lumicmsdf,self.lumialicedf,self.lumilhcbdf = self.getlumi('Fill' + 
                                                                                        str(self.fillnumber) +
                                                                                        'atlas','Fill' + 
                                                                                        str(self.fillnumber)+
                                                                                        'cms',
                                                                                        'Fill' + 
                                                                                        str(self.fillnumber) +
                                                                                        'alice','Fill' +
                                                                                        str(self.fillnumber)+
                                                                                        'lhcb')
        self.bpmhdf    = self.gethorbpm('Fill' + str(self.fillnumber) +'bpmH')
        self.bpmhmask  = self.gethorbpmmask('bpmhmask' + str(self.fillnumber))
    
    # returns fill summary
    def getsummary(self):
        infn   = "Fill" + str(self.fillnumber) + "Summary"
        outfn  = self.basedir + "/Fill" + str(self.fillnumber) + "Summary.CSV"
        bashcmd1 = "./cern-mdb -M FD -fn " + str(self.fillnumber) + " -N  " + infn + " -F CSV"
        bashcmd2 = "mv " + infn + ".CSV " + outfn
        subprocess.call(bashcmd1,shell=True)
        subprocess.call(bashcmd2,shell=True)
        dfsummary = pd.read_csv(outfn,delimiter=',',header=0)
        return dfsummary, outfn+".CSV"
    
    # def convert to unix time
    def converttimetounix(self,t):
        return time.mktime(datetime.datetime.strptime(t,"%Y-%m-%d %H:%M:%S.%f").timetuple())
    
    # function for adding times in YY-mm-dd HH:MM:SS.fff format
    def addtime(self,intime,deltahour):
        mytime = datetime.datetime.strptime(intime,"%Y-%m-%d %H:%M:%S.%f")
        mytime += datetime.timedelta(hours=deltahour)
        return mytime.strftime("%Y-%m-%d %H:%M:%S.%f")
    
    # returns dict with start stop times of modes in a fill
    def gettimes(self):
        timesodic = collections.OrderedDict()
        if os.path.isfile(self.basedir + "/Fill" + str(self.fillnumber) + "Summary.CSV"):
            startdf = pd.read_csv(self.basedir + "/Fill" + str(self.fillnumber) + "Summary.CSV",
                                  delimiter=',',header=0)
            dc = collections.OrderedDict()
            for name, group in startdf.groupby('Value'):
                starttimes = group['StartTime(UTC_TIME)'].values
                stoptimes  = group['EndTime(UTC_TIME)'].values
                arr = np.array([starttimes,stoptimes])
                dc[name] = np.transpose(arr)
            return dc
        else:
            self.getsummary()
            startdf = pd.read_csv(self.basedir + "/Fill" + str(self.fillnumber) + "Summary.CSV",
                                  delimiter=',',header=0)
            dc = collections.OrderedDict()
            for name, group in startdf.groupby('Value'):
                starttimes = group['StartTime(UTC_TIME)'].values
                stoptimes  = group['EndTime(UTC_TIME)'].values
                arr = np.array([starttimes,stoptimes])
                dc[name] = np.transpose(arr)
            return dc
        
    # returns arrays with the bunch slots
    def getbunchpositions(self,fnb1,fnb2):
        tdc = self.gettimes()
        fileexist = False
        outfn1  = self.basedir + "/" + fnb1 + ".CSV"
        outfn2  = self.basedir + "/" + fnb2 + ".CSV"
        
        if (os.path.isfile(self.basedir + '/' + fnb1 + '.CSV')) and                (os.path.isfile(self.basedir + '/' + fnb1 + '.CSV')):
            fileexist = True
        else:
            stop  = tdc[str(self.fillnumber)][-1][1]
            start = tdc[str(self.fillnumber)][-1][0]

            shcmdBQMFB1 = './cern-ldb -vs ' + self.timbervarBQMB1F + ' -t1 \"' + start + '\" -t2 \"' +                            stop + '\" -N ' + fnb1 + ' -F CSV'
            shcmdBQMFB2 = './cern-ldb -vs ' + self.timbervarBQMB2F + ' -t1 \"' + start + '\" -t2 \"' +                            stop + '\" -N ' + fnb2 + ' -F CSV'
#             print shcmdBQMFB1
            try:
                subprocess.call(shcmdBQMFB1,shell=True)
                subprocess.call(shcmdBQMFB2,shell=True)

                bashcmd1 = "mv " + fnb1 + ".CSV " + outfn1
                bashcmd2 = "mv " + fnb2 + ".CSV " + outfn2

                subprocess.call(bashcmd1,shell=True)
                subprocess.call(bashcmd2,shell=True)

                fileexist = True
            except:
                print 'Loading of data failed.'
                
        if fileexist:
            b1bunchdata = pd.read_csv(outfn1,delimiter=',',header=None,skiprows=[0,1,2])
            b2bunchdata = pd.read_csv(outfn2,delimiter=',',header=None,skiprows=[0,1,2])
            return [int((i-1.)/10.) for i in b1bunchdata[range(1,3565)][b1bunchdata[range(1,3565)].sum(axis=1).values>0.1].tail(1).values[0] if i >0.1],                [int((i-1.)/10.) for i in b2bunchdata[range(1,3565)][b2bunchdata[range(1,3565)].sum(axis=1).values>0.1].tail(1).values[0] if i >0.1]
        else:  
            print 'Something went wrong, no data loaded.'
    
    # returns the bunch lengths 
    def getbunchlenghts(self,fnb1,fnb2):
        tdc = self.gettimes()
        fileexist = False
        outfn1  = self.basedir + "/" + fnb1 + ".CSV"
        outfn2  = self.basedir + "/" + fnb2 + ".CSV"
        if (os.path.isfile(self.basedir + '/' + fnb1 + '.CSV')) and                (os.path.isfile(self.basedir + '/' + fnb1 + '.CSV')):
            fileexist = True
        else:
            stop  = tdc[str(self.fillnumber)][-1][1]
            start = tdc[str(self.fillnumber)][-1][0]

            shcmdBQMLB1 = './cern-ldb -vs ' + self.timbervarBQMB1L + ' -t1 \"' + start + '\" -t2 \"' +                            stop + '\" -N ' + fnb1 + ' -F CSV'
            shcmdBQMLB2 = './cern-ldb -vs ' + self.timbervarBQMB2L + ' -t1 \"' + start + '\" -t2 \"' +                            stop + '\" -N ' + fnb2 + ' -F CSV'
#             print shcmdBQMFB1
            try:
                subprocess.call(shcmdBQMLB1,shell=True)
                subprocess.call(shcmdBQMLB2,shell=True)

                bashcmd1 = "mv " + fnb1 + ".CSV " + outfn1
                bashcmd2 = "mv " + fnb2 + ".CSV " + outfn2

                subprocess.call(bashcmd1,shell=True)
                subprocess.call(bashcmd2,shell=True)

                fileexist = True
            except:
                print 'Loading of data failed.'
                
        if fileexist:
            b1bunchdata = pd.read_csv(outfn1,delimiter=',',header=None,skiprows=[0,1,2])
            b2bunchdata = pd.read_csv(outfn2,delimiter=',',header=None,skiprows=[0,1,2])
            bposb1, bposb2 = self.getbunchpositions(fnb1+'pos',fnb2+'pos')
            bposb1[:0]=[0]
            bposb2[:0]=[0]
            return b1bunchdata[bposb1],b2bunchdata[bposb2]
        else:  
            print 'Something went wrong, no data loaded.'
            
    def bsrtsigfromraw(self,bsrtgdreader,bsrtsighreader,bsrtsigvreader,bsrtzerotime):
        exdict = collections.OrderedDict()
        eydict = collections.OrderedDict()
        
        
        # get end of ramp for switching betas
        tdc = self.gettimes()
        rampend = tdc['RAMP'][-1][-1]
        rampend = self.converttimetounix(rampend)

        for lhs,rhs,rrhs in itertools.izip(bsrtgdreader,bsrtsighreader,bsrtsigvreader):
                # skip over the headers
                if len(lhs) > 4:
                    # since the BSRT light is either generated at the undulator or the dipole,
                    # we have to select which beta is used 
                    if (int(lhs[0]) <= rampend):
                        beta = self.betaUndH[0]
                    else:
                        beta = self.betaDipH[0]
                    # looping through the columns of a row in the csv files
                    for i in range(1,len(lhs)):
                        # check if the key is already present in the output dictionary
                        if int(float(lhs[i])) in exdict.keys():
                            excomputed = self.gamma * float(rhs[i])**2/beta
                            exdict[int(float(lhs[i]))].append([(int(lhs[0])-bsrtzerotime)/3600000.,excomputed])
                        else:
                            excomputed = self.gamma * float(rhs[i])**2/beta
                            exdict[int(float(lhs[i]))]=[[(int(lhs[0])-bsrtzerotime)/3600000.,excomputed]]
                        if int(float(lhs[i])) in eydict.keys():
                            eycomputed = self.gamma * float(rrhs[i])**2/beta
                            eydict[int(float(lhs[i]))].append([(int(lhs[0])-bsrtzerotime)/3600000.,eycomputed])
                        else:
                            eycomputed = self.gamma * float(rrhs[i])**2/beta
                            eydict[int(float(lhs[i]))]=[[(int(lhs[0])-bsrtzerotime)/3600000.,eycomputed]]
        return exdict,eydict
    
    def convertdicttodf(self,dictin):
        dflist        = [pd.DataFrame(np.array(dictin[i]),columns=[0,i]) for i in dictin.keys()]
        dflistgrouped = [dflist[i].groupby(0,as_index=False).mean() for i in range(len(dflist))]
        dffinal       = reduce(lambda left,right: pd.merge(left,right,on=0,how='outer'),dflistgrouped)
        dffinal2      = dffinal.fillna(0)
        cols          = dffinal.columns
        return dffinal2.loc[(dffinal2[cols[1:]]!=0).any(1)]
    
    def getemitbsrt(self,fnb1,fnb2):
        if (os.path.isfile(self.basedir + "/" + 'Fill' + str(self.fillnumber) + 'EX1.CSV')):
            ex1dfout = pd.read_csv(self.basedir + "/" + 'Fill' + str(self.fillnumber) + 'EX1.CSV')
            ey1dfout = pd.read_csv(self.basedir + "/" + 'Fill' + str(self.fillnumber) + 'EY1.CSV')
            ex2dfout = pd.read_csv(self.basedir + "/" + 'Fill' + str(self.fillnumber) + 'EX2.CSV')
            ey2dfout = pd.read_csv(self.basedir + "/" + 'Fill' + str(self.fillnumber) + 'EY2.CSV')
            return ex1dfout,ey1dfout,ex2dfout,ey2dfout
        else:
            tdc = self.gettimes()
            fileexist = False
            outfn1  = self.basedir + "/" + fnb1 + 'sigh' + ".CSV"
            outfn2  = self.basedir + "/" + fnb1 + 'sigv' + ".CSV"
            outfn3  = self.basedir + "/" + fnb1 + 'gdh' + ".CSV"
            outfn4  = self.basedir + "/" + fnb1 + 'corh' + ".CSV"
            outfn5  = self.basedir + "/" + fnb1 + 'corv' + ".CSV"
            outfn6  = self.basedir + "/" + fnb2 + 'sigh' + ".CSV"
            outfn7  = self.basedir + "/" + fnb2 + 'sigv' + ".CSV"
            outfn8  = self.basedir + "/" + fnb2 + 'gdh' + ".CSV"
            outfn9  = self.basedir + "/" + fnb2 + 'corh' + ".CSV"
            outfn10  = self.basedir + "/" + fnb2 + 'corv' + ".CSV"
            if (os.path.isfile(self.basedir + '/' + fnb1 + 'sigh' + '.CSV')) and                    (os.path.isfile(self.basedir + '/' + fnb2  + 'sigh' + '.CSV')):
                fileexist = True
            else:
                stop  = tdc[str(self.fillnumber)][-1][1]
                start = tdc[str(self.fillnumber)][-1][0]

                bashcmdBSRTSIGHB1 = './cern-ldb -vs ' +  self.timbervarBSRTB1H + ' -t1 \"' + start +                            '\" -t2 \"' + stop + '\" -N ' + fnb1 + 'sigh' + ' -F CSV'
                bashcmdBSRTSIGVB1 = './cern-ldb -vs ' +  self.timbervarBSRTB1V + ' -t1 \"' + start +                            '\" -t2 \"' + stop + '\" -N ' + fnb1 + 'sigv' + ' -F CSV'
                bashcmdBSRTGDHB1  = './cern-ldb -vs ' +  self.timbervarBSRTB1GD + ' -t1 \"' + start +                            '\" -t2 \"' + stop + '\" -N ' + fnb1 + 'gdh' + ' -F CSV'
                bashcmdBSRTCORHB1 = './cern-ldb -vs ' +  self.timbervarBSRTB1CH + ' -t1 \"' + start +                            '\" -t2 \"' + stop + '\" -N ' + fnb1 + 'corh' + ' -F CSV'
                bashcmdBSRTCORVB1 = './cern-ldb -vs ' +  self.timbervarBSRTB1CV + ' -t1 \"' + start +                            '\" -t2 \"' + stop + '\" -N ' + fnb1 + 'corv' + ' -F CSV'

                bashcmdBSRTSIGHB2 = './cern-ldb -vs ' +  self.timbervarBSRTB2H + ' -t1 \"' + start +                            '\" -t2 \"' + stop + '\" -N ' + fnb2 + 'sigh' + ' -F CSV'
                bashcmdBSRTSIGVB2 = './cern-ldb -vs ' +  self.timbervarBSRTB2V + ' -t1 \"' + start +                            '\" -t2 \"' + stop + '\" -N ' + fnb2 + 'sigv' + ' -F CSV'
                bashcmdBSRTGDHB2  = './cern-ldb -vs ' +  self.timbervarBSRTB2GD + ' -t1 \"' + start +                            '\" -t2 \"' + stop + '\" -N ' + fnb2 + 'gdh' + ' -F CSV'
                bashcmdBSRTCORHB2 = './cern-ldb -vs ' +  self.timbervarBSRTB2CH + ' -t1 \"' + start +                            '\" -t2 \"' + stop + '\" -N ' + fnb2 + 'corh' + ' -F CSV'
                bashcmdBSRTCORVB2 = './cern-ldb -vs ' +  self.timbervarBSRTB2CV + ' -t1 \"' + start +                            '\" -t2 \"' + stop + '\" -N ' + fnb2 + 'corv' + ' -F CSV'

                try:
                    subprocess.call(bashcmdBSRTSIGHB1,shell=True)
                    subprocess.call(bashcmdBSRTSIGVB1,shell=True)
                    subprocess.call(bashcmdBSRTGDHB1,shell=True)
                    subprocess.call(bashcmdBSRTCORHB1,shell=True)
                    subprocess.call(bashcmdBSRTCORVB1,shell=True)

                    subprocess.call(bashcmdBSRTSIGHB2,shell=True)
                    subprocess.call(bashcmdBSRTSIGVB2,shell=True)
                    subprocess.call(bashcmdBSRTGDHB2,shell=True)
                    subprocess.call(bashcmdBSRTCORHB2,shell=True)
                    subprocess.call(bashcmdBSRTCORVB2,shell=True)

                    bashcmd1 = "mv " + fnb1 + 'sigh' + ".CSV " + outfn1
                    bashcmd2 = "mv " + fnb1 + 'sigv' + ".CSV " + outfn2
                    bashcmd3 = "mv " + fnb1 + 'gdh'  + ".CSV " + outfn3
                    bashcmd4 = "mv " + fnb1 + 'corh' + ".CSV " + outfn4
                    bashcmd5 = "mv " + fnb1 + 'corv' + ".CSV " + outfn5

                    bashcmd6 = "mv " + fnb2 + 'sigh' + ".CSV " + outfn6
                    bashcmd7 = "mv " + fnb2 + 'sigv' + ".CSV " + outfn7
                    bashcmd8 = "mv " + fnb2 + 'gdh'  + ".CSV " + outfn8
                    bashcmd9 = "mv " + fnb2 + 'corh' + ".CSV " + outfn9
                    bashcmd10 = "mv " + fnb2 + 'corv' + ".CSV " + outfn10

                    subprocess.call(bashcmd1,shell=True)
                    subprocess.call(bashcmd2,shell=True)
                    subprocess.call(bashcmd3,shell=True)
                    subprocess.call(bashcmd4,shell=True)
                    subprocess.call(bashcmd5,shell=True)
                    subprocess.call(bashcmd6,shell=True)
                    subprocess.call(bashcmd7,shell=True)
                    subprocess.call(bashcmd8,shell=True)
                    subprocess.call(bashcmd9,shell=True)
                    subprocess.call(bashcmd10,shell=True)

                    fileexist = True
                except:
                    print 'Loading of data failed.'
            if fileexist:
                # opening the files
                fgdb1   = open(outfn3,'rU')
                fsighb1 = open(outfn1,'rU')
                fsigvb1 = open(outfn2,'rU')

                fgdb2   = open(outfn8,'rU')
                fsighb2 = open(outfn6,'rU')
                fsigvb2 = open(outfn7,'rU')

                # creating the csv_reader objects
                bsrtgdb1reader = csv.reader(fgdb1)
                bsrtsighb1reader = csv.reader(fsighb1)
                bsrtsigvb1reader = csv.reader(fsigvb1)

                bsrtgdb2reader = csv.reader(fgdb2)
                bsrtsighb2reader = csv.reader(fsighb2)
                bsrtsigvb2reader = csv.reader(fsigvb2)

                # creating a variable containing the zero time moment
                rowlist      = [row for row in bsrtgdb1reader]
                bsrtzerotime = int(rowlist[3][0])

                rowlistb2      = [row for row in bsrtgdb2reader]
                bsrtzerotimeb2 = int(rowlistb2[3][0])

                # reloading the GD file and recreating the csv_reader object since it has been used (Pointers!!!)
                fgdb1 =open(outfn3,'rU')
                bsrtgdb1reader = csv.reader(fgdb1)

                fgdb2 =open(outfn8,'rU')
                bsrtgdb2reader = csv.reader(fgdb2)

                # initializing the dictionary that will contain the processed data
                # keys are the bunchslot numbers
                # values are a list of 2D lists that contain a timestamp and the normalized emittance at that time
                exb1dict = collections.OrderedDict()
                eyb1dict = collections.OrderedDict()
                exb2dict = collections.OrderedDict()
                eyb2dict = collections.OrderedDict()

                exb1dict,eyb1dict = self.bsrtsigfromraw(bsrtgdb1reader,bsrtsighb1reader,bsrtsigvb1reader,bsrtzerotime)
                exb2dict,eyb2dict = self.bsrtsigfromraw(bsrtgdb2reader,bsrtsighb2reader,bsrtsigvb2reader,bsrtzerotimeb2)

                bashcmd1 = "rm " + outfn1
                bashcmd2 = "rm " + outfn2
                bashcmd3 = "rm " + outfn3
                bashcmd4 = "rm " + outfn4
                bashcmd5 = "rm " + outfn5

                bashcmd6 = "rm " + outfn6
                bashcmd7 = "rm " + outfn7
                bashcmd8 = "rm " + outfn8
                bashcmd9 = "rm " + outfn9
                bashcmd10 = "rm " + outfn10
                
                subprocess.call(bashcmd1,shell=True)
                subprocess.call(bashcmd2,shell=True)
                subprocess.call(bashcmd3,shell=True)
                subprocess.call(bashcmd4,shell=True)
                subprocess.call(bashcmd5,shell=True)
                subprocess.call(bashcmd6,shell=True)
                subprocess.call(bashcmd7,shell=True)
                subprocess.call(bashcmd8,shell=True)
                subprocess.call(bashcmd9,shell=True)
                subprocess.call(bashcmd10,shell=True)
                
                ex1dfout =self.convertdicttodf(exb1dict)
                ey1dfout =self.convertdicttodf(eyb1dict)
                ex2dfout =self.convertdicttodf(exb2dict)
                ey2dfout =self.convertdicttodf(eyb2dict)
        
                ex1dfout.to_csv(self.basedir + "/" + 'Fill' + str(self.fillnumber) + 'EX1.CSV')
                ey1dfout.to_csv(self.basedir + "/" + 'Fill' + str(self.fillnumber) + 'EY1.CSV')
                ex2dfout.to_csv(self.basedir + "/" + 'Fill' + str(self.fillnumber) + 'EX2.CSV')
                ey2dfout.to_csv(self.basedir + "/" + 'Fill' + str(self.fillnumber) + 'EY2.CSV')
                
                return ex1dfout,ey1dfout,ex2dfout,ey2dfout
            else:
                print 'Something went wrong, no data loaded'
                
    def getFBCT(self,fnb1,fnb2):
        tdc = self.gettimes()
        fileexist = False
        outfn1  = self.basedir + "/" + fnb1 + ".CSV"
        outfn2  = self.basedir + "/" + fnb2 + ".CSV"
        
        cols1,cols2 = self.getbunchpositions(fnb1+'pos',fnb2+'pos')
        cols1 = [i + 1 for i in cols1]
        cols2 = [i + 1 for i in cols2]
        cols1[:0] = [0]
        cols2[:0] = [0]
        
        if (os.path.isfile(self.basedir + '/' + fnb1 + '.CSV')) and                (os.path.isfile(self.basedir + '/' + fnb1 + '.CSV')):
            fileexist = True
        else:
            stop  = tdc[str(self.fillnumber)][-1][1]
            start = tdc[str(self.fillnumber)][-1][0]
            
            bashcmdFBCTB1 = './cern-ldb -vs ' +  self.timbervarFBCTB1 + ' -t1 \"' + start +                            '\" -t2 \"' + stop + '\" -N ' + fnb1 + ' -F CSV'
            bashcmdFBCTB2 = './cern-ldb -vs ' +  self.timbervarFBCTB2 + ' -t1 \"' + start +                            '\" -t2 \"' + stop + '\" -N ' + fnb2 + ' -F CSV'
            try:
                subprocess.call(bashcmdFBCTB1,shell=True)
                subprocess.call(bashcmdFBCTB2,shell=True)
                
                bashcmd1 = "mv " + fnb1 +  ".CSV " + outfn1
                bashcmd2 = "mv " + fnb2 +  ".CSV " + outfn2
                
                subprocess.call(bashcmd1,shell=True)
                subprocess.call(bashcmd2,shell=True)
                fileexist = True
            
                b1bunchdata = pd.read_csv(outfn1,delimiter=',',header=None,names=range(3565),
                                          skiprows=[0,1,2],usecols=cols1)
                b2bunchdata = pd.read_csv(outfn2,delimiter=',',header=None,names=range(3565),
                                          skiprows=[0,1,2],usecols=cols2)
                b1bunchdata.columns = [i-1 if i !=0 else 0 for i in cols1]
                b2bunchdata.columns = [i-1 if i !=0 else 0 for i in cols2]
                b1bunchdata.to_csv(outfn1,index=None)
                b2bunchdata.to_csv(outfn2,index=None)
            except:
                print 'Loading of data failed.'
        if fileexist:
            
            b1bunchdata = pd.read_csv(outfn1)
            b2bunchdata = pd.read_csv(outfn2)
            return b1bunchdata,b2bunchdata
        else:
            print 'Something went wrong no data loaded.'
            return 0
    
    def getlumi(self,fnatlas,fncms,fnalice,fnlhcb):
        tdc = self.gettimes()
        fileexist = False
        outfnatlas  = self.basedir + "/" + fnatlas + ".CSV"
        outfncms    = self.basedir + "/" + fncms   + ".CSV"
        outfnalice  = self.basedir + "/" + fnalice + ".CSV"
        outfnlhcb   = self.basedir + "/" + fnlhcb  + ".CSV"        
        
        if os.path.isfile(self.basedir + '/' + fnatlas + '.CSV'):
            fileexist = True
        else:
            stop  = tdc[str(self.fillnumber)][-1][1]
            start = tdc[str(self.fillnumber)][-1][0]
            
            bashcmdatlas = './cern-ldb -vs ' +  self.timbervarLumiAtlas + ' -t1 \"' + start +                            '\" -t2 \"' + stop + '\" -N ' + fnatlas + ' -F CSV'
            bashcmdcms   = './cern-ldb -vs ' +  self.timbervarLumiCMS + ' -t1 \"' + start +                            '\" -t2 \"' + stop + '\" -N ' + fncms + ' -F CSV'
            bashcmdalice = './cern-ldb -vs ' +  self.timbervarLumiAlice + ' -t1 \"' + start +                            '\" -t2 \"' + stop + '\" -N ' + fnalice + ' -F CSV'
            bashcmdlhcb  = './cern-ldb -vs ' +  self.timbervarLumiLHCB + ' -t1 \"' + start +                            '\" -t2 \"' + stop + '\" -N ' + fnlhcb + ' -F CSV'
            try:
                subprocess.call(bashcmdatlas,shell=True)
                subprocess.call(bashcmdcms,shell=True)
                subprocess.call(bashcmdalice,shell=True)
                subprocess.call(bashcmdlhcb,shell=True)
                
                bashcmd1 = "mv " + fnatlas +  ".CSV " + outfnatlas
                bashcmd2 = "mv " + fncms   +  ".CSV " + outfncms
                bashcmd3 = "mv " + fnalice +  ".CSV " + outfnalice
                bashcmd4 = "mv " + fnlhcb  +  ".CSV " + outfnlhcb
                
                subprocess.call(bashcmd1,shell=True)
                subprocess.call(bashcmd2,shell=True)
                subprocess.call(bashcmd3,shell=True)
                subprocess.call(bashcmd4,shell=True)
                
                fileexist = True
            
                atlaslumidata = pd.read_csv(outfnatlas,delimiter=',',header=None,names=['t','L'],
                                          skiprows=[0,1,2])
                cmslumidata = pd.read_csv(outfncms,delimiter=',',header=None,names=['t','L'],
                                          skiprows=[0,1,2])
                alicelumidata = pd.read_csv(outfnalice,delimiter=',',header=None,names=['t','L'],
                                          skiprows=[0,1,2])
                lhcblumidata = pd.read_csv(outfnlhcb,delimiter=',',header=None,names=['t','L'],
                                          skiprows=[0,1,2])
                
                atlaslumidata.to_csv(outfnatlas,index=None)
                cmslumidata.to_csv(outfncms,index=None)
                alicelumidata.to_csv(outfnalice,index=None)
                lhcblumidata.to_csv(outfnlhcb,index=None)
            except:
                print 'Loading of data failed.'
        if fileexist:
            
            atlaslumidata = pd.read_csv(outfnatlas)
            cmslumidata = pd.read_csv(outfncms)
            alicelumidata = pd.read_csv(outfnalice)
            lhcblumidata = pd.read_csv(outfnlhcb)
            return atlaslumidata,cmslumidata,alicelumidata,lhcblumidata
        else:
            print 'Something went wrong no data loaded.'
            return 0
    
    def gethorbpm(self,fn):
        tdc = self.gettimes()
        fileexist = False
        outfn  = self.basedir + "/" + fn + ".CSV"
        if os.path.isfile(self.basedir + '/' + fn + '.CSV'):
            fileexist = True
        else:
            stop  = tdc[str(self.fillnumber)][-1][1]
            start = tdc[str(self.fillnumber)][-1][0]
            bashcmdhorbpm = './cern-ldb -vs ' +  self.timbervarhorbpm + ' -t1 \"' + start +                            '\" -t2 \"' + stop + '\" -sa REPEAT -ss 5 -si MINUTE -N ' + fn + ' -F CSV'
            try:
                subprocess.call(bashcmdhorbpm,shell=True)
                fileexist = True
                bashcmd1 = "mv " + fn +  ".CSV " + outfn
                subprocess.call(bashcmd1,shell=True)
                
                bpmdata = pd.read_csv(outfn,delimiter=',',header=None,
                                          skiprows=[0,1,2])
                bpmdata.to_csv(outfn,index=None)
            except:
                print 'Loading of data failed.'
        if fileexist:
            df = pd.read_csv('/afs/cern.ch/work/t/tomerten/HI2015/bpmhnames.csv',skiprows=[0,1,2])
            colnames = list(df.columns[1:])
            colnames[:0] =['t']
            bpmdata = pd.read_csv(outfn,names=colnames,skiprows=[0])
            return bpmdata
        else:
            print 'Something went wrong no data loaded.'
            return 0
        
    def gethorbpmmask(self,fn):
        tdc = self.gettimes()
        fileexist = False
        outfn  = self.basedir + "/" + fn + ".CSV"
        if os.path.isfile(self.basedir + '/' + fn + '.CSV'):
            fileexist = True
        else:
            stop  = tdc[str(self.fillnumber)][-1][1]
            start = tdc[str(self.fillnumber)][-1][0]
            bashcmdhorbpmnames = './cern-ldb -vs ' +  'LHC.BOFSU:BPM_MASK_H' + ' -t1 \"' + start +                            '\" -t2 \"'  + stop + '\" -N ' + 'bpmhmask' + str(self.fillnumber) + ' -F CSV'
            try:
                subprocess.call(bashcmdhorbpmnames,shell=True)
                fileexist = True
                bashcmd1 = "mv " + 'bpmhmask4696' +  ".CSV " + '/afs/cern.ch/work/t/tomerten/HI2015/bpmhmask4696.csv'
                subprocess.call(bashcmd1,shell=True)
                dfmask = pd.read_csv(outfn,skiprows=[0,1,2],header=None)
                dfmask.to_csv(outfn,index=None)
            except:
                print 'Loading of data failed.'
        if fileexist:
            df = pd.read_csv(outfn,skiprows=[0,1,2])
            bpmdata = pd.read_csv(outfn,skiprows=[0,1,2],header=None)
            return bpmdata
        else:
            print 'Something went wrong no data loaded.'
            return 0
        
    def getbpmhreduced(self):
        dfmask = self.bpmhmask

        # selecting the last non-zero row of the maskfile to use as a mask on the bpm data
        # not the best way but does the job for the moment
        # problem to unequal shape of mask and data dataframes
        dfcopy = pd.DataFrame(dfmask[range(1,1089)][dfmask[range(1,1089)].sum(axis=1).values>1.].tail(1).values, 
                              columns =  self.bpmhdf.drop(self.bpmhdf.columns[0],axis=1).columns)

        # taking the bpm data but removing the timestamps in order to be able to apply a mask
        dffff = self.bpmhdf.drop(self.bpmhdf.columns[0],axis=1)
        selection = dffff * dfcopy.iloc[0]
        selectionred= selection[(selection>1.0)| (selection< -1.0)].dropna(axis=1,how='all')
        return selectionred
    
    def transformbpmdata(self,tfslhcb1,tfslhcb2,ipnr=5):
        selectionred = self.getbpmhreduced()
        # selecting the BPM around the desired ip (pandas dataframes)
        bpmrtest = selectionred[(selectionred.columns[selectionred.columns.to_series().str.contains('R' + str(ipnr)
                                                                                                   + '.B1')])]
        bpmltest = selectionred[(selectionred.columns[selectionred.columns.to_series().str.contains('L' + str(ipnr)
                                                                                                   + '.B2')])]
        # adding the s positions of these BPMs for plotting
        tfsb1  = pd.read_csv(tfslhcb1,skiprows=range(45),nrows=2,delim_whitespace=True)
        tfsb1  = tfsb1[tfsb1['NAME']!='%s']
        colsb1 = list(tfsb1.columns[1:])

        tfsb1 = pd.read_csv(tfslhcb1,skiprows=range(46),delim_whitespace=True,names=colsb1,index_col=False)
        tfsb1 = tfsb1[tfsb1['S']!='%s']

        tfsbpmr= tfsb1[(tfsb1['NAME'].str.contains('BPM')) & (tfsb1['NAME'].str.contains('R' + str(ipnr) + '.B1'))]
        tfsbpmr = tfsbpmr[tfsbpmr['NAME']!='BPMSW.1R5.B1_DOROS']
        tfsbpmr = tfsbpmr[['NAME','S']]

        namesr = list(bpmrtest.columns)

        tfsb2 = pd.read_csv(tfslhcb2,skiprows=range(45),nrows=2,delim_whitespace=True)
        tfsb2 = tfsb2[tfsb2['NAME']!='%s']
        colsb2 = list(tfsb2.columns[1:])
        tfsb2 = pd.read_csv(tfslhcb2,skiprows=range(46),delim_whitespace=True,names=colsb2,index_col=False)
        tfsb2 = tfsb2[tfsb2['S']!='%s']
        tfsbpml = tfsb2[(tfsb2['NAME'].str.contains('BPM')) & (tfsb2['NAME'].str.contains('L' + str(ipnr) + '.B2'))]
        tfsbpml = tfsbpml[1:]
        tfsbpml = tfsbpml[['NAME','S']]

        namesl = list(bpmltest.columns)

        return bpmltest,namesl,tfsbpml,bpmrtest,namesr,tfsbpmr
    
    def plotbpm(self,tfslhcb1,tfslhcb2,ipnr=5,nn=50,step=20,xmin=0,xmax=600):
        bpmltest,namesl,tfsbpml,bpmrtest,namesr,tfsbpmr = self.transformbpmdata(
                                                    tfslhcb1,
                                                    tfslhcb2,
                                                    ipnr=ipnr
                                                   )
        from IPython.html.widgets import FloatProgress
        from IPython.display import display
        f = FloatProgress(min=0,max=2*nn/float(step))
        display(f)

        error = False

        tfsb1  = pd.read_csv(tfslhcb1,skiprows=range(45),nrows=2,delim_whitespace=True)
        tfsb1  = tfsb1[tfsb1['NAME']!='%s']
        colsb1 = list(tfsb1.columns[1:])

        tfsb1 = pd.read_csv(tfslhcb1,skiprows=range(46),delim_whitespace=True,names=colsb1,index_col=False)
        tfsb1 = tfsb1[tfsb1['S']!='%s']
        sipb1 = tfsb1[tfsb1['NAME']== 'IP'+ str(ipnr)]['S'].values[0]

        tfsb2  = pd.read_csv(tfslhcb2,skiprows=range(45),nrows=2,delim_whitespace=True)
        tfsb2  = tfsb2[tfsb2['NAME']!='%s']
        colsb2 = list(tfsb2.columns[1:])

        tfsb2 = pd.read_csv(tfslhcb2, skiprows=range(46),delim_whitespace=True,names=colsb2,index_col=False)
        tfsb2 = tfsb2[tfsb2['S']!='%s']
        sipb2 = tfsb2[tfsb2['NAME']== 'IP'+ str(ipnr)]['S'].values[0]

        tfsbpmr['S'] = tfsbpmr['S'].apply(lambda x: float(x)-float(sipb1))
        tfsbpml['S'] = tfsbpml['S'].apply(lambda x: float(x)-float(sipb2))

    #     print tfsbpmr

        fig = plt.figure(figsize=(18,20))
        ax  = fig.add_subplot(211)

        s1   = np.array([float(tfsbpml[tfsbpml['NAME']==n]['S'].values[0]) for n in namesl])
        max1 = len(bpmltest)


        for i in range(0,len(bpmltest.tail(nn)),step):
            try:
                data = np.array([bpmltest[n][max1-nn+i] for n in namesl])
                plt.scatter(s1,data)
                plt.plot(s1,data)
                f.value = i
            except:
                print 'nn out of range'
                error = True
                break


    #     plt.plot((0,500),(500,500),'k-')
    #     plt.plot((0,500),(-500,-500),'k-')
    #     plt.plot((0,500),(-3000,-3000),'k-')
    #     plt.plot((0,500),(-3500,-3500),'k-')
    #     plt.plot((0,500),(0,0),'k-')
        plt.grid()
        plt.xlim(xmin,xmax)
        plt.ylim(-5000,5000)

        ax.set_xticks(np.arange(xmin,xmax,100))
        ax.set_xticks(np.arange(xmin,xmax,10),minor=True)
        ax.set_yticks(np.arange(-5000,5000,2000))
        ax.set_yticks(np.arange(-5000,5000,500),minor=True)
        ax.grid(which='minor',alpha=0.65)

        plt.xlabel('s',fontsize=16.0)
        plt.ylabel('x [um]')
        plt.title('BPM orbits Horizontal left of ' + str(ipnr) + ' - fill ' + str(LHCfill.fillnumber))

    #     ax.text(100,4000, 'Each line (colour) is at a different timestamp.')

    #     ax.annotate('roughly 0.5 mm below \n 3mm from bump',xy=(440,-3400),xytext=(400,-5000),
    #                 arrowprops=dict(facecolor='red',shrink=0.5))

    #     ax.plot([440],[500],'o')

    #     ax.annotate('Expected with 0.5 mm bump',xy=(440,500),xytext=(400,2500),
    #                 arrowprops=dict(facecolor='red',shrink=0.2))

        ax2  = fig.add_subplot(2,1,2)
        max2 = len(bpmrtest)
        s2   = np.array([float(tfsbpmr[tfsbpmr['NAME']==n]['S'].values[0]) for n in namesr])

        for i in range(0,len(bpmrtest.tail(nn)),step):
            try:
                data =  np.array([bpmrtest[n][max2-nn+i] for n in namesr])
                ax2.scatter(s2,data)
                ax2.plot(s2,data)
                f.value = f.value +1 
            except:
                print 'nn out of range'
                error = True
                break
        if error:
            print 'No valid plot generated'
        else:
    #         plt.plot((0,500),(500,500),'k-')
    #         plt.plot((0,500),(-500,-500),'k-')
    #         plt.plot((0,500),(-3000,-3000),'k-')
    #         plt.plot((0,500),(-3500,-3500),'k-')
    #         plt.plot((0,500),(0,0),'k-')
            plt.grid()
            plt.xlim(xmin,xmax)
            plt.ylim(-5000,5000)

            ax2.set_xticks(np.arange(xmin,xmax,100))
            ax2.set_xticks(np.arange(xmin,xmax,10),minor=True)
            ax2.set_yticks(np.arange(-5000,5000,2000))
            ax2.set_yticks(np.arange(-5000,5000,500),minor=True)

            ax2.grid(which='minor',alpha=0.65)
    #         ax2.annotate('roughly 0.5 mm above \n 3mm from bump',xy=(433,-2650),xytext=(400,-5000),
    #                     arrowprops=dict(facecolor='red',shrink=0.2))

            plt.xlabel('s',fontsize=16.0)
            plt.ylabel('x [um]')
            plt.title('BPM orbits Horizontal right of ' + str(ipnr) + '  - fill ' + str(LHCfill.fillnumber) )

            plt.show()
            print f.value
            # plt.savefig('Fill4707IP5bpm.png',format='png')
