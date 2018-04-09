import pymssql
import pandas as pd
import numpy as np
import datetime
from fbprophet import Prophet
import matplotlib.pyplot as plt
from fourier import *
import pdb
import pickle
from collections import OrderedDict


#Number of available samples (sample panels) 
NUMMAX=22

#Create datalist 
#datalist= [extract_data(groups, i) for i in range(NUMMAX+1)]
#with open('./data.pkl', 'wb') as f:
#   pickle.dump(datalist, f)

#Load datalist
with open('data.pkl', 'rb') as f:
    datalist = pickle.load(f)

#First sample
YEAR0=2016
MONTH0=1


#Agregated categories and possible values
categories = {'Canal' : [u'NO PRESENCIAL', u'PRESENCIAL'], 'Subcanal' : [u'NO PRESENCIAL', u'NIVEL 1 COMPARTIDO', u'NIVEL 1 EXCLUSIVO\r\n', u'NIVEL 2\r\n'], 'AgrupacionTerritorial' :  [u'N/D', u'CAST LEON NORTE', u'ZARAGOZA', u'LAS PALMAS', u'RESTO CATALU\xd1A', u'EXTREMADURA', u'RESTO GALICIA', u'MADRID', u'GRANADA', u'ZAMORA', u'RESTO PAIS VASCO', u'BARCELONA', u'ALICANTE', u'ALBACETE', u'CANTABRIA', u'CASTELLON', u'RESTO CAST MANCH', u'VIZCAYA', u'ASTURIAS', u'LA RIOJA', u'MALAGA', u'SANTA CRUZ DE TENERIFE', u'VALENCIA', u'MURCIA', u'SEVILLA', u'CADIZ', u'RESTO ANDALUCIA', u'PONTEVEDRA', u'A CORU\xd1A', u'RESTO ARAGON', u'CAST LEON SUR', u'NAVARRA', u'BALEARES'], 'Operador': [u'JAZZTEL', u'LIBRES', u'MASMOVIL', u'ORANGE', u'VODAFONE',u'YOIGO', u'TELECABLE', u'PEPEPHONE', u'SIMYO', u'LEBARA',u'REPUBLICA MOVIL', u'LOWI', u'LYCAMOBILE', u'MOVISTAR', u'TUENTI',u'EUSKALTEL', u'LLAMAYA', u'AMENA', u'CARREFOUR', u'ONO'], 'Gama' : [u'N/D', u'\xa0ALTA', u'\xa0MEDIA', u'\xa0PREMIUM', u'\xa0TOP',u'BAJA'], 'Activacion' : [u'PORTABILIDAD', u'LIBRES', u'ALTA', u'RENOVE POSPAGO',u'MIGRACI\xd3N', u'ALTA SIM', u'RENOVE EMPRESA CIF',u'RENOVE EMPRESA NIF', u'PORTABILIDAD SIM', u'ALTA PACK', u'-',u'SIN ESPECIFICAR', u'PREPAGO SIN ESPECIFICAR TOTAL', u'SIM',u'ADSL', u'PORTABILIDAD PACK', u'RENOVE PREPAGO',u'PREPAGO EMPRESA', u'TELEFON\xcdA FIJA', u'PACK SIN ESPECIFICAR',u'MIGRACI\xd3N PACK', u'MIGRACI\xd3N SIM']}

#Categories for aggregation
grouping_0 =  ['Canal', 'Subcanal','AgrupacionTerritorial', 'Operador', 'Gama', 'Activacion']
aggregation_0 = [[0],[1],[2],[3],[4],[5]]

#grouping_1 =  ['Canal, Subcanal','AgrupacionTerritorial, Operador', 'Gama', 'Activacion']
#aggregation_1 = [[0,1],[2,3],[4],[5]]

#Full split
groups =  'Canal, Subcanal, AgrupacionTerritorial, Operador, Gama, Activacion'

#CANAL = [u'NO PRESENCIAL', u'PRESENCIAL']
#SUBCANAL = [u'NO PRESENCIAL', u'NIVEL 1 COMPARTIDO', u'NIVEL 1 EXCLUSIVO\r\n',u'NIVEL 2\r\n']
#AGRUPACIONTERRITORIAL = [u'N/D', u'CAST LEON NORTE', u'ZARAGOZA', u'LAS PALMAS', u'RESTO CATALU\xd1A', u'EXTREMADURA', u'RESTO GALICIA', u'MADRID', u'GRANADA', u'ZAMORA', u'RESTO PAIS VASCO', u'BARCELONA', u'ALICANTE', u'ALBACETE', u'CANTABRIA', u'CASTELLON', u'RESTO CAST MANCH', u'VIZCAYA', u'ASTURIAS', u'LA RIOJA', u'MALAGA', u'SANTA CRUZ DE TENERIFE', u'VALENCIA', u'MURCIA', u'SEVILLA', u'CADIZ', u'RESTO ANDALUCIA', u'PONTEVEDRA', u'A CORU\xd1A', u'RESTO ARAGON', u'CAST LEON SUR', u'NAVARRA', u'BALEARES']
#OPERADOR = [u'JAZZTEL', u'LIBRES', u'MASMOVIL', u'ORANGE', u'VODAFONE',u'YOIGO', u'TELECABLE', u'PEPEPHONE', u'SIMYO', u'LEBARA',u'REPUBLICA MOVIL', u'LOWI', u'LYCAMOBILE', u'MOVISTAR', u'TUENTI',u'EUSKALTEL', u'LLAMAYA', u'AMENA', u'CARREFOUR', u'ONO']
#GAMA = [u'N/D', u'\xa0ALTA', u'\xa0MEDIA', u'\xa0PREMIUM', u'\xa0TOP',u'BAJA']
#ACTIVACION = [u'PORTABILIDAD', u'LIBRES', u'ALTA', u'RENOVE POSPAGO',u'MIGRACI\xd3N', u'ALTA SIM', u'RENOVE EMPRESA CIF',u'RENOVE EMPRESA NIF', u'PORTABILIDAD SIM', u'ALTA PACK', u'-',u'SIN ESPECIFICAR', u'PREPAGO SIN ESPECIFICAR TOTAL', u'SIM',u'ADSL', u'PORTABILIDAD PACK', u'RENOVE PREPAGO',u'PREPAGO EMPRESA', u'TELEFON\xcdA FIJA', u'PACK SIN ESPECIFICAR',u'MIGRACI\xd3N PACK', u'MIGRACI\xd3N SIM']


#Merging lists
def listunion(a,b):
    for e in b:
        if e not in a:
            a=np.append(a,e)
    return a 

#Sample number to database dateformat    
def num2dstr(number):
    month = MONTH0 + number % 12
    year = YEAR0+number/12
    return str(year)+str(month).zfill(2)


#Sample number to datatime format
def num2date(number): 
    month = MONTH0+((number) % 12)    # number -> 1 + number
    year = YEAR0+(number)/12
    return datetime.date(year,month,1)


#Datetime format to sample number
def date2num(date):
    year = date.year
    month = date.month
    return (year-YEAR0)*12+month-MONTH0


#Datebase date format to sample number
def dstr2num(dstr):
    year = int(dstr[:4])
    month = int(dstr[-2:])
    return (year-YEAR0)*12+month-MONTH0


#Datebase date format to datetime format
def dstr2date(dstr): 
    return num2date(dstr2num(dstr))


#Creates a tuple of time series dataframes for the total amount values of sales0 / sales1
def dfseries(num):
    #Group by "AgrupacionTerritorial" and sum to obtain total amount of sales0 / sales1
    y0 = [datalist[i].groupby([2])[6].sum().sum() for i in range(num+1)]
    y1 = [datalist[i].groupby([2])[7].sum().sum() for i in range(num+1)]
    ds = [num2date(i) for i in range(num+1)]

    #Returns a tuple of 2 dataframes both with a time column and a value column
    return [pd.DataFrame(list(zip(ds, y0)), columns=['ds','y']), pd.DataFrame(list(zip(ds, y1)), columns=['ds','y'])]


#Creates an aggregated dataframe in the categories indicated by "grouping" for the panel corresponding to the sample number "num"
def extract_data(grouping, num=NUMMAX):
    conn = pymssql.connect(server='217.126.168.71', user= 'Quantum', password= '8vR8X7xxb.TwZBpiG6irc', database= 'IOI')
    cursor=conn.cursor()
    cursor.execute("EXECUTE dbo.getGroupedData @groupby='"+grouping+"', @typeData='1', @yearMonth= '"+num2dstr(num)+"'")
    groupdf = pd.DataFrame(cursor.fetchall())
    return groupdf

#Prints rows of a pymssql cursor
def print_cursor(cursor):
    for row in cursor:
        print(row)

#Predicts next value of series dataframe with the Prophet method
def predict_next_Prophet(df):  
    m = Prophet(yearly_seasonality = True)
    m.fit(df)
    nextdate = [num2date(date2num(df['ds'][-1:].sum())+1)]
    future=pd.DataFrame(nextdate,columns=['ds'])
    forecast = m.predict(future)
    return forecast['yhat'][0]


#Predicts next value of series dataframe with the Fourier method
def predict_next_Fourier(df, cutoff=0.3): 
    full=extrapoFourier(df['y'], 1, cutoff)
    return full[-1]


#Predicts next value of series dataframe with the naif windowed method
def predict_next_naif(df,window=1):
    x = df['y']
    #pdb.set_trace()
    if x[-12-window:-12].sum()==0:
        value=2*(x[-1:].sum()-x[-2:-1].sum())+x[-2:-1].sum()
    else:
        value=x[-12:-11].sum()/x[-12-window:-12].sum()*x[-window:].sum()
    return value



#Selection of series predictor

def predict_best(df): 
    #return predict_next_Prophet(df)
    #return predict_next_naif(df,1)
    return predict_next_Fourier(df,0.3)

# This function readjusts all sales values of a dataframe df to sum the final values x=[x[0],x[1]]

# Takes:    df(dataframe)
#           x (adjust values)    x=[x[0],x[1]]

# Returns:  Adjusted dataframe



def adjustdf(df, x):
    v0=df.loc[:,6].sum()
    v1=df.loc[:,7].sum()
    pt = pd.DataFrame(df)
    pt.loc[:,6]=pt.loc[:,6]*x[0]/v0
    pt.loc[:,7]=pt.loc[:,7]*x[1]/v1
    return pt




# This function takes a dataframe N. For each single value in the dataframe constructs a series of its past values. Calls a predict function to get a predicted value 
# for the next month panel, and in this way builds a full prediction panel for next month. 

# Takes:     N (last sample to be considered) 
# Returns:   Panel (N+1)


def predict_full(N): 
    pt = pd.DataFrame(datalist[N])
    pt = pt.set_index(range(6))
    ds = [num2date(i) for i in range(N+1)]
    for i in range(len(pt)):
        if i%1000==0: 
            print(str((i+1))+"/"+str(len(pt)))
        idx=pt.index[i]
        y0=[]
        y1=[]
        for j in range(N+1): 
            dfj = pd.DataFrame(datalist[j])
            dfj = dfj.set_index(range(6))
            try: 
                y0=y0+[np.nan_to_num(dfj.loc[idx][6])]
            except:
                y0=y0+[0.0]

            try: 
                y1=y1+[np.nan_to_num(dfj.loc[idx][7])]
            except:
                y1=y1+[0.0]
        #pdb.set_trace()

        pt.loc[idx][6]=predict_best(pd.DataFrame(list(zip(ds, y0)), columns=['ds','y']))
        pt.loc[idx][7]=predict_best(pd.DataFrame(list(zip(ds, y1)), columns=['ds','y']))

        #print(pd.DataFrame(list(zip(ds, y0))))
        #print(pt.loc[idx][6])
        #print(pd.DataFrame(list(zip(ds, y1))))   
        #print(pt.loc[idx][7]) 
        #pdb.set_trace()
    return pt



# This function extrapolates the sales value series and uses it to readjust the averaged sales values of the last W sample panels as a prediction for next month sales

# Takes:     N (last sample to be considered) 
#            W (window size)

# Returns:   Panel (N+1)

def predict_naif(N,W=1): 
    [series0, series1]= dfseries(N)
    value0= predict_best(series0)
    value1= predict_best(series1)
    pt = pd.DataFrame(datalist[N])
    pt = pt.set_index(range(6))
    for i in range(W-1):
        sumpt=pd.DataFrame(datalist[N-i-1])
        sumpt=sumpt.set_index(range(6))
        pt=pt.add(sumpt,fill_value=0)
    return adjustdf(pt, [value0,value1])





# This function operates like predict_full but makes an adjustment based on the full predicted series. 
# Takes:     N (last sample to be considered) 
# Returns:   Panel (N+1)

def predict_full_adjust(N): 
    pt = predict_full(N)
    [series0, series1]= dfseries(N)
    value0= predict_best(series0)
    value1= predict_best(series1)
    return adjustdf(pt, [value0,value1])



# This function calculates the prediction efficiency of a panel prediction method

# Takes:     panelpredictfunction takes N and args
#            M (size of test set - last M panels)
#            args of the panelpredictfunction after 1st argument 

# Returns:   MAPE error


def test_accuracy(M, panelpredictfunction,*args):
    SAPE0 = 0
    SAPE1 = 0
    for j in range(M): 
        ptdf=panelpredictfunction(NUMMAX-j-1,*args) 
        #pdb.set_trace()
        redf = datalist[NUMMAX-j]
        #ptdf= ptdf.set_index(range(6)) (index is already set)
        redf = redf.set_index(range(6))
        sales0 = redf.loc[:,6].sum()
        sales1 = redf.loc[:,7].sum()
        diff = redf.sub(ptdf,fill_value=0)
        diff.loc[:,6]=abs(diff.loc[:,6])
        diff.loc[:,7]=abs(diff.loc[:,7])
        SAPE0 = SAPE0 + diff.loc[:,6].sum()/sales0 
        SAPE1 = SAPE1 + diff.loc[:,7].sum()/sales1
    MAPE0 = SAPE0/M
    MAPE1 = SAPE1/M
    return [MAPE0, MAPE1]


