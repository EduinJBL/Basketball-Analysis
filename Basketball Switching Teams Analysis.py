# -*- coding: utf-8 -*-
"""
Created on Sat Nov 28 09:51:19 2020

@author: eblat
"""
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from base64 import b64encode
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.express as px
import plotly.graph_objs as go
from linearmodels import PanelOLS

#SECTION 1: Importing and Manipulating Data

#read data from Neil Paine GitHub
raptor_data=pd.read_csv(r'https://raw.githubusercontent.com/fivethirtyeight/nba-player-advanced-metrics/master/nba-data-historical.csv')

#Some cleaning of data
#Keep only players who played more than 700 mins
raptor_data=raptor_data[raptor_data.Min>700]
#Keep only regular season data
raptor_data=raptor_data[raptor_data.type=="RS"]
#Keep only data from 2000-2019
raptor_data=raptor_data[raptor_data.year_id>=2000]
raptor_data=raptor_data[raptor_data.year_id<=2019]
#sort values
raptor_data.sort_values(['year_id','player_id','Min'],ascending=False,inplace=True)
#drop observations when players have transferred midseason. Keep observation with most minutes, and drop other instances
raptor_data.drop_duplicates(subset=['year_id','player_id'],keep='first',inplace=True)

#subset data to columns used in analysis
raptor_data=raptor_data[['player_id','name_common','year_id',
                         'USG%','2P%','3P%','FT%',
                         'TS%','AST%','TOV%','ORB%',
                         'DRB%','STL%','BLK%','Raptor O','Raptor D', 
                         'Raptor+/-', 'franch_id','age','Min']]


#generate a rounded age variable to be used as a dummy. Round to the nearest 5  
def custom_round(x, base=5):
    return int(base * round(float(x)/base))

raptor_data['round_age']=raptor_data.age.apply(lambda x: custom_round(x, base=5))

#generate lagged version of dataset and merge with original. From here on out
# _x refers to values from season t-1 and _y refers to values from season t
raptor_data_lag=raptor_data
raptor_data_lag['year_id_lag']=raptor_data.year_id.add(1)
raptor_data=raptor_data_lag.merge(raptor_data, left_on=['year_id_lag','player_id'],
                                  right_on=['year_id','player_id'])
#drop observations where not two consecutive seasons with 700 mins
raptor_data.dropna(how='any',inplace=True)

#generate value indicating if switched teams
condition=[raptor_data.franch_id_x!=raptor_data.franch_id_y]
choice=[1]
raptor_data['traded']=np.select(condition,choice,default=0)
#create seperate datasets with relevant subset of data
raptor_data_S=raptor_data[raptor_data.franch_id_x==raptor_data.franch_id_y]
raptor_data_D=raptor_data[raptor_data.franch_id_x!=raptor_data.franch_id_y]
#set index for future analysis
raptor_data.set_index(['player_id','year_id_y'],inplace=True)

#SECTION 2: Generating functions to carry out core analysis
#Aim is to define a function that takes variable from data and finds the absolute 
#difference between t0 and t-1 and then runs regression to explain that difference

#first define a function that returns variable string names and seperated x and y data
def xyvars(data,v):
    xvar=v+'_x'
    yvar=v+'_y'
    xdata=data[xvar]
    ydata=data[yvar]
    outputs=[xdata,ydata]
    return(outputs)

#then define a function that finds absolute difference between y and x/ypred 
#normalises the variable and adds it to df

def absdiff(data,v,x,y):
    absstr=v+'_absdiff'
    absstrm=v+'_absmean'
    absstrsd=v+'_abssd'
    data[absstr]=x-y
    data[absstr]=data[absstr].abs()
    data[absstrm]=data[absstr].mean()
    data[absstrsd]=data[absstr].std()
    #normalise
    data[absstr]=(data[absstr]-data[absstrm])/data[absstrsd]
    return(data[absstr])

#Now define function  which carries out complete process: 
#It takes absolute difference between variables and 
#then runs fixed effect regression 
#and returns summary stats for key coefficient of interest

def compproc(data,v):
    out=xyvars(data,v)
    xdata=out[0]
    ydata=out[1]
    abdata=absdiff(data,v,xdata,ydata)
    X=pd.get_dummies(data['round_age_x'],drop_first=True)
    xvar=v+'_x'
    X['Min_x']=data['Min_x']
    X['Min_y']=data['Min_y']
    X['traded']=data['traded']
    X=sm.add_constant(X)
    absstr=v+'_absdiff'
    mod=PanelOLS(data[absstr],X,entity_effects=True)
    res=mod.fit()
    print(res)
    params=res.params
    tradecoeff=params.loc['traded']
    conf_int=res.conf_int()
    conf_int=conf_int.loc['traded']
    lowconf=conf_int.iloc[0]
    upconf=conf_int.iloc[1]
    absstrm=v+'_absmean'
    absstrsd=v+'_abssd'
    absmean=data[absstrm].mean()
    abssd=data[absstrsd].mean()  
    return([tradecoeff,lowconf,upconf,absmean,abssd])

#Section 3: Loop over performance variables carrying out process 
#and summarise results in a df
varlist=['2P%','3P%','FT%','TS%','USG%','AST%','TOV%','ORB%',
                         'DRB%','STL%','BLK%','Raptor O','Raptor D', 
                         'Raptor+/-']
reslist=[]
for var in varlist:
    results=compproc(raptor_data,var)
    reslist.append(results)

resdf=pd.DataFrame(reslist,columns=["Coefficient","Upper CI","Lower CI",'Mean Abs Diff','SD Abs Diff'])
resdf['Variable']=varlist
resdf['NonStandardCoeff']=resdf['Coefficient']*resdf['SD Abs Diff']
    
#Section 4: Plot Results

#create barchart with coefficients and CIs 
plt.style.use('seaborn-white')
plt.clf()
r1=np.arange(len(resdf.Variable))
r1=r1*3
yer1=resdf["Upper CI"]-resdf["Coefficient"]
plt.bar(r1,resdf['Coefficient'],yerr=yer1,capsize=3)
plt.xlim([-1,41])
plt.xticks(r1,varlist,fontsize='small',rotation=45)
plt.ylabel("Effect of switching teams (standardised)",wrap=True, fontsize='small')
plt.title("Effect of switching teams on absolute"
           " difference in performance (standardised)", wrap=True)
plt.axhline(color='black', lw=1)
plt.show()

#create scatter graph plot 
plt.clf()
fig,ax=plt.subplots()
ax.scatter(raptor_data_S['Raptor+/-_x'],raptor_data_S['Raptor+/-_y'], s=2,marker='x', color='orange').set_label('Same Team')
ax.scatter(raptor_data_D['Raptor+/-_x'],raptor_data_D['Raptor+/-_y'], s=2).set_label('Switched Team')
plt.xlabel('Raptor +/- Year t-1')
plt.ylabel('Raptor +/- Year t')
plt.title('Raptor +/- in consecutive years')
line=np.arange(-10,15)
ax.plot(line,line, color='black', ls='--')
ax.legend()

                                        
# #INTERACTIVE SCATTER GRAPH:
# 
x=raptor_data['Raptor+/-_x']
y=raptor_data['Raptor+/-_y']
raptor_data['traded']=raptor_data.traded.astype(str)
raptor_data['traded']=raptor_data.traded.str.replace('0','Same Team')
raptor_data['traded']=raptor_data.traded.str.replace('1','Switched Team')
cl=raptor_data['traded']
labs={'Raptor+/-_x':'Raptor_+/- Year 1', 'Raptor+/-_y':'Raptor+/- Year 2',
      'name_common_x':'Name','year_id_x':'Year 1','year_id_y':'Year 2',
      'traded':'Team Status','franch_id_x':'Team 1','franch_id_y':'Team 2'}

fig = px.scatter(x='Raptor+/-_x', y='Raptor+/-_y', color='traded', hover_name='name_common_x', 
                  hover_data=['year_id_y','franch_id_y','year_id_x','franch_id_x'],data_frame=raptor_data,
                  labels=labs)
fig.add_trace(go.Scatter(x=line,y=line, mode='lines', 
                         line=dict(color='black',dash='dash'),showlegend=False))

fig.update_layout(
    autosize=False,
    width=500,
    height=500,)

app = dash.Dash(__name__)


app = dash.Dash()
app.layout = html.Div([
    dcc.Graph(figure=fig)
])

#This line deploys interactive graph to local server, currently commented out. 
#If you want to view interactive graph, remove hash

#app.run_server(debug=True) 
