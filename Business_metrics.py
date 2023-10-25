def save_bm():
    x = business_metrics_resource_m()
    y,z = business_metrics_CIP()
    sum = summary()
    writer = pd.ExcelWriter('20211017_Takeda_Priority Production(With CIP Freezed)_v11.xlsx', engine='xlsxwriter')
    x.to_excel(writer, sheet_name='Business_metrics_Resource')
    y.to_excel(writer, sheet_name='Business_metrics_Constraint')
    z.to_excel(writer, sheet_name='Business_metrics_CIP')
    sum.to_excel(writer, sheet_name='Summary')
    writer.save()

def get_data():
    get_lib()
    global dataset
    #Importing the Output file for QC Check
    dataset=pd.read_excel(r'C:\Users\AtulPoddar\OneDrive - TheMathCompany Private Limited\Documents\Takeda\Modelling_Test_Results\20211017_Takeda_Priority Production(With CIP Freezed)_v11.xlsx',engine='openpyxl')
    dataset['Cleaning Duration'] = dataset['Cleaning Duration'] / 60
    dataset['MC Group'] = dataset['MC Group'].fillna(1)
    dataset['Missed Cleanings'] = np.where(dataset['MC Group']==1,1,0)
    dataset['timebw'] = dataset['Clean_Start_Time'] - dataset['Usage_End']
    dataset['timebw'] = dataset['timebw'] / np.timedelta64(1, 'h')
    return dataset

def get_lib():
    global pd
    global np
    global math
    global warnings
    global sp
    global sm
    global dt
    global ew
    import pandas as pd
    import numpy as np
    import math
    import os
    import warnings
    warnings.filterwarnings("ignore")
    import scipy as sp
    import statsmodels as sm
    import datetime as dt
    from pandas import ExcelWriter as ew
#Function to calculate Business Metrics at a Resource Level WITH Maintenance
def business_metrics_resource_m():
    dataset = get_data()
    dataset['DHT Maintenance Violation'] = np.where((dataset['Maint_Flag'] == 1) & (dataset['DHT Violation Flag']==1),1,0)
    dataset['Maintenance_time'] = np.where(dataset['Maint_Flag']==1,dataset['Utilized'],0)
    if dataset['Clean_Start_Time'].min() > dataset['Usage_Start'].min():
        working_hours = (dataset['Clean_End_Time'].max()-dataset['Usage_Start'].min())/ (np.timedelta64(1, 'h'))
    else:
        working_hours = (dataset['Clean_End_Time'].max()-dataset['Clean_Start_Time'].min())/ (np.timedelta64(1, 'h'))
    dataset['Utilized'] = np.where(dataset['Maint_Flag']==1,0,dataset['Utilized'])
    bm = dataset.groupby('Resource Alt', as_index = False).agg({'Usage_Start': 'min',
                                                    'Usage_End': 'max',
                                                    'Utilized': 'sum',
                                                   'run_flag': 'sum',
                                                   'clean_flag': 'sum',
                                                   'Unutilized_gap':'sum',
                                                   'Cleaning Duration':'max',
                                                   'DHT Violation Flag': 'sum',
                                                   'CHT Violation Flag':'sum',
                                                   'CHT Violation Flag Pre Clean':'sum',
                                                    'Parallel Clean Flag':pd.Series.nunique,
                                                            'Missed Cleanings':'sum',
                                                            'DHT Maintenance Violation':'sum',
                                                             'Maint_Flag':'sum',
                                                             'Maintenance_time':'sum'
                                                  })
    bm['Unutilized_gap'] = bm['Unutilized_gap'].round(1)
    bm['Utilized'] = bm['Utilized'].round(1)
    bm.Schedule_Start= pd.to_datetime(bm.Usage_Start, format = '%d/%m/%Y %H:%M:%S')
    bm.Schedule_End= pd.to_datetime(bm.Usage_End, format = '%d/%m/%Y %H:%M:%S')
    bm = bm.rename(columns = {'Resource Alt':'Resource','Usage_Start': 'Schedule Start', 'Usage_End': 'Schedule End', 'run_flag':'No. of Utilizations',
                                  'clean_flag': 'Total no. of Cleanings','DHT Violation Flag': 'DHT Violations','Unutilized_gap':'Unutilized time',
                                 'Utilized':'Utilized Duration','CHT Violation Flag':'CHT Violations','CHT Violation Flag Pre Clean':'Number of Pre-cleanings',
                                  'Parallel Clean Flag':'Parallel Cleanings', 'Maint_Flag':'No of Maintenance Schedules','Maintenance_time':'Maintenance Duration'})
    bm['Number of Pre-cleanings']= np.where((bm['Number of Pre-cleanings']>0),(bm['Number of Pre-cleanings']-bm['Missed Cleanings']),bm['Number of Pre-cleanings'])
    bm['Total no. of Cleanings'] = bm['Total no. of Cleanings'] - bm['Missed Cleanings'] - bm['Parallel Cleanings']
    bm['Available working Hours'] = round(working_hours,1)
    bm['Cleaning time Available'] = (bm['Available working Hours'] - bm['Utilized Duration']).round(1)
    bm['Total actual Cleaning time'] = (bm['Total no. of Cleanings']*(bm['Cleaning Duration']))
    bm['Resource idle duration'] = (bm['Cleaning time Available'] - bm['Total actual Cleaning time']).round(1)
    bm['Avg Idle Time %'] = ((bm['Resource idle duration']/bm['Available working Hours'])*100).round(1)
    bm['Resource Usage %'] = ((bm['Utilized Duration']/bm['Available working Hours'])*100).round(1)
    bm.loc[(bm['Resource']=='Flexible'),['Utilized Duration','Unutilized time','Available working Hours','Cleaning time Available','Total actual Cleaning time','Resource idle duration','Avg Idle Time %','Resource Usage %']] = ''
    bm.set_index("Resource", inplace=True)
    return bm
#Function to calculate Business Metrics at a Resource Level WITHOUT Maintenance
def business_metrics_resource():
    dataset = get_data()
    if dataset['Clean_Start_Time'].min() > dataset['Usage_Start'].min():
        working_hours = (dataset['Clean_End_Time'].max()-dataset['Usage_Start'].min())/ (np.timedelta64(1, 'h'))
    else:
        working_hours = (dataset['Clean_End_Time'].max()-dataset['Clean_Start_Time'].min())/ (np.timedelta64(1, 'h'))
    bm = dataset.groupby('Resource Alt', as_index = False).agg({'Usage_Start': 'min',
                                                    'Usage_End': 'max',
                                                    'Utilized': 'sum',
                                                   'run_flag': 'sum',
                                                   'clean_flag': 'sum',
                                                   'Unutilized_gap':'sum',
                                                   'Cleaning Duration':'max',
                                                   'DHT Violation Flag': 'sum',
                                                   'CHT Violation Flag':'sum',
                                                   'CHT Violation Flag Pre Clean':'sum',
                                                    'Parallel Clean Flag':pd.Series.nunique,
                                                            'Missed Cleanings':'sum',
                                                            'DHT Maintenance Violation':'sum',
                                                             'Maint_Flag':'sum',
                                                             'Maintenance_time':'sum'
                                                  })

    bm['Unutilized_gap'] = bm['Unutilized_gap'].round(1)
    bm['Utilized'] = bm['Utilized'].round(1)
    bm.Schedule_Start= pd.to_datetime(bm.Usage_Start, format = '%d/%m/%Y %H:%M:%S')
    bm.Schedule_End= pd.to_datetime(bm.Usage_End, format = '%d/%m/%Y %H:%M:%S')
    bm = bm.rename(columns = {'Resource Alt':'Resource','Usage_Start': 'Schedule Start', 'Usage_End': 'Schedule End', 'run_flag':'No. of Utilizations',
                              'clean_flag': 'Total no. of Cleanings','DHT Violation Flag': 'DHT Violations','Unutilized_gap':'Unutilized time',
                             'Utilized':'Utilized Duration','CHT Violation Flag':'CHT Violations','CHT Violation Flag Pre Clean':'Number of Pre-cleanings',
                              'Parallel Clean Flag':'Parallel Cleanings'})
    bm['Number of Pre-cleanings']= np.where((bm['Number of Pre-cleanings']>0),(bm['Number of Pre-cleanings']-bm['Missed Cleanings']),bm['Number of Pre-cleanings'])
    bm['Total no. of Cleanings'] = bm['Total no. of Cleanings'] - bm['Missed Cleanings'] - bm['Parallel Cleanings']
    bm['Available working Hours'] = round(working_hours,1)
    bm['Cleaning time Available'] = (bm['Available working Hours'] - bm['Utilized Duration']).round(1)
    bm['Total actual Cleaning time'] = (bm['Total no. of Cleanings']*(bm['Cleaning Duration']))
    bm['Resource idle duration'] = (bm['Cleaning time Available'] - bm['Total actual Cleaning time']).round(1)
    bm['Avg Idle Time %'] = ((bm['Resource idle duration']/bm['Available working Hours'])*100).round(1)
    bm['Resource Usage %'] = ((bm['Utilized Duration']/bm['Available working Hours'])*100).round(1)
    bm.loc[(bm['Resource']=='Flexible'),['Utilized Duration','Unutilized time','Available working Hours','Cleaning time Available','Total actual Cleaning time','Resource idle duration','Avg Idle Time %','Resource Usage %']] = ''
    bm.set_index("Resource", inplace=True)
    return bm
#Function to calculate Business Metrics at a constraint and CIP level
def business_metrics_CIP():
    dataset = get_data()
    if dataset['Clean_Start_Time'].min() > dataset['Usage_Start'].min():
        working_hours = (dataset['Clean_End_Time'].max()-dataset['Usage_Start'].min())/ (np.timedelta64(1, 'h'))
    else:
        working_hours = (dataset['Clean_End_Time'].max()-dataset['Clean_Start_Time'].min())/ (np.timedelta64(1, 'h'))
    dataset['setup_flag'] = np.where(dataset['timebw'] > 0.166 , 1, 0)
    dataset['setup_flag'] = dataset['setup_flag']*10
    dataset = dataset.rename(columns = {'setup_flag': 'Setup time'})
    dataset['Setup time'] = (dataset['Setup time']/60).round(1)
    metrics_cip = dataset.groupby(['MC Group','Constraint'], as_index = False).agg({
                                                   'Resource': pd.Series.nunique,
                                                    'clean_flag': 'sum',
                                                   'Cleaning Duration':'sum',
                                                    'Setup time':'sum'
                                                    })
    metrics_cip = metrics_cip.rename(columns = {'Resource': 'Resources cleaned',
                               'clean_flag':'No. of cleanings','Cleaning Duration':'CIP used Duration',
                               })
    metrics_cip['Available Hours']= working_hours
    metrics_cip['CIP idle time'] = (metrics_cip['Available Hours'] - metrics_cip['CIP used Duration']).round(1)
    metrics_cip['CIP idle time%'] = ((metrics_cip['CIP idle time']/(metrics_cip['CIP used Duration']+metrics_cip['Setup time']+metrics_cip['CIP idle time']))*100).round(1)
    metrics_cip['CIP Usage%'] = (((metrics_cip['CIP used Duration']+metrics_cip['Setup time'])/(metrics_cip['CIP used Duration']+metrics_cip['Setup time']+metrics_cip['CIP idle time']))*100).round(1)

    metrics_cip_total = metrics_cip.groupby(['MC Group'], as_index = False).agg({
                                                   'Resources cleaned': 'sum',
                                                    'No. of cleanings': 'sum', #No of cleanings
                                                   'CIP used Duration':'sum',#CIP used duration
                                                    'Setup time':'sum',
                                                    'Available Hours':'max'#CIP setup duration
                                                  })
    metrics_cip_total['CIP idle time'] = (metrics_cip_total['Available Hours'] - metrics_cip_total['Setup time']-metrics_cip_total['CIP used Duration']).round(1)
    metrics_cip_total['No. of possible cleanings'] = (metrics_cip_total['CIP idle time']/(metrics_cip_total['CIP used Duration']/metrics_cip_total['No. of cleanings'])).apply(np.floor)
    metrics_cip_total['CIP idle time%'] = ((metrics_cip_total['CIP idle time']/(metrics_cip_total['CIP used Duration']+metrics_cip_total['Setup time']+metrics_cip_total['CIP idle time']))*100).round(1)
    metrics_cip_total['CIP Usage%'] = (((metrics_cip_total['CIP used Duration']+metrics_cip_total['Setup time'])/(metrics_cip_total['CIP used Duration']+metrics_cip_total['Setup time']+metrics_cip_total['CIP idle time']))*100).round(1)
    df1=pd.DataFrame()
    for i in dataset['MC Group'].unique():
        temp_data=dataset[dataset['MC Group']==i]
        temp_data.sort_values('Clean_Start_Time',inplace=True)
        temp_data['Idle_Time']=temp_data['Clean_Start_Time']-temp_data['Clean_End_Time'].shift(1)
        temp_data['Idle_Time']=temp_data['Idle_Time']/(np.timedelta64(1, 's')*3600)
        inter_arrival_time=temp_data['Idle_Time'].mean()
        df1=df1.append({'MC Group':i,'Inter_Arrival_Time_Model':inter_arrival_time},ignore_index=True)
        df1['Min Downtime between 2 CIP cleanings']=df1['Inter_Arrival_Time_Model'].round(2)
        df1 = df1.dropna()
    metrics_cip_total = pd.merge(metrics_cip_total,df1,on=['MC Group'])
    return metrics_cip,metrics_cip_total
#Summary
def summary():
    dataset=get_data()
    metrics_cip = dataset.groupby(['MC Group','Constraint'], as_index = False).agg({
                                                   'Resource': pd.Series.nunique,
                                                    'Parallel Clean Flag':pd.Series.nunique
                                                    })
    pl=metrics_cip['Parallel Clean Flag'].sum()
    dataset['Immediate Assignment']=np.where((dataset['timebw']<0.166667),1,0)
    immediateCIP = round((dataset[dataset['Immediate Assignment']==1].shape[0]/dataset.shape[0])*100,2)
    dataset['MC Group'] = dataset['MC Group'].fillna(1)
    dataset['Missed Cleanings'] = np.where(dataset['MC Group']==1,1,0)
    summary = pd.DataFrame([['Cleaning Adjusted to avoid DHT Violation',dataset['Push Assignment'].sum()],
                            ['Cleaning Adjusted to avoid CHT Violation',dataset['CHT Adjusted'].sum()],
                            ['Utilization Shifted to Avoid Reclean',dataset['Utilization Shifted'].sum()],
                            ['No of Parallel Cleaning to avoid extra cleanings',pl],
                            ['% of the times Resource got CIP immediately',immediateCIP]
                             ],columns = ['Insights', 'No.'])
    return summary
