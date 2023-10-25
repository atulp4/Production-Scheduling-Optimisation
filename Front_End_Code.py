#!/usr/bin/env python
# coding: utf-8

# import sys
# 
# !{sys.executable} -m pip install keyring artifacts-keyring
# !{sys.executable} -m pip install --pre --upgrade --trusted-host pkgs.dev.azure.com --trusted-host pypi.org --trusted-host "*.blob.core.windows.net" --trusted-host files.pythonhosted.org --extra-index-url https://pkgs.dev.azure.com/mathco-products/_packaging/pip-codex-wf%40Local/pypi/simple/ "codex-widget-factory<=0.1rc0"
# 
# 

# In[1]:


# tags to identify this iteration when submitted
# example: codex_tags = {'env': 'dev', 'region': 'USA', 'product_category': 'A'}

codex_tags = {
}

from codex_widget_factory import utils
results_json=[]
import pandas as pd
import numpy as np
import plotly.figure_factory as ff
import plotly


# # Utilization Timeline Code String #

# In[2]:


filter_code_string_Resource='''
import pandas as pd
from pathlib import Path
from azure.storage.blob import BlockBlobService
from io import StringIO
import json

from azure.cosmosdb.table.tableservice import TableService
from pathlib import Path
from azure.storage.blob import BlockBlobService
from io import StringIO
from datetime import datetime, timedelta
import os
import pandas as pd
import json
from dateutil import parser
# from azure.identity import DefaultAzureCredential


key_value = 's94v9Q0+SXPPRagBKJUfB2WOA2Ts5mFPUb4sUg+C0tRzagJel6eYeeyrA7Dlz7s2mmPorbtOJQohLgjefo++Vw=='
connection_string = 'DefaultEndpointsProtocol=https;AccountName=mathcotakedastorage;AccountKey=s94v9Q0+SXPPRagBKJUfB2WOA2Ts5mFPUb4sUg+C0tRzagJel6eYeeyrA7Dlz7s2mmPorbtOJQohLgjefo++Vw==;EndpointSuffix=core.windows.net'
accountname = 'mathcotakedastorage'
data_source = 'azure_blob_storage'

def get_ingested_data(file_path, datasource_type, connection_uri, container_name):
    path = Path(file_path)
    block_blob_service = BlockBlobService(connection_string=connection_uri)
    blob_data = block_blob_service.get_blob_to_text(container_name=container_name, 
                                                    blob_name=file_path)
    ingested_df = pd.read_csv(StringIO(blob_data.content))
    return ingested_df

block_blob_service = BlockBlobService(connection_string=connection_string)
blob_data = block_blob_service.get_blob_to_text(container_name='input1', blob_name='Scheduler_Input/schedule_input.json')
f = pd.read_json(StringIO(blob_data.content))
#Start Date UTC and round handling
Start_Date = f['data']['Start Date'][0:19]
Start_Date=parser.parse(datetime.strptime(Start_Date ,'%Y-%m-%dT%H:%M:%S').strftime('%Y-%m-%dT%H:%M:%S'))
Start_Date=datetime.strftime(Start_Date ,'%Y-%m-%d')
Start_Date=parser.parse(datetime.strptime(Start_Date ,'%Y-%m-%d').strftime('%Y-%m-%d'))
#End Date UTC and round handling
End_Date = f['data']['End Date'][0:19]
End_Date=parser.parse(datetime.strptime(End_Date ,'%Y-%m-%dT%H:%M:%S').strftime('%Y-%m-%dT%H:%M:%S'))
End_Date=datetime.strftime(End_Date ,'%Y-%m-%d')
End_Date=parser.parse(datetime.strptime(End_Date ,'%Y-%m-%d').strftime('%Y-%m-%d'))+timedelta(hours=23.99)
# All the metadata is stored into table
table_service= TableService(account_name='mathcotakedastorage', account_key='s94v9Q0+SXPPRagBKJUfB2WOA2Ts5mFPUb4sUg+C0tRzagJel6eYeeyrA7Dlz7s2mmPorbtOJQohLgjefo++Vw==')
tasks=table_service.query_entities('schedule', f"PartitionKey eq 'schedule' and type eq 'Planned'")
table=pd.DataFrame(tasks)
# Converting start date and end date in metadata df to datetime
table['start_at'] = pd.to_datetime(table['start_at'],utc=True).dt.date
table['end_at'] = pd.to_datetime(table['end_at'],utc=True).dt.date
table['start_at']=pd.to_datetime(table['start_at'])
table['end_at']=pd.to_datetime(table['end_at'])
# Query df such that we get file path of data with schedule date between the given date on UI
# path eg = 'Src1/300921_Schedule/2021-09-04T04-10/output/20210930_2021-09-04T04-10_Optimized_Schedule_Output_(R).csv'
if f['data']['use_latest_schedule']==True:
    queried_table = table[(table['start_at'] <= datetime.strftime(Start_Date,'%Y-%m-%d')) & (table['end_at'] >= datetime.strftime(End_Date ,'%Y-%m-%d'))]
    queried_table = queried_table.tail(1)
    queried_table.reset_index(inplace = True)
    queried_path = queried_table['result_file'][0]
    queried_path = queried_path[6:76] + '_Optimized_Schedule_Output_(R).csv'
    condition1=[(queried_table['start_at'] == datetime.strftime(Start_Date,'%Y-%m-%d')) & (queried_table['end_at'] == datetime.strftime(End_Date ,'%Y-%m-%d'))][0]
    condition2=[(queried_table['start_at'] < datetime.strftime(Start_Date,'%Y-%m-%d')) | (queried_table['end_at'] > datetime.strftime(End_Date ,'%Y-%m-%d'))][0]
    if condition1[0]==True:
      schedule = get_ingested_data(file_path=queried_path,datasource_type=data_source,connection_uri=connection_string,container_name='input')
    elif condition2[0]==True:
      temp_schedule = queried_path
      temp_schedule = get_ingested_data(file_path=queried_path,datasource_type=data_source,connection_uri=connection_string,container_name='input')
      temp_schedule['Usage_Start'] = pd.to_datetime(temp_schedule['Usage_Start'])
      temp_schedule['Usage_End'] = pd.to_datetime(temp_schedule['Usage_End'])
      temp_schedule.sort_values(by='Usage_Start')
      schedule = temp_schedule[(temp_schedule['Usage_Start']>=Start_Date) & (temp_schedule['Usage_End']<=End_Date)]
    else:
      temp_schedule = get_ingested_data(file_path='merged_output_repository/merged_outputs.csv',datasource_type=data_source,connection_uri=connection_string,container_name='input')
      temp_schedule['Usage_Start'] = pd.to_datetime(temp_schedule['Usage_Start'])
      temp_schedule['Usage_End'] = pd.to_datetime(temp_schedule['Usage_End'])
      schedule = temp_schedule[(temp_schedule['Usage_Start']>=Start_Date) & (temp_schedule['Usage_End']<=End_Date)]
else:
    schedule =  get_ingested_data(file_path='merged_output_repository/latest_output.csv',datasource_type=data_source,connection_uri=connection_string,container_name='input')
schedule=schedule.reset_index(drop=True)

def get_filter_options(df= pd.DataFrame(), on_col="", using_cols={}, default_val='All'):
    """gets the filter options from a column based on a set of dependent columns
    and the values provided against these columns

    Args:
        df ([type]): DataFrame to be searched in.
        on_col (str): Column to be searched in.
        using_cols (dict, optional): the unique values to be searched in column 'on' will be based on the options provided
        in this dict, Its keys will be columns and values will be list of values. Defaults to {}.
        default_val (str, optional): a default value to be passed when no filter options are present. Defaults to 'All'
        if default_val == ''/None, Nothing will be added to dict

    Returns:
        [list]: list of options for the filter based on the data passed in using_cols param
    """
    cols_in_df = list(df.columns)

    filter_options = []
    if on_col in cols_in_df:
        if len(using_cols) == 0:
            filter_options = list(df[on_col].unique())
        else:
            for key in using_cols.keys():
                if key in cols_in_df:
                    if type(using_cols[key]) == str or type(using_cols[key]) == int :
                        if using_cols[key] == 'All':
                            continue
                        else:
                            df = df[df[key] == using_cols[key]]
                    else:
                        df = df[df[key].isin(using_cols[key])]
                else:
                    continue
        filter_options = list(df[on_col].unique())
    if default_val:
        filter_options.insert(0, default_val)
    return filter_options

# Reference
list_of_cols = ['Resource','Date']
using_cols_for_test = {}


def_options_for_filters= {}
for x in list_of_cols:
       #\print('------------------'+x+'---------------------------')
    if x == 'Resource':
        def_options_for_filters[x]=get_filter_options(schedule, on_col=x, using_cols=using_cols_for_test)
    elif x== 'Date':
        def_options_for_filters[x]={'start_date':str(pd.to_datetime(schedule['Clean_Start_Time'],infer_datetime_format=True,utc=True).min().isoformat()).replace("+00:00", ""), 'end_date': str(pd.to_datetime(schedule['Clean_End_Time'],infer_datetime_format=True,utc=True).max().isoformat()).replace("+00:00", "")}
    else:
        def_options_for_filters[x]={}



fil = {
    'Task': {
        'index': 0,
        'label': 'Resource',
        'type': 'multiple',
        'options': def_options_for_filters['Resource']
    },
    'Date':{
        'index':1,
        'label':'Date Range',
        'type':'date_range',
        'options':[]
    }
}



def generate_filter_json(current_filter_params,filter_options=fil, default_values=def_options_for_filters):
    basic_filter_dict = {
        "widget_filter_index": 0,
        "widget_filter_function": False,
        "widget_filter_function_parameter": False,
        "widget_filter_hierarchy_key": False,
        "widget_filter_isall": False,
        "widget_filter_multiselect": False,
        "widget_tag_input_type": "select",
        "widget_tag_key": "",
        "widget_tag_label": "",
        "widget_tag_value": [],
        "widget_filter_type": "",
        "widget_filter_params":None
    }
    dataValues = []
    defaultValues = {}



    for filter in filter_options.keys():
        instance_dict = dict(basic_filter_dict)
        instance_dict['widget_tag_key'] = filter
        instance_dict['widget_filter_index'] = filter_options[filter]['index']
        instance_dict['widget_tag_label'] = filter_options[filter]['label']
        instance_dict['widget_tag_value'] = filter_options[filter]['options']
        instance_dict['widget_filter_multiselect'] = True if filter_options[filter]['type'] == 'multiple' else False
        dataValues.append(instance_dict)
        if filter_options[filter]['type']=='date_range':
            instance_dict['widget_filter_type']='date_range'
            instance_dict['widget_filter_params']={'start_date':{'format':"DD/MM/yyyy", 'suppressUTC': True},'end_date':{'format':"DD/MM/yyyy", 'suppressUTC': True}}
        if current_filter_params=={}:
          defaultValues[filter] = default_values.get(
            filter, ['All'] if instance_dict['widget_filter_multiselect'] else 'All')
        else:
          defaultValues[filter] = current_filter_params['selected'].get(
            filter, ['All'] if instance_dict['widget_filter_multiselect'] else 'All')
    final_json = {'dataValues': dataValues, 'defaultValues': defaultValues}
    return final_json

dynamic_outputs=json.dumps(generate_filter_json(current_filter_params,filter_options=fil, default_values=def_options_for_filters))

'''


# In[3]:


code_string_Resource ="""
import pandas as pd
from pathlib import Path
from azure.storage.blob import BlockBlobService
from io import StringIO
import json
import numpy as np
import plotly.figure_factory as ff
import plotly


from azure.cosmosdb.table.tableservice import TableService
from pathlib import Path
from azure.storage.blob import BlockBlobService
from io import StringIO
from datetime import datetime, timedelta
import os
import pandas as pd
import json
from dateutil import parser
# from azure.identity import DefaultAzureCredential

key_value = 's94v9Q0+SXPPRagBKJUfB2WOA2Ts5mFPUb4sUg+C0tRzagJel6eYeeyrA7Dlz7s2mmPorbtOJQohLgjefo++Vw=='
connection_string = 'DefaultEndpointsProtocol=https;AccountName=mathcotakedastorage;AccountKey=s94v9Q0+SXPPRagBKJUfB2WOA2Ts5mFPUb4sUg+C0tRzagJel6eYeeyrA7Dlz7s2mmPorbtOJQohLgjefo++Vw==;EndpointSuffix=core.windows.net'
accountname = 'mathcotakedastorage'
data_source = 'azure_blob_storage'

def get_ingested_data(file_path, datasource_type, connection_uri, container_name):
    path = Path(file_path)
    block_blob_service = BlockBlobService(connection_string=connection_uri)
    blob_data = block_blob_service.get_blob_to_text(container_name=container_name, 
                                                    blob_name=file_path)
    ingested_df = pd.read_csv(StringIO(blob_data.content))
    return ingested_df

block_blob_service = BlockBlobService(connection_string=connection_string)
blob_data = block_blob_service.get_blob_to_text(container_name='input1', blob_name='Scheduler_Input/schedule_input.json')
f = pd.read_json(StringIO(blob_data.content))
#Start Date UTC and round handling
Start_Date = f['data']['Start Date'][0:19]
Start_Date=parser.parse(datetime.strptime(Start_Date ,'%Y-%m-%dT%H:%M:%S').strftime('%Y-%m-%dT%H:%M:%S'))
Start_Date=datetime.strftime(Start_Date ,'%Y-%m-%d')
Start_Date=parser.parse(datetime.strptime(Start_Date ,'%Y-%m-%d').strftime('%Y-%m-%d'))
#End Date UTC and round handling
End_Date = f['data']['End Date'][0:19]
End_Date=parser.parse(datetime.strptime(End_Date ,'%Y-%m-%dT%H:%M:%S').strftime('%Y-%m-%dT%H:%M:%S'))
End_Date=datetime.strftime(End_Date ,'%Y-%m-%d')
End_Date=parser.parse(datetime.strptime(End_Date ,'%Y-%m-%d').strftime('%Y-%m-%d'))+timedelta(hours=23.99)
# All the metadata is stored into table
table_service= TableService(account_name='mathcotakedastorage', account_key='s94v9Q0+SXPPRagBKJUfB2WOA2Ts5mFPUb4sUg+C0tRzagJel6eYeeyrA7Dlz7s2mmPorbtOJQohLgjefo++Vw==')
tasks=table_service.query_entities('schedule', f"PartitionKey eq 'schedule' and type eq 'Planned'")
table=pd.DataFrame(tasks)
# Converting start date and end date in metadata df to datetime
table['start_at'] = pd.to_datetime(table['start_at'],utc=True).dt.date
table['end_at'] = pd.to_datetime(table['end_at'],utc=True).dt.date
table['start_at']=pd.to_datetime(table['start_at'])
table['end_at']=pd.to_datetime(table['end_at'])
# Query df such that we get file path of data with schedule date between the given date on UI
# path eg = 'Src1/300921_Schedule/2021-09-04T04-10/output/20210930_2021-09-04T04-10_Optimized_Schedule_Output_(R).csv'
if f['data']['use_latest_schedule']==True:
    queried_table = table[(table['start_at'] <= datetime.strftime(Start_Date,'%Y-%m-%d')) & (table['end_at'] >= datetime.strftime(End_Date ,'%Y-%m-%d'))]
    queried_table = queried_table.tail(1)
    queried_table.reset_index(inplace = True)
    queried_path = queried_table['result_file'][0]
    queried_path = queried_path[6:76] + '_Optimized_Schedule_Output_(R).csv'
    condition1=[(queried_table['start_at'] == datetime.strftime(Start_Date,'%Y-%m-%d')) & (queried_table['end_at'] == datetime.strftime(End_Date ,'%Y-%m-%d'))][0]
    condition2=[(queried_table['start_at'] < datetime.strftime(Start_Date,'%Y-%m-%d')) | (queried_table['end_at'] > datetime.strftime(End_Date ,'%Y-%m-%d'))][0]
    if condition1[0]==True:
      schedule = get_ingested_data(file_path=queried_path,datasource_type=data_source,connection_uri=connection_string,container_name='input')
    elif condition2[0]==True:
      temp_schedule = queried_path
      temp_schedule = get_ingested_data(file_path=queried_path,datasource_type=data_source,connection_uri=connection_string,container_name='input')
      temp_schedule['Usage_Start'] = pd.to_datetime(temp_schedule['Usage_Start'])
      temp_schedule['Usage_End'] = pd.to_datetime(temp_schedule['Usage_End'])
      temp_schedule.sort_values(by='Usage_Start')
      schedule = temp_schedule[(temp_schedule['Usage_Start']>=Start_Date) & (temp_schedule['Usage_End']<=End_Date)]
    else:
      temp_schedule = get_ingested_data(file_path='merged_output_repository/merged_outputs.csv',datasource_type=data_source,connection_uri=connection_string,container_name='input')
      temp_schedule['Usage_Start'] = pd.to_datetime(temp_schedule['Usage_Start'])
      temp_schedule['Usage_End'] = pd.to_datetime(temp_schedule['Usage_End'])
      schedule = temp_schedule[(temp_schedule['Usage_Start']>=Start_Date) & (temp_schedule['Usage_End']<=End_Date)]
else:
    schedule =  get_ingested_data(file_path='merged_output_repository/latest_output.csv',datasource_type=data_source,connection_uri=connection_string,container_name='input')
schedule=schedule.reset_index(drop=True)



def filter_data(selected_filter, df):
    cols_in_df = list(df.columns)
    # Filtering data
    if len(selected_filters) > 0:
        for key in selected_filters.keys():
            if type(selected_filters[key])==dict and selected_filters[key].get('start_date',False) and selected_filters[key].get('end_date',False):
                selected_filters[key]['start_date']=pd.to_datetime(selected_filters[key]['start_date'],infer_datetime_format=True,utc=True)
                selected_filters[key]['end_date']=pd.to_datetime(selected_filters[key]['end_date'],infer_datetime_format=True,utc=True)
                print(selected_filters)
                df=df[(df['Start']>selected_filters[key]['start_date']) & (df['Finish']<selected_filters[key]['end_date'])]
            elif key in cols_in_df:
                if type(selected_filter[key]) == str or type(selected_filter[key]) == int:
                    if selected_filter[key] == 'All':
                        continue
                    else:
                        df = df[df[key] == selected_filter[key]]
                else:
                    if isinstance(selected_filter[key],list) and 'All' in selected_filter[key]:
                        continue
                    else:
                        df = df[df[key].isin(selected_filter[key])]
            else:
                continue
    return df



util_features=['Resource','Usage_Start','Usage_End','Cleaning Type','MC Group','Cleaning Duration']
clean_features=['Resource','Clean_Start_Time','Clean_End_Time','Cleaning Type','MC Group','Cleaning Duration']
util_schedule=schedule[util_features]
util_schedule=util_schedule.rename(columns={'Resource':'Task','Usage_Start':'Start','Usage_End':'Finish'})
options=['Pre-Cleaning','Re-Cleaning']
util_schedule['Resource']=np.where((util_schedule['MC Group'].isnull()) & (util_schedule['Cleaning Type'].isin(options)),'Utilization Post CHT violation','Utilization')
util_schedule['Resource']=np.where((util_schedule['Cleaning Type']=='Cleaning') & (util_schedule['Cleaning Duration'].isnull()) & (util_schedule['Resource']=='Utilization'),'Cleaning Post DHT violation',util_schedule['Resource'])
clean_schedule=schedule[clean_features]
clean_schedule=clean_schedule.rename(columns={'Resource':'Task','Clean_Start_Time':'Start','Clean_End_Time':'Finish','Cleaning Type':'Resource'})
final_schedule=pd.concat([util_schedule,clean_schedule])
final_schedule['Start']= pd.to_datetime(final_schedule['Start'],infer_datetime_format=True,utc=True)
final_schedule['Finish']= pd.to_datetime(final_schedule['Finish'],infer_datetime_format=True,utc=True)
final_schedule=final_schedule.rename(columns={'MC Group':'Description'})
final_schedule=filter_data(selected_filters,final_schedule)
sorter=['P1','P2','P3','P4','SA','SB','SC','SD','Short Line - L717S','AS26 TD Line - L2445','Factor Line - L777','AS16 Line - L2103','AS26 VII Line - L2464','Long Transfer Line - L717L','4F Line - L2437','Dome 121','Dome 2215','Dome 89','615','616','617','I/N 1619','I/N 2668','I/N 97','I/N 2315','I/N 2667','I/N 2224','Flexibles - Cryo KIT A','Flexibles - Cryo KIT B',
'Flexibles - Cryo KIT Colorati 1','Flexibles - Cryo KIT Colorati 2','Flexibles - Mass Capture 1/2','Flexibles - Mass Capture 3/4','Flexibles - Factor IX/FEIBA','Flexibles - AS16','Flexibles - AS26 VII','I/N 2316','I/N 1554','Gauthier Filter','T50','T51','T52','T53','T54','T55','T58','T59',
'T60','T61','T62','T63','T64','T65','Flexibles - Kit 4F','Flexibles - Kit UF/DF', 'Flexibles - Kit Colonna','Flexibles - Kit Eluato','Flexibles - Kit Lavaggio','Flexibles - Kit Manifold','Flexibles - DDCPP']
final_schedule['Task'] = pd.Categorical(final_schedule['Task'], categories = sorter)
final_schedule=final_schedule.sort_values('Task').reset_index(drop= True)
colors = {'Pre-Cleaning': 'rgb(153, 0, 255)',
          'Utilization': 'rgb(0, 255, 100)',
          'Cleaning':(1, 0.9, 0.16),
          'Recleaning':'rgb(0, 0, 220)',
          'Cleaning Post DHT violation':'rgb(220, 0, 0)',
           'Utilization Post CHT violation':'rgb(255, 140, 0)',
           'Maintenance':'rgb(100, 50, 100)'}
fig = ff.create_gantt(final_schedule, colors=colors, index_col='Resource', show_colorbar=True,
                      group_tasks=True)
fig.update_xaxes(
    tickformat="%d-%b-%Y %H:%M")

fig.layout.xaxis.rangeselector = None
dynamic_outputs = plotly.io.to_json(fig)
fig.show()
"""


# In[4]:


filter_code_string_CIP='''
import pandas as pd
from pathlib import Path
from azure.storage.blob import BlockBlobService
from io import StringIO
import json

from azure.cosmosdb.table.tableservice import TableService
from pathlib import Path
from azure.storage.blob import BlockBlobService
from io import StringIO
from datetime import datetime, timedelta
import os
import pandas as pd
import json
from dateutil import parser
# from azure.identity import DefaultAzureCredential


data_source = 'azure_blob_storage'
connection_string = 'DefaultEndpointsProtocol=https;AccountName=mathcotakedastorage;AccountKey=s94v9Q0+SXPPRagBKJUfB2WOA2Ts5mFPUb4sUg+C0tRzagJel6eYeeyrA7Dlz7s2mmPorbtOJQohLgjefo++Vw==;EndpointSuffix=core.windows.net'

def get_ingested_data(file_path, datasource_type, connection_uri, container_name):
    path = Path(file_path)
    block_blob_service = BlockBlobService(connection_string=connection_uri)
    blob_data = block_blob_service.get_blob_to_text(container_name=container_name, 
                                                    blob_name=file_path)
    ingested_df = pd.read_csv(StringIO(blob_data.content))

    return ingested_df

block_blob_service = BlockBlobService(connection_string=connection_string)
blob_data = block_blob_service.get_blob_to_text(container_name='input1', blob_name='Scheduler_Input/schedule_input.json')
f = pd.read_json(StringIO(blob_data.content))
#Start Date UTC and round handling
Start_Date = f['data']['Start Date'][0:19]
Start_Date=parser.parse(datetime.strptime(Start_Date ,'%Y-%m-%dT%H:%M:%S').strftime('%Y-%m-%dT%H:%M:%S'))
Start_Date=datetime.strftime(Start_Date ,'%Y-%m-%d')
Start_Date=parser.parse(datetime.strptime(Start_Date ,'%Y-%m-%d').strftime('%Y-%m-%d'))
#End Date UTC and round handling
End_Date = f['data']['End Date'][0:19]
End_Date=parser.parse(datetime.strptime(End_Date ,'%Y-%m-%dT%H:%M:%S').strftime('%Y-%m-%dT%H:%M:%S'))
End_Date=datetime.strftime(End_Date ,'%Y-%m-%d')
End_Date=parser.parse(datetime.strptime(End_Date ,'%Y-%m-%d').strftime('%Y-%m-%d'))+timedelta(hours=23.99)
# All the metadata is stored into table
table_service= TableService(account_name='mathcotakedastorage', account_key='s94v9Q0+SXPPRagBKJUfB2WOA2Ts5mFPUb4sUg+C0tRzagJel6eYeeyrA7Dlz7s2mmPorbtOJQohLgjefo++Vw==')
tasks=table_service.query_entities('schedule', f"PartitionKey eq 'schedule' and type eq 'Planned'")
table=pd.DataFrame(tasks)
# Converting start date and end date in metadata df to datetime
table['start_at'] = pd.to_datetime(table['start_at'],utc=True).dt.date
table['end_at'] = pd.to_datetime(table['end_at'],utc=True).dt.date
table['start_at']=pd.to_datetime(table['start_at'])
table['end_at']=pd.to_datetime(table['end_at'])
# Query df such that we get file path of data with schedule date between the given date on UI
# path eg = 'Src1/300921_Schedule/2021-09-04T04-10/output/20210930_2021-09-04T04-10_Optimized_Schedule_Output_(R).csv'
if f['data']['use_latest_schedule']==True:
    queried_table = table[(table['start_at'] <= datetime.strftime(Start_Date,'%Y-%m-%d')) & (table['end_at'] >= datetime.strftime(End_Date ,'%Y-%m-%d'))]
    queried_table = queried_table.tail(1)
    queried_table.reset_index(inplace = True)
    queried_path = queried_table['result_file'][0]
    queried_path = queried_path[6:76] + '_Optimized_Schedule_Output_(R).csv'
    condition1=[(queried_table['start_at'] == datetime.strftime(Start_Date,'%Y-%m-%d')) & (queried_table['end_at'] == datetime.strftime(End_Date ,'%Y-%m-%d'))][0]
    condition2=[(queried_table['start_at'] < datetime.strftime(Start_Date,'%Y-%m-%d')) | (queried_table['end_at'] > datetime.strftime(End_Date ,'%Y-%m-%d'))][0]
    if condition1[0]==True:
      schedule = get_ingested_data(file_path=queried_path,datasource_type=data_source,connection_uri=connection_string,container_name='input')
    elif condition2[0]==True:
      temp_schedule = queried_path
      temp_schedule = get_ingested_data(file_path=queried_path,datasource_type=data_source,connection_uri=connection_string,container_name='input')
      temp_schedule['Usage_Start'] = pd.to_datetime(temp_schedule['Usage_Start'])
      temp_schedule['Usage_End'] = pd.to_datetime(temp_schedule['Usage_End'])
      temp_schedule.sort_values(by='Usage_Start')
      schedule = temp_schedule[(temp_schedule['Usage_Start']>=Start_Date) & (temp_schedule['Usage_End']<=End_Date)]
    else:
      temp_schedule = get_ingested_data(file_path='merged_output_repository/merged_outputs.csv',datasource_type=data_source,connection_uri=connection_string,container_name='input')
      temp_schedule['Usage_Start'] = pd.to_datetime(temp_schedule['Usage_Start'])
      temp_schedule['Usage_End'] = pd.to_datetime(temp_schedule['Usage_End'])
      schedule = temp_schedule[(temp_schedule['Usage_Start']>=Start_Date) & (temp_schedule['Usage_End']<=End_Date)]
else:
    schedule =  get_ingested_data(file_path='merged_output_repository/latest_output.csv',datasource_type=data_source,connection_uri=connection_string,container_name='input')
schedule=schedule.reset_index(drop=True)


def get_filter_options(df= pd.DataFrame(), on_col="", using_cols={}, default_val='All'):
    """gets the filter options from a column based on a set of dependent columns
    and the values provided against these columns

    Args:
        df ([type]): DataFrame to be searched in.
        on_col (str): Column to be searched in.
        using_cols (dict, optional): the unique values to be searched in column 'on' will be based on the options provided
        in this dict, Its keys will be columns and values will be list of values. Defaults to {}.
        default_val (str, optional): a default value to be passed when no filter options are present. Defaults to 'All'
        if default_val == ''/None, Nothing will be added to dict

    Returns:
        [list]: list of options for the filter based on the data passed in using_cols param
    """
    cols_in_df = list(df.columns)

    filter_options = []
    if on_col in cols_in_df:
        if len(using_cols) == 0:
            filter_options = list(df[on_col].unique())
        else:
            for key in using_cols.keys():
                if key in cols_in_df:
                    if type(using_cols[key]) == str or type(using_cols[key]) == int :
                        if using_cols[key] == 'All':
                            continue
                        else:
                            df = df[df[key] == using_cols[key]]
                    else:
                        df = df[df[key].isin(using_cols[key])]
                else:
                    continue
        filter_options = list(df[on_col].unique())
    if default_val:
        filter_options.insert(0, default_val)
    return filter_options

# Reference
list_of_cols = ['MC Group','Date']
using_cols_for_test = {}

schedule.dropna(subset = ["MC Group"], inplace=True)
def_options_for_filters= {}
for x in list_of_cols:
       #\print('------------------'+x+'---------------------------')
    if x == 'MC Group':
        def_options_for_filters[x]=get_filter_options(schedule, on_col=x, using_cols=using_cols_for_test)
    elif x== 'Date':
        def_options_for_filters[x]={'start_date':str(pd.to_datetime(schedule['Clean_Start_Time'],infer_datetime_format=True,utc=True).min().isoformat()).replace("+00:00", ""), 'end_date': str(pd.to_datetime(schedule['Clean_End_Time'],infer_datetime_format=True,utc=True).max().isoformat()).replace("+00:00", "")}
    else:
        def_options_for_filters[x]={}


fil = {
    'Task': {
        'index': 0,
        'label': 'MC Group',
        'type': 'multiple',
        'options': def_options_for_filters['MC Group']
    },
    'Date':{
        'index':1,
        'label':'Date Range',
        'type':'date_range',
        'options':[]
    }
}



def generate_filter_json(current_filter_params,filter_options=fil, default_values=def_options_for_filters):
    basic_filter_dict = {
        "widget_filter_index": 0,
        "widget_filter_function": False,
        "widget_filter_function_parameter": False,
        "widget_filter_hierarchy_key": False,
        "widget_filter_isall": False,
        "widget_filter_multiselect": False,
        "widget_tag_input_type": "select",
        "widget_tag_key": "",
        "widget_tag_label": "",
        "widget_tag_value": [],
        "widget_filter_type": "",
        "widget_filter_params":None
    }
    dataValues = []
    defaultValues = {}

    for filter in filter_options.keys():
        instance_dict = dict(basic_filter_dict)
        instance_dict['widget_tag_key'] = filter
        instance_dict['widget_filter_index'] = filter_options[filter]['index']
        instance_dict['widget_tag_label'] = filter_options[filter]['label']
        instance_dict['widget_tag_value'] = filter_options[filter]['options']
        instance_dict['widget_filter_multiselect'] = True if filter_options[filter]['type'] == 'multiple' else False
        dataValues.append(instance_dict)
        if filter_options[filter]['type']=='date_range':
            instance_dict['widget_filter_type']='date_range'
            instance_dict['widget_filter_params']={'start_date':{'format':"DD/MM/yyyy", 'suppressUTC': True},'end_date':{'format':"DD/MM/yyyy", 'suppressUTC': True}}
        if current_filter_params=={}:
          defaultValues[filter] = default_values.get(
            filter, ['All'] if instance_dict['widget_filter_multiselect'] else 'All')
        else:
          defaultValues[filter] = current_filter_params['selected'].get(
            filter, ['All'] if instance_dict['widget_filter_multiselect'] else 'All')
    final_json = {'dataValues': dataValues, 'defaultValues': defaultValues}
    return final_json

dynamic_outputs=json.dumps(generate_filter_json(current_filter_params,filter_options=fil, default_values=def_options_for_filters))

'''


# In[5]:


code_string_CIP = """
import pandas as pd
from pathlib import Path
from azure.storage.blob import BlockBlobService
from io import StringIO
import json
import numpy as np
import plotly.figure_factory as ff
import plotly

from azure.cosmosdb.table.tableservice import TableService
from pathlib import Path
from azure.storage.blob import BlockBlobService
from io import StringIO
from datetime import datetime, timedelta
import os
import pandas as pd
import json
from dateutil import parser
# from azure.identity import DefaultAzureCredential

key_value = 's94v9Q0+SXPPRagBKJUfB2WOA2Ts5mFPUb4sUg+C0tRzagJel6eYeeyrA7Dlz7s2mmPorbtOJQohLgjefo++Vw=='
connection_string = 'DefaultEndpointsProtocol=https;AccountName=mathcotakedastorage;AccountKey=s94v9Q0+SXPPRagBKJUfB2WOA2Ts5mFPUb4sUg+C0tRzagJel6eYeeyrA7Dlz7s2mmPorbtOJQohLgjefo++Vw==;EndpointSuffix=core.windows.net'
accountname = 'mathcotakedastorage'
data_source = 'azure_blob_storage'

def get_ingested_data(file_path, datasource_type, connection_uri, container_name):
    path = Path(file_path)
    block_blob_service = BlockBlobService(connection_string=connection_uri)
    blob_data = block_blob_service.get_blob_to_text(container_name=container_name, 
                                                    blob_name=file_path)
    ingested_df = pd.read_csv(StringIO(blob_data.content))
    return ingested_df

block_blob_service = BlockBlobService(connection_string=connection_string)
blob_data = block_blob_service.get_blob_to_text(container_name='input1', blob_name='Scheduler_Input/schedule_input.json')
f = pd.read_json(StringIO(blob_data.content))
#Start Date UTC and round handling
Start_Date = f['data']['Start Date'][0:19]
Start_Date=parser.parse(datetime.strptime(Start_Date ,'%Y-%m-%dT%H:%M:%S').strftime('%Y-%m-%dT%H:%M:%S'))
Start_Date=datetime.strftime(Start_Date ,'%Y-%m-%d')
Start_Date=parser.parse(datetime.strptime(Start_Date ,'%Y-%m-%d').strftime('%Y-%m-%d'))
#End Date UTC and round handling
End_Date = f['data']['End Date'][0:19]
End_Date=parser.parse(datetime.strptime(End_Date ,'%Y-%m-%dT%H:%M:%S').strftime('%Y-%m-%dT%H:%M:%S'))
End_Date=datetime.strftime(End_Date ,'%Y-%m-%d')
End_Date=parser.parse(datetime.strptime(End_Date ,'%Y-%m-%d').strftime('%Y-%m-%d'))+timedelta(hours=23.99)
# All the metadata is stored into table
table_service= TableService(account_name='mathcotakedastorage', account_key='s94v9Q0+SXPPRagBKJUfB2WOA2Ts5mFPUb4sUg+C0tRzagJel6eYeeyrA7Dlz7s2mmPorbtOJQohLgjefo++Vw==')
tasks=table_service.query_entities('schedule', f"PartitionKey eq 'schedule' and type eq 'Planned'")
table=pd.DataFrame(tasks)
# Converting start date and end date in metadata df to datetime
table['start_at'] = pd.to_datetime(table['start_at'],utc=True).dt.date
table['end_at'] = pd.to_datetime(table['end_at'],utc=True).dt.date
table['start_at']=pd.to_datetime(table['start_at'])
table['end_at']=pd.to_datetime(table['end_at'])
# Query df such that we get file path of data with schedule date between the given date on UI
# path eg = 'Src1/300921_Schedule/2021-09-04T04-10/output/20210930_2021-09-04T04-10_Optimized_Schedule_Output_(R).csv'
if f['data']['use_latest_schedule']==True:
    queried_table = table[(table['start_at'] <= datetime.strftime(Start_Date,'%Y-%m-%d')) & (table['end_at'] >= datetime.strftime(End_Date ,'%Y-%m-%d'))]
    queried_table = queried_table.tail(1)
    queried_table.reset_index(inplace = True)
    queried_path = queried_table['result_file'][0]
    queried_path = queried_path[6:76] + '_Optimized_Schedule_Output_(R).csv'
    condition1=[(queried_table['start_at'] == datetime.strftime(Start_Date,'%Y-%m-%d')) & (queried_table['end_at'] == datetime.strftime(End_Date ,'%Y-%m-%d'))][0]
    condition2=[(queried_table['start_at'] < datetime.strftime(Start_Date,'%Y-%m-%d')) | (queried_table['end_at'] > datetime.strftime(End_Date ,'%Y-%m-%d'))][0]
    if condition1[0]==True:
      schedule = get_ingested_data(file_path=queried_path,datasource_type=data_source,connection_uri=connection_string,container_name='input')
    elif condition2[0]==True:
      temp_schedule = queried_path
      temp_schedule = get_ingested_data(file_path=queried_path,datasource_type=data_source,connection_uri=connection_string,container_name='input')
      temp_schedule['Usage_Start'] = pd.to_datetime(temp_schedule['Usage_Start'])
      temp_schedule['Usage_End'] = pd.to_datetime(temp_schedule['Usage_End'])
      temp_schedule.sort_values(by='Usage_Start')
      schedule = temp_schedule[(temp_schedule['Usage_Start']>=Start_Date) & (temp_schedule['Usage_End']<=End_Date)]
    else:
      temp_schedule = get_ingested_data(file_path='merged_output_repository/merged_outputs.csv',datasource_type=data_source,connection_uri=connection_string,container_name='input')
      temp_schedule['Usage_Start'] = pd.to_datetime(temp_schedule['Usage_Start'])
      temp_schedule['Usage_End'] = pd.to_datetime(temp_schedule['Usage_End'])
      schedule = temp_schedule[(temp_schedule['Usage_Start']>=Start_Date) & (temp_schedule['Usage_End']<=End_Date)]
else:
    schedule =  get_ingested_data(file_path='merged_output_repository/latest_output.csv',datasource_type=data_source,connection_uri=connection_string,container_name='input')
schedule=schedule.reset_index(drop=True)

def filter_data(selected_filter, df):
    cols_in_df = list(df.columns)
    # Filtering data
    if len(selected_filters) > 0:
        for key in selected_filters.keys():
            if type(selected_filters[key])==dict and selected_filters[key].get('start_date',False) and selected_filters[key].get('end_date',False):
                selected_filters[key]['start_date']=pd.to_datetime(selected_filters[key]['start_date'],infer_datetime_format=True,utc=True)
                selected_filters[key]['end_date']=pd.to_datetime(selected_filters[key]['end_date'],infer_datetime_format=True,utc=True)
                print(selected_filters)
                df=df[(df['Start']>selected_filters[key]['start_date']) & (df['Finish']<selected_filters[key]['end_date'])]
            elif key in cols_in_df:
                if type(selected_filter[key]) == str or type(selected_filter[key]) == int:
                    if selected_filter[key] == 'All':
                        continue
                    else:
                        df = df[df[key] == selected_filter[key]]
                else:
                    if isinstance(selected_filter[key],list) and 'All' in selected_filter[key]:
                        continue
                    else:
                        df = df[df[key].isin(selected_filter[key])]
            else:
                continue
    return df


clean_features=['Clean_Start_Time','Clean_End_Time','Cleaning Type','MC Group','CHT Violation(Intermediate Gap)','preclean','Resource']
clean_schedule=schedule[clean_features]
clean_schedule=clean_schedule.rename(columns={'Resource':'Description','MC Group':'Task','Clean_Start_Time':'Start','Clean_End_Time':'Finish','Cleaning Type':'Resource'})
clean_schedule.dropna(subset = ["Start"], inplace=True)
clean_schedule['Resource']=np.where((clean_schedule['Resource']=='Pre-Cleaning') & (clean_schedule['CHT Violation(Intermediate Gap)']==1) & (clean_schedule['preclean']==1),'Recleaning',clean_schedule['Resource'])
clean_schedule['Resource']=np.where((clean_schedule['Resource']=='Pre-Cleaning') & (clean_schedule['CHT Violation(Intermediate Gap)']==0) & (clean_schedule['preclean']==1),'Pre-Cleaning',clean_schedule['Resource'])
clean_schedule['Start']= pd.to_datetime(clean_schedule['Start'],infer_datetime_format=True,utc=True)
clean_schedule['Finish']= pd.to_datetime(clean_schedule['Finish'],infer_datetime_format=True,utc=True)
clean_schedule=filter_data(selected_filters,clean_schedule)
clean_schedule=clean_schedule.sort_values('Start').reset_index(drop= True)
import plotly.figure_factory as ff
colors = {'Pre-Cleaning': (1, 0.9, 0.16),
          'Cleaning':'rgb(0, 255, 100)',
          'Parallel Cleaning':'rgb(0, 0, 220)',
           'Recleaning':'rgb(153, 0, 255)',
           'Maintenance':'rgb(100, 50, 100)'}

fig = ff.create_gantt(clean_schedule, colors=colors, index_col='Resource', show_colorbar=True,
                      group_tasks=True)
fig.update_xaxes(tickformat="%d-%b-%Y %H:%M")
fig.layout.title = None
fig.layout.xaxis.rangeselector = None
dynamic_outputs = plotly.io.to_json(fig)
fig.show()
"""


# # CIP Summary Code String #

# In[6]:


kpi_cip_utilization="""

'''
Libraries & Data Import
'''

import pandas as pd
from pathlib import Path
from azure.storage.blob import BlockBlobService
from io import StringIO
import json
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly


from azure.cosmosdb.table.tableservice import TableService
from pathlib import Path
from azure.storage.blob import BlockBlobService
from io import StringIO
from datetime import datetime, timedelta
import os
import pandas as pd
import json
from dateutil import parser
# from azure.identity import DefaultAzureCredential

key_value = 's94v9Q0+SXPPRagBKJUfB2WOA2Ts5mFPUb4sUg+C0tRzagJel6eYeeyrA7Dlz7s2mmPorbtOJQohLgjefo++Vw=='
connection_string = 'DefaultEndpointsProtocol=https;AccountName=mathcotakedastorage;AccountKey=s94v9Q0+SXPPRagBKJUfB2WOA2Ts5mFPUb4sUg+C0tRzagJel6eYeeyrA7Dlz7s2mmPorbtOJQohLgjefo++Vw==;EndpointSuffix=core.windows.net'
accountname = 'mathcotakedastorage'
data_source = 'azure_blob_storage'

def get_ingested_data(file_path, datasource_type, connection_uri, container_name):
    path = Path(file_path)
    block_blob_service = BlockBlobService(connection_string=connection_uri)
    blob_data = block_blob_service.get_blob_to_text(container_name=container_name, 
                                                    blob_name=file_path)
    ingested_df = pd.read_csv(StringIO(blob_data.content))
    return ingested_df
block_blob_service = BlockBlobService(connection_string=connection_string)
blob_data = block_blob_service.get_blob_to_text(container_name='input1', blob_name='Scheduler_Input/schedule_input.json')
f = pd.read_json(StringIO(blob_data.content))

#Start Date UTC and round handling
Start_Date = f['data']['Start Date'][0:19]
Start_Date=parser.parse(datetime.strptime(Start_Date ,'%Y-%m-%dT%H:%M:%S').strftime('%Y-%m-%dT%H:%M:%S'))
Start_Date=datetime.strftime(Start_Date ,'%Y-%m-%d')
Start_Date=parser.parse(datetime.strptime(Start_Date ,'%Y-%m-%d').strftime('%Y-%m-%d'))
#End Date UTC and round handling
End_Date = f['data']['End Date'][0:19]
End_Date=parser.parse(datetime.strptime(End_Date ,'%Y-%m-%dT%H:%M:%S').strftime('%Y-%m-%dT%H:%M:%S'))
End_Date=datetime.strftime(End_Date ,'%Y-%m-%d')
End_Date=parser.parse(datetime.strptime(End_Date ,'%Y-%m-%d').strftime('%Y-%m-%d'))+timedelta(hours=23.99)
# All the metadata is stored into table
table_service= TableService(account_name='mathcotakedastorage', account_key='s94v9Q0+SXPPRagBKJUfB2WOA2Ts5mFPUb4sUg+C0tRzagJel6eYeeyrA7Dlz7s2mmPorbtOJQohLgjefo++Vw==')
tasks=table_service.query_entities('schedule', f"PartitionKey eq 'schedule' and type eq 'Planned'")
table=pd.DataFrame(tasks)
# Converting start date and end date in metadata df to datetime
table['start_at'] = pd.to_datetime(table['start_at'],utc=True).dt.date
table['end_at'] = pd.to_datetime(table['end_at'],utc=True).dt.date
table['start_at']=pd.to_datetime(table['start_at'])
table['end_at']=pd.to_datetime(table['end_at'])
# Query df such that we get file path of data with schedule date between the given date on UI
# path eg = 'Src1/300921_Schedule/2021-09-04T04-10/output/20210930_2021-09-04T04-10_Optimized_Schedule_Output_(R).csv'
if f['data']['use_latest_schedule']==True:
    queried_table = table[(table['start_at'] <= datetime.strftime(Start_Date,'%Y-%m-%d')) & (table['end_at'] >= datetime.strftime(End_Date ,'%Y-%m-%d'))]
    queried_table = queried_table.tail(1)
    queried_table.reset_index(inplace = True)
    queried_path = queried_table['result_file'][0]
    queried_path = queried_path[6:76] + '_Optimized_Schedule_Output_(R).csv'
    condition1=[(queried_table['start_at'] == datetime.strftime(Start_Date,'%Y-%m-%d')) & (queried_table['end_at'] == datetime.strftime(End_Date ,'%Y-%m-%d'))][0]
    condition2=[(queried_table['start_at'] < datetime.strftime(Start_Date,'%Y-%m-%d')) | (queried_table['end_at'] > datetime.strftime(End_Date ,'%Y-%m-%d'))][0]
    if condition1[0]==True:
      schedule = get_ingested_data(file_path=queried_path,datasource_type=data_source,connection_uri=connection_string,container_name='input')
    elif condition2[0]==True:
      temp_schedule = queried_path
      temp_schedule = get_ingested_data(file_path=queried_path,datasource_type=data_source,connection_uri=connection_string,container_name='input')
      temp_schedule['Usage_Start'] = pd.to_datetime(temp_schedule['Usage_Start'])
      temp_schedule['Usage_End'] = pd.to_datetime(temp_schedule['Usage_End'])
      temp_schedule.sort_values(by='Usage_Start')
      schedule = temp_schedule[(temp_schedule['Usage_Start']>=Start_Date) & (temp_schedule['Usage_End']<=End_Date)]
    else:
      temp_schedule = get_ingested_data(file_path='merged_output_repository/merged_outputs.csv',datasource_type=data_source,connection_uri=connection_string,container_name='input')
      temp_schedule['Usage_Start'] = pd.to_datetime(temp_schedule['Usage_Start'])
      temp_schedule['Usage_End'] = pd.to_datetime(temp_schedule['Usage_End'])
      schedule = temp_schedule[(temp_schedule['Usage_Start']>=Start_Date) & (temp_schedule['Usage_End']<=End_Date)]
else:
    schedule =  get_ingested_data(file_path='merged_output_repository/latest_output.csv',datasource_type=data_source,connection_uri=connection_string,container_name='input')

dataset=schedule.reset_index(drop=True)
dataset['Clean_Start_Time']=pd.to_datetime(dataset['Clean_Start_Time'],format='%Y/%m/%d %H:%M:%S')
dataset['Usage_End']=pd.to_datetime(dataset['Usage_End'],format='%Y/%m/%d %H:%M:%S')
dataset['Clean_End_Time']=pd.to_datetime(dataset['Clean_End_Time'],format='%Y/%m/%d %H:%M:%S')
dataset['Usage_Start']=pd.to_datetime(dataset['Usage_Start'],format='%Y/%m/%d %H:%M:%S')
if dataset['Clean_Start_Time'].min() > dataset['Usage_Start'].min():
    working_hours = (dataset['Clean_End_Time'].max()-dataset['Usage_Start'].min())/ (np.timedelta64(1, 'h'))
else:
    working_hours = (dataset['Clean_End_Time'].max()-dataset['Clean_Start_Time'].min())/ (np.timedelta64(1, 'h'))
dataset['timebw'] = dataset['Clean_Start_Time'] - dataset['Usage_End']
dataset['timebw'] = dataset['timebw'] / np.timedelta64(1, 'h')
dataset['setup_flag'] = np.where(dataset['timebw'] > 0.166 , 1, 0)
dataset['setup_flag'] = dataset['setup_flag']*10
dataset = dataset.rename(columns = {'setup_flag': 'Setup time'})
dataset['Setup time'] = (dataset['Setup time']/60).round(1)
dataset['Cleaning Duration']=dataset['Cleaning Duration']/60
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
average_cip=metrics_cip_total['CIP Usage%'].mean().round(0)


#1. CIP USAGE %
fig_cip = px.bar(metrics_cip_total,
             y=metrics_cip_total['CIP Usage%'],
             x=metrics_cip_total['MC Group'], text=metrics_cip_total['CIP Usage%'],title='CIP USAGE %').update_xaxes(categoryorder="total descending")
fig_cip.update_traces(texttemplate='%{text:0f}',textposition='outside',width=0.3,cliponaxis = False)
fig_cip.update_layout(xaxis_title='Machine',
                  yaxis_title='CIP Utilisation %')
fig_cip.show()
average_cip_kpi= {
  "value":str(int(average_cip))+" %",
  "extra_dir":"down" if average_cip>90 else "",
  "is_kpi":True
}
print(average_cip_kpi)

average_cip_graph=json.loads(plotly.io.to_json(fig_cip))
print(average_cip_graph)

utilization_kpi= {
  "front":{
    "data":{
      "value":average_cip_kpi
      }
  },
  "back":{
    "data":{
      "value":average_cip_graph
    }
  },
  "is_flip":True
}

dynamic_outputs=json.dumps(utilization_kpi)
print(dynamic_outputs)
"""


# In[7]:


kpi_recleanings="""
import pandas as pd
from pathlib import Path
from azure.storage.blob import BlockBlobService
from io import StringIO
import json
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly



from azure.cosmosdb.table.tableservice import TableService
from pathlib import Path
from azure.storage.blob import BlockBlobService
from io import StringIO
from datetime import datetime, timedelta
import os
import pandas as pd
import json
from dateutil import parser
# from azure.identity import DefaultAzureCredential

key_value = 's94v9Q0+SXPPRagBKJUfB2WOA2Ts5mFPUb4sUg+C0tRzagJel6eYeeyrA7Dlz7s2mmPorbtOJQohLgjefo++Vw=='
connection_string = 'DefaultEndpointsProtocol=https;AccountName=mathcotakedastorage;AccountKey=s94v9Q0+SXPPRagBKJUfB2WOA2Ts5mFPUb4sUg+C0tRzagJel6eYeeyrA7Dlz7s2mmPorbtOJQohLgjefo++Vw==;EndpointSuffix=core.windows.net'
accountname = 'mathcotakedastorage'
data_source = 'azure_blob_storage'

def get_ingested_data(file_path, datasource_type, connection_uri, container_name):
    path = Path(file_path)
    block_blob_service = BlockBlobService(connection_string=connection_uri)
    blob_data = block_blob_service.get_blob_to_text(container_name=container_name, 
                                                    blob_name=file_path)
    ingested_df = pd.read_csv(StringIO(blob_data.content))
    return ingested_df
block_blob_service = BlockBlobService(connection_string=connection_string)
blob_data = block_blob_service.get_blob_to_text(container_name='input1', blob_name='Scheduler_Input/schedule_input.json')
f = pd.read_json(StringIO(blob_data.content))

#Start Date UTC and round handling
Start_Date = f['data']['Start Date'][0:19]
Start_Date=parser.parse(datetime.strptime(Start_Date ,'%Y-%m-%dT%H:%M:%S').strftime('%Y-%m-%dT%H:%M:%S'))
Start_Date=datetime.strftime(Start_Date ,'%Y-%m-%d')
Start_Date=parser.parse(datetime.strptime(Start_Date ,'%Y-%m-%d').strftime('%Y-%m-%d'))
#End Date UTC and round handling
End_Date = f['data']['End Date'][0:19]
End_Date=parser.parse(datetime.strptime(End_Date ,'%Y-%m-%dT%H:%M:%S').strftime('%Y-%m-%dT%H:%M:%S'))
End_Date=datetime.strftime(End_Date ,'%Y-%m-%d')
End_Date=parser.parse(datetime.strptime(End_Date ,'%Y-%m-%d').strftime('%Y-%m-%d'))+timedelta(hours=23.99)
# All the metadata is stored into table
table_service= TableService(account_name='mathcotakedastorage', account_key='s94v9Q0+SXPPRagBKJUfB2WOA2Ts5mFPUb4sUg+C0tRzagJel6eYeeyrA7Dlz7s2mmPorbtOJQohLgjefo++Vw==')
tasks=table_service.query_entities('schedule', f"PartitionKey eq 'schedule' and type eq 'Planned'")
table=pd.DataFrame(tasks)
# Converting start date and end date in metadata df to datetime
table['start_at'] = pd.to_datetime(table['start_at'],utc=True).dt.date
table['end_at'] = pd.to_datetime(table['end_at'],utc=True).dt.date
table['start_at']=pd.to_datetime(table['start_at'])
table['end_at']=pd.to_datetime(table['end_at'])
# Query df such that we get file path of data with schedule date between the given date on UI
# path eg = 'Src1/300921_Schedule/2021-09-04T04-10/output/20210930_2021-09-04T04-10_Optimized_Schedule_Output_(R).csv'
if f['data']['use_latest_schedule']==True:
    queried_table = table[(table['start_at'] <= datetime.strftime(Start_Date,'%Y-%m-%d')) & (table['end_at'] >= datetime.strftime(End_Date ,'%Y-%m-%d'))]
    queried_table = queried_table.tail(1)
    queried_table.reset_index(inplace = True)
    queried_path = queried_table['result_file'][0]
    queried_path = queried_path[6:76] + '_Optimized_Schedule_Output_(R).csv'
    condition1=[(queried_table['start_at'] == datetime.strftime(Start_Date,'%Y-%m-%d')) & (queried_table['end_at'] == datetime.strftime(End_Date ,'%Y-%m-%d'))][0]
    condition2=[(queried_table['start_at'] < datetime.strftime(Start_Date,'%Y-%m-%d')) | (queried_table['end_at'] > datetime.strftime(End_Date ,'%Y-%m-%d'))][0]
    if condition1[0]==True:
      schedule = get_ingested_data(file_path=queried_path,datasource_type=data_source,connection_uri=connection_string,container_name='input')
    elif condition2[0]==True:
      temp_schedule = queried_path
      temp_schedule = get_ingested_data(file_path=queried_path,datasource_type=data_source,connection_uri=connection_string,container_name='input')
      temp_schedule['Usage_Start'] = pd.to_datetime(temp_schedule['Usage_Start'])
      temp_schedule['Usage_End'] = pd.to_datetime(temp_schedule['Usage_End'])
      temp_schedule.sort_values(by='Usage_Start')
      schedule = temp_schedule[(temp_schedule['Usage_Start']>=Start_Date) & (temp_schedule['Usage_End']<=End_Date)]
    else:
      temp_schedule = get_ingested_data(file_path='merged_output_repository/merged_outputs.csv',datasource_type=data_source,connection_uri=connection_string,container_name='input')
      temp_schedule['Usage_Start'] = pd.to_datetime(temp_schedule['Usage_Start'])
      temp_schedule['Usage_End'] = pd.to_datetime(temp_schedule['Usage_End'])
      schedule = temp_schedule[(temp_schedule['Usage_Start']>=Start_Date) & (temp_schedule['Usage_End']<=End_Date)]
else:
    schedule =  get_ingested_data(file_path='merged_output_repository/latest_output.csv',datasource_type=data_source,connection_uri=connection_string,container_name='input')

dataset=schedule.reset_index(drop=True)
dataset['Clean_Start_Time']=pd.to_datetime(dataset['Clean_Start_Time'],format='%Y/%m/%d %H:%M:%S')
dataset['Usage_End']=pd.to_datetime(dataset['Usage_End'],format='%Y/%m/%d %H:%M:%S')
dataset['Clean_End_Time']=pd.to_datetime(dataset['Clean_End_Time'],format='%Y/%m/%d %H:%M:%S')
dataset['Usage_Start']=pd.to_datetime(dataset['Usage_Start'],format='%Y/%m/%d %H:%M:%S')
dataset['MC Group'] = dataset['MC Group'].fillna(1)
dataset['Missed ReCleanings'] = np.where((dataset['MC Group']==1) & (dataset['Cleaning Type']=='Pre-Cleaning'),1,0)
dataset['Missed Cleanings'] = np.where((dataset['MC Group']==1) & (dataset['Cleaning Type']=='Cleaning'),1,0)
type=['Pre-Cleaning','Recleaning']
m=dataset[(dataset['Cleaning Type'].isin(type)) & (dataset['MC Group']!=1)]

recleanings=len(m)
recleanings
# def convert(o):
#     if isinstance(o, np.generic): return o.item()  
#     raise TypeError
# recleanings=convert(recleanings)
m=m.groupby('Cleaning Type',as_index=False).agg({'Resource':'count'})

fig_pie = px.pie(m, values='Resource',names='Cleaning Type')
fig_pie.update_traces(textinfo='value')
fig_pie.update_traces(textposition='inside', textfont=dict(color="white"),textfont_family='Bevan')
fig_pie.show()

recleanings_kpi= {
  "value":recleanings,
  "is_kpi":True
}

print(recleanings_kpi)

recleanings_graph=json.loads(plotly.io.to_json(fig_pie))

recleanings_kpi= {
  "front":{
    "data":{
      "value":recleanings_kpi
      }
  },
  "back":{
    "data":{
      "value":recleanings_graph
    }
  },
  "is_flip":True
}

dynamic_outputs=json.dumps(recleanings_kpi)
print(dynamic_outputs)
"""




# In[8]:


kpi_dht_violations="""
import pandas as pd
from pathlib import Path
from azure.storage.blob import BlockBlobService
from io import StringIO
import json
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly


from azure.cosmosdb.table.tableservice import TableService
from pathlib import Path
from azure.storage.blob import BlockBlobService
from io import StringIO
from datetime import datetime, timedelta
import os
import pandas as pd
import json
from dateutil import parser
# from azure.identity import DefaultAzureCredential

key_value = 's94v9Q0+SXPPRagBKJUfB2WOA2Ts5mFPUb4sUg+C0tRzagJel6eYeeyrA7Dlz7s2mmPorbtOJQohLgjefo++Vw=='
connection_string = 'DefaultEndpointsProtocol=https;AccountName=mathcotakedastorage;AccountKey=s94v9Q0+SXPPRagBKJUfB2WOA2Ts5mFPUb4sUg+C0tRzagJel6eYeeyrA7Dlz7s2mmPorbtOJQohLgjefo++Vw==;EndpointSuffix=core.windows.net'
accountname = 'mathcotakedastorage'
data_source = 'azure_blob_storage'

def get_ingested_data(file_path, datasource_type, connection_uri, container_name):
    path = Path(file_path)
    block_blob_service = BlockBlobService(connection_string=connection_uri)
    blob_data = block_blob_service.get_blob_to_text(container_name=container_name, 
                                                    blob_name=file_path)
    ingested_df = pd.read_csv(StringIO(blob_data.content))
    return ingested_df
block_blob_service = BlockBlobService(connection_string=connection_string)
blob_data = block_blob_service.get_blob_to_text(container_name='input1', blob_name='Scheduler_Input/schedule_input.json')
f = pd.read_json(StringIO(blob_data.content))

#Start Date UTC and round handling
Start_Date = f['data']['Start Date'][0:19]
Start_Date=parser.parse(datetime.strptime(Start_Date ,'%Y-%m-%dT%H:%M:%S').strftime('%Y-%m-%dT%H:%M:%S'))
Start_Date=datetime.strftime(Start_Date ,'%Y-%m-%d')
Start_Date=parser.parse(datetime.strptime(Start_Date ,'%Y-%m-%d').strftime('%Y-%m-%d'))
#End Date UTC and round handling
End_Date = f['data']['End Date'][0:19]
End_Date=parser.parse(datetime.strptime(End_Date ,'%Y-%m-%dT%H:%M:%S').strftime('%Y-%m-%dT%H:%M:%S'))
End_Date=datetime.strftime(End_Date ,'%Y-%m-%d')
End_Date=parser.parse(datetime.strptime(End_Date ,'%Y-%m-%d').strftime('%Y-%m-%d'))+timedelta(hours=23.99)
# All the metadata is stored into table
table_service= TableService(account_name='mathcotakedastorage', account_key='s94v9Q0+SXPPRagBKJUfB2WOA2Ts5mFPUb4sUg+C0tRzagJel6eYeeyrA7Dlz7s2mmPorbtOJQohLgjefo++Vw==')
tasks=table_service.query_entities('schedule', f"PartitionKey eq 'schedule' and type eq 'Planned'")
table=pd.DataFrame(tasks)
# Converting start date and end date in metadata df to datetime
table['start_at'] = pd.to_datetime(table['start_at'],utc=True).dt.date
table['end_at'] = pd.to_datetime(table['end_at'],utc=True).dt.date
table['start_at']=pd.to_datetime(table['start_at'])
table['end_at']=pd.to_datetime(table['end_at'])
# Query df such that we get file path of data with schedule date between the given date on UI
# path eg = 'Src1/300921_Schedule/2021-09-04T04-10/output/20210930_2021-09-04T04-10_Optimized_Schedule_Output_(R).csv'
if f['data']['use_latest_schedule']==True:
    queried_table = table[(table['start_at'] <= datetime.strftime(Start_Date,'%Y-%m-%d')) & (table['end_at'] >= datetime.strftime(End_Date ,'%Y-%m-%d'))]
    queried_table = queried_table.tail(1)
    queried_table.reset_index(inplace = True)
    queried_path = queried_table['result_file'][0]
    queried_path = queried_path[6:76] + '_Optimized_Schedule_Output_(R).csv'
    condition1=[(queried_table['start_at'] == datetime.strftime(Start_Date,'%Y-%m-%d')) & (queried_table['end_at'] == datetime.strftime(End_Date ,'%Y-%m-%d'))][0]
    condition2=[(queried_table['start_at'] < datetime.strftime(Start_Date,'%Y-%m-%d')) | (queried_table['end_at'] > datetime.strftime(End_Date ,'%Y-%m-%d'))][0]
    if condition1[0]==True:
      schedule = get_ingested_data(file_path=queried_path,datasource_type=data_source,connection_uri=connection_string,container_name='input')
    elif condition2[0]==True:
      temp_schedule = queried_path
      temp_schedule = get_ingested_data(file_path=queried_path,datasource_type=data_source,connection_uri=connection_string,container_name='input')
      temp_schedule['Usage_Start'] = pd.to_datetime(temp_schedule['Usage_Start'])
      temp_schedule['Usage_End'] = pd.to_datetime(temp_schedule['Usage_End'])
      temp_schedule.sort_values(by='Usage_Start')
      schedule = temp_schedule[(temp_schedule['Usage_Start']>=Start_Date) & (temp_schedule['Usage_End']<=End_Date)]
    else:
      temp_schedule = get_ingested_data(file_path='merged_output_repository/merged_outputs.csv',datasource_type=data_source,connection_uri=connection_string,container_name='input')
      temp_schedule['Usage_Start'] = pd.to_datetime(temp_schedule['Usage_Start'])
      temp_schedule['Usage_End'] = pd.to_datetime(temp_schedule['Usage_End'])
      schedule = temp_schedule[(temp_schedule['Usage_Start']>=Start_Date) & (temp_schedule['Usage_End']<=End_Date)]
else:
    schedule =  get_ingested_data(file_path='merged_output_repository/latest_output.csv',datasource_type=data_source,connection_uri=connection_string,container_name='input')

dataset=schedule.reset_index(drop=True)
dataset['Clean_Start_Time']=pd.to_datetime(dataset['Clean_Start_Time'],format='%Y/%m/%d %H:%M:%S')
dataset['Usage_End']=pd.to_datetime(dataset['Usage_End'],format='%Y/%m/%d %H:%M:%S')
dataset['Clean_End_Time']=pd.to_datetime(dataset['Clean_End_Time'],format='%Y/%m/%d %H:%M:%S')
dataset['Usage_Start']=pd.to_datetime(dataset['Usage_Start'],format='%Y/%m/%d %H:%M:%S')
dht_violations=dataset['DHT Violation Flag'].sum()
def convert(o):
    if isinstance(o, np.generic): return o.item()  
    raise TypeError
dht_violations=convert(dht_violations)

dht= dataset.groupby('Resource',as_index = False).agg({
                                            'DHT Violation Flag':'sum'
})
dht=dht[dht['DHT Violation Flag']>0]
fig_dht = go.Figure(data=[
    go.Bar(name='Post Utilization CHT Violation', x=dht['Resource'], y=dht['DHT Violation Flag'],text=dht['DHT Violation Flag']),
])
# Change the bar mode
fig_dht.update_traces(texttemplate='%{text:0f}',textposition='outside',width=0.3,cliponaxis = False)
fig_dht.update_layout(title='DHT Violations')
fig_dht.update_layout(xaxis_title='Resource',
                  yaxis_title='# DHT Violations')
fig_dht.update_yaxes(nticks=5)
fig_dht.show()

dht_kpi_value= {
  "value":dht_violations,
  "extra_dir":"down" if dht_violations>0 else "",
  "suppress_arrow":True,
  "is_kpi":True
}


dht_graph=json.loads(plotly.io.to_json(fig_dht))

dht_kpi= {
  "front":{
    "data":{
      "value":dht_kpi_value
      }
  },
  "back":{
    "data":{
      "value":dht_graph
    }
  },
  "is_flip":True
}

dynamic_outputs=json.dumps(dht_kpi)
print(dynamic_outputs)


"""


# In[9]:


kpi_cht_violations="""
import pandas as pd
from pathlib import Path
from azure.storage.blob import BlockBlobService
from io import StringIO
import json
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly



from azure.cosmosdb.table.tableservice import TableService
from pathlib import Path
from azure.storage.blob import BlockBlobService
from io import StringIO
from datetime import datetime, timedelta
import os
import pandas as pd
import json
from dateutil import parser
# from azure.identity import DefaultAzureCredential

key_value = 's94v9Q0+SXPPRagBKJUfB2WOA2Ts5mFPUb4sUg+C0tRzagJel6eYeeyrA7Dlz7s2mmPorbtOJQohLgjefo++Vw=='
connection_string = 'DefaultEndpointsProtocol=https;AccountName=mathcotakedastorage;AccountKey=s94v9Q0+SXPPRagBKJUfB2WOA2Ts5mFPUb4sUg+C0tRzagJel6eYeeyrA7Dlz7s2mmPorbtOJQohLgjefo++Vw==;EndpointSuffix=core.windows.net'
accountname = 'mathcotakedastorage'
data_source = 'azure_blob_storage'

def get_ingested_data(file_path, datasource_type, connection_uri, container_name):
    path = Path(file_path)
    block_blob_service = BlockBlobService(connection_string=connection_uri)
    blob_data = block_blob_service.get_blob_to_text(container_name=container_name, 
                                                    blob_name=file_path)
    ingested_df = pd.read_csv(StringIO(blob_data.content))
    return ingested_df
block_blob_service = BlockBlobService(connection_string=connection_string)
blob_data = block_blob_service.get_blob_to_text(container_name='input1', blob_name='Scheduler_Input/schedule_input.json')
f = pd.read_json(StringIO(blob_data.content))

#Start Date UTC and round handling
Start_Date = f['data']['Start Date'][0:19]
Start_Date=parser.parse(datetime.strptime(Start_Date ,'%Y-%m-%dT%H:%M:%S').strftime('%Y-%m-%dT%H:%M:%S'))
Start_Date=datetime.strftime(Start_Date ,'%Y-%m-%d')
Start_Date=parser.parse(datetime.strptime(Start_Date ,'%Y-%m-%d').strftime('%Y-%m-%d'))
#End Date UTC and round handling
End_Date = f['data']['End Date'][0:19]
End_Date=parser.parse(datetime.strptime(End_Date ,'%Y-%m-%dT%H:%M:%S').strftime('%Y-%m-%dT%H:%M:%S'))
End_Date=datetime.strftime(End_Date ,'%Y-%m-%d')
End_Date=parser.parse(datetime.strptime(End_Date ,'%Y-%m-%d').strftime('%Y-%m-%d'))+timedelta(hours=23.99)
# All the metadata is stored into table
table_service= TableService(account_name='mathcotakedastorage', account_key='s94v9Q0+SXPPRagBKJUfB2WOA2Ts5mFPUb4sUg+C0tRzagJel6eYeeyrA7Dlz7s2mmPorbtOJQohLgjefo++Vw==')
tasks=table_service.query_entities('schedule', f"PartitionKey eq 'schedule' and type eq 'Planned'")
table=pd.DataFrame(tasks)
# Converting start date and end date in metadata df to datetime
table['start_at'] = pd.to_datetime(table['start_at'],utc=True).dt.date
table['end_at'] = pd.to_datetime(table['end_at'],utc=True).dt.date
table['start_at']=pd.to_datetime(table['start_at'])
table['end_at']=pd.to_datetime(table['end_at'])
# Query df such that we get file path of data with schedule date between the given date on UI
# path eg = 'Src1/300921_Schedule/2021-09-04T04-10/output/20210930_2021-09-04T04-10_Optimized_Schedule_Output_(R).csv'
if f['data']['use_latest_schedule']==True:
    queried_table = table[(table['start_at'] <= datetime.strftime(Start_Date,'%Y-%m-%d')) & (table['end_at'] >= datetime.strftime(End_Date ,'%Y-%m-%d'))]
    queried_table = queried_table.tail(1)
    queried_table.reset_index(inplace = True)
    queried_path = queried_table['result_file'][0]
    queried_path = queried_path[6:76] + '_Optimized_Schedule_Output_(R).csv'
    condition1=[(queried_table['start_at'] == datetime.strftime(Start_Date,'%Y-%m-%d')) & (queried_table['end_at'] == datetime.strftime(End_Date ,'%Y-%m-%d'))][0]
    condition2=[(queried_table['start_at'] < datetime.strftime(Start_Date,'%Y-%m-%d')) | (queried_table['end_at'] > datetime.strftime(End_Date ,'%Y-%m-%d'))][0]
    if condition1[0]==True:
      schedule = get_ingested_data(file_path=queried_path,datasource_type=data_source,connection_uri=connection_string,container_name='input')
    elif condition2[0]==True:
      temp_schedule = queried_path
      temp_schedule = get_ingested_data(file_path=queried_path,datasource_type=data_source,connection_uri=connection_string,container_name='input')
      temp_schedule['Usage_Start'] = pd.to_datetime(temp_schedule['Usage_Start'])
      temp_schedule['Usage_End'] = pd.to_datetime(temp_schedule['Usage_End'])
      temp_schedule.sort_values(by='Usage_Start')
      schedule = temp_schedule[(temp_schedule['Usage_Start']>=Start_Date) & (temp_schedule['Usage_End']<=End_Date)]
    else:
      temp_schedule = get_ingested_data(file_path='merged_output_repository/merged_outputs.csv',datasource_type=data_source,connection_uri=connection_string,container_name='input')
      temp_schedule['Usage_Start'] = pd.to_datetime(temp_schedule['Usage_Start'])
      temp_schedule['Usage_End'] = pd.to_datetime(temp_schedule['Usage_End'])
      schedule = temp_schedule[(temp_schedule['Usage_Start']>=Start_Date) & (temp_schedule['Usage_End']<=End_Date)]
else:
    schedule =  get_ingested_data(file_path='merged_output_repository/latest_output.csv',datasource_type=data_source,connection_uri=connection_string,container_name='input')

dataset=schedule.reset_index(drop=True)
dataset['Clean_Start_Time']=pd.to_datetime(dataset['Clean_Start_Time'],format='%Y/%m/%d %H:%M:%S')
dataset['Usage_End']=pd.to_datetime(dataset['Usage_End'],format='%Y/%m/%d %H:%M:%S')
dataset['Clean_End_Time']=pd.to_datetime(dataset['Clean_End_Time'],format='%Y/%m/%d %H:%M:%S')
dataset['Usage_Start']=pd.to_datetime(dataset['Usage_Start'],format='%Y/%m/%d %H:%M:%S')
dataset['MC Group'] = dataset['MC Group'].fillna(1)
dataset['Missed Cleanings'] = np.where(dataset['MC Group']==1,1,0)

cht_violations=dataset['CHT Violation Flag'].sum()
def convert(o):
    if isinstance(o, np.generic): return o.item()  
    raise TypeError
cht_violations=convert(cht_violations)


cht= dataset.groupby('Resource',as_index = False).agg({
                                            'CHT Violation Flag':'sum'})
cht=cht[cht['CHT Violation Flag']>0]
if len(cht)==0:
    cht=0
    fig_cht={}
else:
  fig_cht = px.bar(cht,x=cht['Resource'],y=cht['CHT Violation Flag'],text=cht['CHT Violation Flag']).update_xaxes(categoryorder="total descending")
  # Change the bar mode
  fig_cht.update_traces(texttemplate='%{text:0f}',textposition='outside',width=0.3,cliponaxis = False)
  fig_cht.update_layout(title='# CHT Violations')
  fig_cht.update_layout(barmode='stack')
  fig_cht.update_layout(xaxis_title='Resource',
                  yaxis_title='# CHT Violations')
  fig_cht.update_yaxes(nticks=4)
  fig_cht.show()



cht_kpi_value= {
  "value":cht_violations,
  "is_kpi":True,
  "extra_dir":"down" if cht_violations>0 else "",
  "suppress_arrow":True,
}


cht_graph=json.loads(plotly.io.to_json(fig_cht))

cht_kpi= {
  "front":{
    "data":{
      "value":cht_kpi_value
      }
  },
  "back":{
    "data":{
      "value":cht_graph
    }
  },
  "is_flip":True
}

dynamic_outputs=json.dumps(cht_kpi)
print(dynamic_outputs)
"""


# In[10]:


kpi_cleaning_cycles="""
import pandas as pd
from pathlib import Path
from azure.storage.blob import BlockBlobService
from io import StringIO
import json
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly



from azure.cosmosdb.table.tableservice import TableService
from pathlib import Path
from azure.storage.blob import BlockBlobService
from io import StringIO
from datetime import datetime, timedelta
import os
import pandas as pd
import json
from dateutil import parser
# from azure.identity import DefaultAzureCredential

key_value = 's94v9Q0+SXPPRagBKJUfB2WOA2Ts5mFPUb4sUg+C0tRzagJel6eYeeyrA7Dlz7s2mmPorbtOJQohLgjefo++Vw=='
connection_string = 'DefaultEndpointsProtocol=https;AccountName=mathcotakedastorage;AccountKey=s94v9Q0+SXPPRagBKJUfB2WOA2Ts5mFPUb4sUg+C0tRzagJel6eYeeyrA7Dlz7s2mmPorbtOJQohLgjefo++Vw==;EndpointSuffix=core.windows.net'
accountname = 'mathcotakedastorage'
data_source = 'azure_blob_storage'

def get_ingested_data(file_path, datasource_type, connection_uri, container_name):
    path = Path(file_path)
    block_blob_service = BlockBlobService(connection_string=connection_uri)
    blob_data = block_blob_service.get_blob_to_text(container_name=container_name, 
                                                    blob_name=file_path)
    ingested_df = pd.read_csv(StringIO(blob_data.content))
    return ingested_df
block_blob_service = BlockBlobService(connection_string=connection_string)
blob_data = block_blob_service.get_blob_to_text(container_name='input1', blob_name='Scheduler_Input/schedule_input.json')
f = pd.read_json(StringIO(blob_data.content))

#Start Date UTC and round handling
Start_Date = f['data']['Start Date'][0:19]
Start_Date=parser.parse(datetime.strptime(Start_Date ,'%Y-%m-%dT%H:%M:%S').strftime('%Y-%m-%dT%H:%M:%S'))
Start_Date=datetime.strftime(Start_Date ,'%Y-%m-%d')
Start_Date=parser.parse(datetime.strptime(Start_Date ,'%Y-%m-%d').strftime('%Y-%m-%d'))
#End Date UTC and round handling
End_Date = f['data']['End Date'][0:19]
End_Date=parser.parse(datetime.strptime(End_Date ,'%Y-%m-%dT%H:%M:%S').strftime('%Y-%m-%dT%H:%M:%S'))
End_Date=datetime.strftime(End_Date ,'%Y-%m-%d')
End_Date=parser.parse(datetime.strptime(End_Date ,'%Y-%m-%d').strftime('%Y-%m-%d'))+timedelta(hours=23.99)
# All the metadata is stored into table
table_service= TableService(account_name='mathcotakedastorage', account_key='s94v9Q0+SXPPRagBKJUfB2WOA2Ts5mFPUb4sUg+C0tRzagJel6eYeeyrA7Dlz7s2mmPorbtOJQohLgjefo++Vw==')
tasks=table_service.query_entities('schedule', f"PartitionKey eq 'schedule' and type eq 'Planned'")
table=pd.DataFrame(tasks)
# Converting start date and end date in metadata df to datetime
table['start_at'] = pd.to_datetime(table['start_at'],utc=True).dt.date
table['end_at'] = pd.to_datetime(table['end_at'],utc=True).dt.date
table['start_at']=pd.to_datetime(table['start_at'])
table['end_at']=pd.to_datetime(table['end_at'])
# Query df such that we get file path of data with schedule date between the given date on UI
# path eg = 'Src1/300921_Schedule/2021-09-04T04-10/output/20210930_2021-09-04T04-10_Optimized_Schedule_Output_(R).csv'
if f['data']['use_latest_schedule']==True:
    queried_table = table[(table['start_at'] <= datetime.strftime(Start_Date,'%Y-%m-%d')) & (table['end_at'] >= datetime.strftime(End_Date ,'%Y-%m-%d'))]
    queried_table = queried_table.tail(1)
    queried_table.reset_index(inplace = True)
    queried_path = queried_table['result_file'][0]
    queried_path = queried_path[6:76] + '_Optimized_Schedule_Output_(R).csv'
    condition1=[(queried_table['start_at'] == datetime.strftime(Start_Date,'%Y-%m-%d')) & (queried_table['end_at'] == datetime.strftime(End_Date ,'%Y-%m-%d'))][0]
    condition2=[(queried_table['start_at'] < datetime.strftime(Start_Date,'%Y-%m-%d')) | (queried_table['end_at'] > datetime.strftime(End_Date ,'%Y-%m-%d'))][0]
    if condition1[0]==True:
      schedule = get_ingested_data(file_path=queried_path,datasource_type=data_source,connection_uri=connection_string,container_name='input')
    elif condition2[0]==True:
      temp_schedule = queried_path
      temp_schedule = get_ingested_data(file_path=queried_path,datasource_type=data_source,connection_uri=connection_string,container_name='input')
      temp_schedule['Usage_Start'] = pd.to_datetime(temp_schedule['Usage_Start'])
      temp_schedule['Usage_End'] = pd.to_datetime(temp_schedule['Usage_End'])
      temp_schedule.sort_values(by='Usage_Start')
      schedule = temp_schedule[(temp_schedule['Usage_Start']>=Start_Date) & (temp_schedule['Usage_End']<=End_Date)]
    else:
      temp_schedule = get_ingested_data(file_path='merged_output_repository/merged_outputs.csv',datasource_type=data_source,connection_uri=connection_string,container_name='input')
      temp_schedule['Usage_Start'] = pd.to_datetime(temp_schedule['Usage_Start'])
      temp_schedule['Usage_End'] = pd.to_datetime(temp_schedule['Usage_End'])
      schedule = temp_schedule[(temp_schedule['Usage_Start']>=Start_Date) & (temp_schedule['Usage_End']<=End_Date)]
else:
    schedule =  get_ingested_data(file_path='merged_output_repository/latest_output.csv',datasource_type=data_source,connection_uri=connection_string,container_name='input')

dataset=schedule.reset_index(drop=True)
dataset['Clean_Start_Time']=pd.to_datetime(dataset['Clean_Start_Time'],format='%Y/%m/%d %H:%M:%S')
dataset['Usage_End']=pd.to_datetime(dataset['Usage_End'],format='%Y/%m/%d %H:%M:%S')
dataset['Clean_End_Time']=pd.to_datetime(dataset['Clean_End_Time'],format='%Y/%m/%d %H:%M:%S')
dataset['Usage_Start']=pd.to_datetime(dataset['Usage_Start'],format='%Y/%m/%d %H:%M:%S')
dataset['MC Group'] = dataset['MC Group'].fillna(1)
dataset['Missed Cleanings'] = np.where(dataset['MC Group']==1,1,0)
def business_metrics_resource(dataset):
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
                                                            'Missed Cleanings':'sum'
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
    return bm
def business_metrics_CIP(dataset):
    if dataset['Clean_Start_Time'].min() > dataset['Usage_Start'].min():
        working_hours = (dataset['Clean_End_Time'].max()-dataset['Usage_Start'].min())/ (np.timedelta64(1, 'h'))
    else:
        working_hours = (dataset['Clean_End_Time'].max()-dataset['Clean_Start_Time'].min())/ (np.timedelta64(1, 'h'))
    dataset['timebw'] = dataset['Clean_Start_Time'] - dataset['Usage_End']
    dataset['timebw'] = dataset['timebw'] / np.timedelta64(1, 'h')
    dataset['setup_flag'] = np.where(dataset['timebw'] > 0.166 , 1, 0)
    dataset['setup_flag'] = dataset['setup_flag']*10
    dataset = dataset.rename(columns = {'setup_flag': 'Setup time'})
    dataset['Setup time'] = (dataset['Setup time']/60).round(1)
    metrics_cip = dataset.groupby(['MC Group','Constraint'], as_index = False).agg({
                                                   'Resource': pd.Series.nunique, 
                                                    'clean_flag': 'sum', 
                                                   'Cleaning Duration':'sum',
                                                    'Setup time':'sum',
                                                    'Parallel Clean Flag':pd.Series.nunique
                                                    })
    metrics_cip = metrics_cip.rename(columns = {'Resource': 'Resources cleaned',
                               'clean_flag':'No. of cleanings','Cleaning Duration':'CIP used Duration',
                               })
    metrics_cip['No. of cleanings'] = metrics_cip['No. of cleanings'] - metrics_cip['Parallel Clean Flag']
    metrics_cip['Available Hours']= working_hours
    metrics_cip['CIP idle time'] = (metrics_cip['Available Hours'] - metrics_cip['CIP used Duration']).round(1)
    metrics_cip['CIP idle time%'] = ((metrics_cip['CIP idle time']/(metrics_cip['CIP used Duration']+metrics_cip['Setup time']+metrics_cip['CIP idle time']))*100).round(1)
    metrics_cip['CIP Usage%'] = (((metrics_cip['CIP used Duration']+metrics_cip['Setup time'])/(metrics_cip['CIP used Duration']+metrics_cip['Setup time']+metrics_cip['CIP idle time']))*100).round(1)

    metrics_cip_total = metrics_cip.groupby(['MC Group'], as_index = False).agg({
                                                   'Resources cleaned': 'sum',
                                                    'No. of cleanings': 'sum', #No of cleanings
                                                   'CIP used Duration':'sum',#CIP used duration
                                                    'Setup time':'sum',
                                                    'Available Hours':'max',
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
        df1=df1.append({'MC Group':i,'Min Downtime between 2 CIP cleanings':inter_arrival_time},ignore_index=True)
        df1['Min Downtime between 2 CIP cleanings']=df1['Min Downtime between 2 CIP cleanings'].round(2)
        df1 = df1.dropna()
    metrics_cip_total = pd.merge(metrics_cip_total,df1,on=['MC Group'])
    return metrics_cip,metrics_cip_total
df_res = business_metrics_resource(dataset)
df_pipe,df_cip = business_metrics_CIP(dataset)
non_missed = dataset[dataset['MC Group'].isin(['MC1','MC2','MC3','MC4'])]
def pre_re_cleaning(dataframe):   
    df1 = dataframe[(dataframe['preclean']==1)]
    df1['Cleaning Duration'] = df1['Cleaning Duration']/60
    data_temp = pd.DataFrame()
    for res in dataframe.Resource.unique():
        temp_df = dataframe[dataframe['Resource']==res]
        y=temp_df.min()['Usage_Start']
        data_temp = data_temp.append({"Resource":res,'Min_Usage_Start':y},ignore_index=True)
    new = pd.merge(df1,data_temp,on=['Resource'])
    new['CHT 1st U']=np.where((new['Usage_Start']== new['Min_Usage_Start']),'PREclean','REclean')
    x = new[new['CHT 1st U']=='PREclean'].shape[0]
    y =new[new['CHT 1st U']=='REclean'].shape[0]
    pre_re = pd.DataFrame([['Total no. of Pre-cleanings ',x],['Total no. of Re-cleanings',y]],columns=['Insights','Count'])
    return pre_re,new 
m,new1 = pre_re_cleaning(non_missed)
pc = df_res['Parallel Cleanings'].sum()
cleaning_pie1=pd.DataFrame([['Cleaning',non_missed[non_missed['Cleaning Type']=='Cleaning'].shape[0]],
                     ['Pre-Cleaning',non_missed[non_missed['Cleaning Type']=='Pre-Cleaning'].shape[0]],
                     ['Re-Cleaning',non_missed[non_missed['Cleaning Type']=='Recleaning'].shape[0]]], 
                     columns = ['Cleaning Type','Count'])
fig_clean_cycles = px.pie(cleaning_pie1, values='Count', names='Cleaning Type', title='TOTAL NUMBER OF CLEANINGS')
fig_clean_cycles.update_traces(texttemplate="%{percent:0%f}",textfont=dict(color="white"))
fig_clean_cycles.show()

dynamic_outputs = plotly.io.to_json(fig_clean_cycles)
"""


# In[11]:


kpi_preclean="""
import pandas as pd
from pathlib import Path
from azure.storage.blob import BlockBlobService
from io import StringIO
import json
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly
# selected_filters=selected_filters={'MC Group':'All'}


from azure.cosmosdb.table.tableservice import TableService
from pathlib import Path
from azure.storage.blob import BlockBlobService
from io import StringIO
from datetime import datetime, timedelta
import os
import pandas as pd
import json
from dateutil import parser
# from azure.identity import DefaultAzureCredential

key_value = 's94v9Q0+SXPPRagBKJUfB2WOA2Ts5mFPUb4sUg+C0tRzagJel6eYeeyrA7Dlz7s2mmPorbtOJQohLgjefo++Vw=='
connection_string = 'DefaultEndpointsProtocol=https;AccountName=mathcotakedastorage;AccountKey=s94v9Q0+SXPPRagBKJUfB2WOA2Ts5mFPUb4sUg+C0tRzagJel6eYeeyrA7Dlz7s2mmPorbtOJQohLgjefo++Vw==;EndpointSuffix=core.windows.net'
accountname = 'mathcotakedastorage'
data_source = 'azure_blob_storage'

def get_ingested_data(file_path, datasource_type, connection_uri, container_name):
    path = Path(file_path)
    block_blob_service = BlockBlobService(connection_string=connection_uri)
    blob_data = block_blob_service.get_blob_to_text(container_name=container_name, 
                                                    blob_name=file_path)
    ingested_df = pd.read_csv(StringIO(blob_data.content))
    return ingested_df
block_blob_service = BlockBlobService(connection_string=connection_string)
blob_data = block_blob_service.get_blob_to_text(container_name='input1', blob_name='Scheduler_Input/schedule_input.json')
f = pd.read_json(StringIO(blob_data.content))

#Start Date UTC and round handling
Start_Date = f['data']['Start Date'][0:19]
Start_Date=parser.parse(datetime.strptime(Start_Date ,'%Y-%m-%dT%H:%M:%S').strftime('%Y-%m-%dT%H:%M:%S'))
Start_Date=datetime.strftime(Start_Date ,'%Y-%m-%d')
Start_Date=parser.parse(datetime.strptime(Start_Date ,'%Y-%m-%d').strftime('%Y-%m-%d'))
#End Date UTC and round handling
End_Date = f['data']['End Date'][0:19]
End_Date=parser.parse(datetime.strptime(End_Date ,'%Y-%m-%dT%H:%M:%S').strftime('%Y-%m-%dT%H:%M:%S'))
End_Date=datetime.strftime(End_Date ,'%Y-%m-%d')
End_Date=parser.parse(datetime.strptime(End_Date ,'%Y-%m-%d').strftime('%Y-%m-%d'))+timedelta(hours=23.99)
# All the metadata is stored into table
table_service= TableService(account_name='mathcotakedastorage', account_key='s94v9Q0+SXPPRagBKJUfB2WOA2Ts5mFPUb4sUg+C0tRzagJel6eYeeyrA7Dlz7s2mmPorbtOJQohLgjefo++Vw==')
tasks=table_service.query_entities('schedule', f"PartitionKey eq 'schedule' and type eq 'Planned'")
table=pd.DataFrame(tasks)
# Converting start date and end date in metadata df to datetime
table['start_at'] = pd.to_datetime(table['start_at'],utc=True).dt.date
table['end_at'] = pd.to_datetime(table['end_at'],utc=True).dt.date
table['start_at']=pd.to_datetime(table['start_at'])
table['end_at']=pd.to_datetime(table['end_at'])
# Query df such that we get file path of data with schedule date between the given date on UI
# path eg = 'Src1/300921_Schedule/2021-09-04T04-10/output/20210930_2021-09-04T04-10_Optimized_Schedule_Output_(R).csv'
if f['data']['use_latest_schedule']==True:
    queried_table = table[(table['start_at'] <= datetime.strftime(Start_Date,'%Y-%m-%d')) & (table['end_at'] >= datetime.strftime(End_Date ,'%Y-%m-%d'))]
    queried_table = queried_table.tail(1)
    queried_table.reset_index(inplace = True)
    queried_path = queried_table['result_file'][0]
    queried_path = queried_path[6:76] + '_Optimized_Schedule_Output_(R).csv'
    condition1=[(queried_table['start_at'] == datetime.strftime(Start_Date,'%Y-%m-%d')) & (queried_table['end_at'] == datetime.strftime(End_Date ,'%Y-%m-%d'))][0]
    condition2=[(queried_table['start_at'] < datetime.strftime(Start_Date,'%Y-%m-%d')) | (queried_table['end_at'] > datetime.strftime(End_Date ,'%Y-%m-%d'))][0]
    if condition1[0]==True:
      schedule = get_ingested_data(file_path=queried_path,datasource_type=data_source,connection_uri=connection_string,container_name='input')
    elif condition2[0]==True:
      temp_schedule = queried_path
      temp_schedule = get_ingested_data(file_path=queried_path,datasource_type=data_source,connection_uri=connection_string,container_name='input')
      temp_schedule['Usage_Start'] = pd.to_datetime(temp_schedule['Usage_Start'])
      temp_schedule['Usage_End'] = pd.to_datetime(temp_schedule['Usage_End'])
      temp_schedule.sort_values(by='Usage_Start')
      schedule = temp_schedule[(temp_schedule['Usage_Start']>=Start_Date) & (temp_schedule['Usage_End']<=End_Date)]
    else:
      temp_schedule = get_ingested_data(file_path='merged_output_repository/merged_outputs.csv',datasource_type=data_source,connection_uri=connection_string,container_name='input')
      temp_schedule['Usage_Start'] = pd.to_datetime(temp_schedule['Usage_Start'])
      temp_schedule['Usage_End'] = pd.to_datetime(temp_schedule['Usage_End'])
      schedule = temp_schedule[(temp_schedule['Usage_Start']>=Start_Date) & (temp_schedule['Usage_End']<=End_Date)]
else:
    schedule =  get_ingested_data(file_path='merged_output_repository/latest_output.csv',datasource_type=data_source,connection_uri=connection_string,container_name='input')

dataset=schedule.reset_index(drop=True)
dataset['Clean_Start_Time']=pd.to_datetime(dataset['Clean_Start_Time'],format='%Y/%m/%d %H:%M:%S')
dataset['Usage_End']=pd.to_datetime(dataset['Usage_End'],format='%Y/%m/%d %H:%M:%S')
dataset['Clean_End_Time']=pd.to_datetime(dataset['Clean_End_Time'],format='%Y/%m/%d %H:%M:%S')
dataset['Usage_Start']=pd.to_datetime(dataset['Usage_Start'],format='%Y/%m/%d %H:%M:%S')
dataset['MC Group'] = dataset['MC Group'].fillna(1)
dataset['Missed Cleanings'] = np.where(dataset['MC Group']==1,1,0)
def business_metrics_resource(dataset):
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
                                                            'Missed Cleanings':'sum'
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
    return bm
def business_metrics_CIP(dataset):
    if dataset['Clean_Start_Time'].min() > dataset['Usage_Start'].min():
        working_hours = (dataset['Clean_End_Time'].max()-dataset['Usage_Start'].min())/ (np.timedelta64(1, 'h'))
    else:
        working_hours = (dataset['Clean_End_Time'].max()-dataset['Clean_Start_Time'].min())/ (np.timedelta64(1, 'h'))
    dataset['timebw'] = dataset['Clean_Start_Time'] - dataset['Usage_End']
    dataset['timebw'] = dataset['timebw'] / np.timedelta64(1, 'h')
    dataset['setup_flag'] = np.where(dataset['timebw'] > 0.166 , 1, 0)
    dataset['setup_flag'] = dataset['setup_flag']*10
    dataset = dataset.rename(columns = {'setup_flag': 'Setup time'})
    dataset['Setup time'] = (dataset['Setup time']/60).round(1)
    metrics_cip = dataset.groupby(['MC Group','Constraint'], as_index = False).agg({
                                                   'Resource': pd.Series.nunique, 
                                                    'clean_flag': 'sum', 
                                                   'Cleaning Duration':'sum',
                                                    'Setup time':'sum',
                                                    'Parallel Clean Flag':pd.Series.nunique
                                                    })
    metrics_cip = metrics_cip.rename(columns = {'Resource': 'Resources cleaned',
                               'clean_flag':'No. of cleanings','Cleaning Duration':'CIP used Duration',
                               })
    metrics_cip['No. of cleanings'] = metrics_cip['No. of cleanings'] - metrics_cip['Parallel Clean Flag']
    metrics_cip['Available Hours']= working_hours
    metrics_cip['CIP idle time'] = (metrics_cip['Available Hours'] - metrics_cip['CIP used Duration']).round(1)
    metrics_cip['CIP idle time%'] = ((metrics_cip['CIP idle time']/(metrics_cip['CIP used Duration']+metrics_cip['Setup time']+metrics_cip['CIP idle time']))*100).round(1)
    metrics_cip['CIP Usage%'] = (((metrics_cip['CIP used Duration']+metrics_cip['Setup time'])/(metrics_cip['CIP used Duration']+metrics_cip['Setup time']+metrics_cip['CIP idle time']))*100).round(1)

    metrics_cip_total = metrics_cip.groupby(['MC Group'], as_index = False).agg({
                                                   'Resources cleaned': 'sum',
                                                    'No. of cleanings': 'sum', #No of cleanings
                                                   'CIP used Duration':'sum',#CIP used duration
                                                    'Setup time':'sum',
                                                    'Available Hours':'max',
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
        df1=df1.append({'MC Group':i,'Min Downtime between 2 CIP cleanings':inter_arrival_time},ignore_index=True)
        df1['Min Downtime between 2 CIP cleanings']=df1['Min Downtime between 2 CIP cleanings'].round(2)
        df1 = df1.dropna()
    metrics_cip_total = pd.merge(metrics_cip_total,df1,on=['MC Group'])
    return metrics_cip,metrics_cip_total
df_res = business_metrics_resource(dataset)
df_pipe,df_cip = business_metrics_CIP(dataset)
non_missed = dataset[dataset['MC Group'].isin(['MC1','MC2','MC3','MC4'])]
def pre_re_cleaning(dataframe):   
    df1 = dataframe[(dataframe['preclean']==1)]
    df1['Cleaning Duration'] = df1['Cleaning Duration']/60
    data_temp = pd.DataFrame()
    for res in dataframe.Resource.unique():
        temp_df = dataframe[dataframe['Resource']==res]
        y=temp_df.min()['Usage_Start']
        data_temp = data_temp.append({"Resource":res,'Min_Usage_Start':y},ignore_index=True)
    new = pd.merge(df1,data_temp,on=['Resource'])
    new['CHT 1st U']=np.where((new['Usage_Start']== new['Min_Usage_Start']),'PREclean','REclean')
    x = new[new['CHT 1st U']=='PREclean'].shape[0]
    y =new[new['CHT 1st U']=='REclean'].shape[0]
    pre_re = pd.DataFrame([['Total no. of Pre-cleanings ',x],['Total no. of Re-cleanings',y]],columns=['Insights','Count'])
    return pre_re,new 
m,new1 = pre_re_cleaning(non_missed)
new1_preclean = new1[new1['CHT 1st U']=='PREclean']
new1_preclean = new1_preclean.groupby('MC Group',as_index = False).agg({
                                                    'Cleaning Duration':'count'})
new1_preclean = new1_preclean.rename(columns = {'Cleaning Duration': 'Preclean'})
new1_reclean = new1[new1['CHT 1st U']=='REclean']
new1_reclean = new1_reclean.groupby('MC Group',as_index = False).agg({
                                                    'Cleaning Duration':'count'})
new1_reclean = new1_reclean.rename(columns = {'Cleaning Duration': 'Reclean'})
fig_duration = go.Figure(data=[go.Bar(name='Precleaning',x=new1_preclean['MC Group'],y =new1_preclean['Preclean'], text =new1_preclean['Preclean'], textposition= 'outside'),
                         go.Bar(name ='Recleaning',x=new1_reclean['MC Group'],y=new1_reclean['Reclean'],text =new1_reclean['Reclean'], textposition='outside')])
fig_duration.update_layout(barmode= 'group')
fig_duration.update_traces(texttemplate='%{text:0f}',textposition='outside',width=0.3,cliponaxis = False)
fig_duration.update_layout(title_text='# PRECLEANINGS AND RECLEANINGS')
fig_duration.update_layout(xaxis_title='Machine',
                  yaxis_title='COUNT')
fig_duration.show()
dynamic_outputs = plotly.io.to_json(fig_duration)

"""


# In[12]:


kpi_reclean="""
import pandas as pd
from pathlib import Path
from azure.storage.blob import BlockBlobService
from io import StringIO
import json
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly


from azure.cosmosdb.table.tableservice import TableService
from pathlib import Path
from azure.storage.blob import BlockBlobService
from io import StringIO
from datetime import datetime, timedelta
import os
import pandas as pd
import json
from dateutil import parser
# from azure.identity import DefaultAzureCredential

key_value = 's94v9Q0+SXPPRagBKJUfB2WOA2Ts5mFPUb4sUg+C0tRzagJel6eYeeyrA7Dlz7s2mmPorbtOJQohLgjefo++Vw=='
connection_string = 'DefaultEndpointsProtocol=https;AccountName=mathcotakedastorage;AccountKey=s94v9Q0+SXPPRagBKJUfB2WOA2Ts5mFPUb4sUg+C0tRzagJel6eYeeyrA7Dlz7s2mmPorbtOJQohLgjefo++Vw==;EndpointSuffix=core.windows.net'
accountname = 'mathcotakedastorage'
data_source = 'azure_blob_storage'

def get_ingested_data(file_path, datasource_type, connection_uri, container_name):
    path = Path(file_path)
    block_blob_service = BlockBlobService(connection_string=connection_uri)
    blob_data = block_blob_service.get_blob_to_text(container_name=container_name, 
                                                    blob_name=file_path)
    ingested_df = pd.read_csv(StringIO(blob_data.content))
    return ingested_df
block_blob_service = BlockBlobService(connection_string=connection_string)
blob_data = block_blob_service.get_blob_to_text(container_name='input1', blob_name='Scheduler_Input/schedule_input.json')
f = pd.read_json(StringIO(blob_data.content))

#Start Date UTC and round handling
Start_Date = f['data']['Start Date'][0:19]
Start_Date=parser.parse(datetime.strptime(Start_Date ,'%Y-%m-%dT%H:%M:%S').strftime('%Y-%m-%dT%H:%M:%S'))
Start_Date=datetime.strftime(Start_Date ,'%Y-%m-%d')
Start_Date=parser.parse(datetime.strptime(Start_Date ,'%Y-%m-%d').strftime('%Y-%m-%d'))
#End Date UTC and round handling
End_Date = f['data']['End Date'][0:19]
End_Date=parser.parse(datetime.strptime(End_Date ,'%Y-%m-%dT%H:%M:%S').strftime('%Y-%m-%dT%H:%M:%S'))
End_Date=datetime.strftime(End_Date ,'%Y-%m-%d')
End_Date=parser.parse(datetime.strptime(End_Date ,'%Y-%m-%d').strftime('%Y-%m-%d'))+timedelta(hours=23.99)
# All the metadata is stored into table
table_service= TableService(account_name='mathcotakedastorage', account_key='s94v9Q0+SXPPRagBKJUfB2WOA2Ts5mFPUb4sUg+C0tRzagJel6eYeeyrA7Dlz7s2mmPorbtOJQohLgjefo++Vw==')
tasks=table_service.query_entities('schedule', f"PartitionKey eq 'schedule' and type eq 'Planned'")
table=pd.DataFrame(tasks)
# Converting start date and end date in metadata df to datetime
table['start_at'] = pd.to_datetime(table['start_at'],utc=True).dt.date
table['end_at'] = pd.to_datetime(table['end_at'],utc=True).dt.date
table['start_at']=pd.to_datetime(table['start_at'])
table['end_at']=pd.to_datetime(table['end_at'])
# Query df such that we get file path of data with schedule date between the given date on UI
# path eg = 'Src1/300921_Schedule/2021-09-04T04-10/output/20210930_2021-09-04T04-10_Optimized_Schedule_Output_(R).csv'
if f['data']['use_latest_schedule']==True:
    queried_table = table[(table['start_at'] <= datetime.strftime(Start_Date,'%Y-%m-%d')) & (table['end_at'] >= datetime.strftime(End_Date ,'%Y-%m-%d'))]
    queried_table = queried_table.tail(1)
    queried_table.reset_index(inplace = True)
    queried_path = queried_table['result_file'][0]
    queried_path = queried_path[6:76] + '_Optimized_Schedule_Output_(R).csv'
    condition1=[(queried_table['start_at'] == datetime.strftime(Start_Date,'%Y-%m-%d')) & (queried_table['end_at'] == datetime.strftime(End_Date ,'%Y-%m-%d'))][0]
    condition2=[(queried_table['start_at'] < datetime.strftime(Start_Date,'%Y-%m-%d')) | (queried_table['end_at'] > datetime.strftime(End_Date ,'%Y-%m-%d'))][0]
    if condition1[0]==True:
      schedule = get_ingested_data(file_path=queried_path,datasource_type=data_source,connection_uri=connection_string,container_name='input')
    elif condition2[0]==True:
      temp_schedule = queried_path
      temp_schedule = get_ingested_data(file_path=queried_path,datasource_type=data_source,connection_uri=connection_string,container_name='input')
      temp_schedule['Usage_Start'] = pd.to_datetime(temp_schedule['Usage_Start'])
      temp_schedule['Usage_End'] = pd.to_datetime(temp_schedule['Usage_End'])
      temp_schedule.sort_values(by='Usage_Start')
      schedule = temp_schedule[(temp_schedule['Usage_Start']>=Start_Date) & (temp_schedule['Usage_End']<=End_Date)]
    else:
      temp_schedule = get_ingested_data(file_path='merged_output_repository/merged_outputs.csv',datasource_type=data_source,connection_uri=connection_string,container_name='input')
      temp_schedule['Usage_Start'] = pd.to_datetime(temp_schedule['Usage_Start'])
      temp_schedule['Usage_End'] = pd.to_datetime(temp_schedule['Usage_End'])
      schedule = temp_schedule[(temp_schedule['Usage_Start']>=Start_Date) & (temp_schedule['Usage_End']<=End_Date)]
else:
    schedule =  get_ingested_data(file_path='merged_output_repository/latest_output.csv',datasource_type=data_source,connection_uri=connection_string,container_name='input')

dataset=schedule.reset_index(drop=True)
dataset['Clean_Start_Time']=pd.to_datetime(dataset['Clean_Start_Time'],format='%Y/%m/%d %H:%M:%S')
dataset['Usage_End']=pd.to_datetime(dataset['Usage_End'],format='%Y/%m/%d %H:%M:%S')
dataset['Clean_End_Time']=pd.to_datetime(dataset['Clean_End_Time'],format='%Y/%m/%d %H:%M:%S')
dataset['Usage_Start']=pd.to_datetime(dataset['Usage_Start'],format='%Y/%m/%d %H:%M:%S')
dataset['MC Group'] = dataset['MC Group'].fillna(1)
dataset['Missed Cleanings'] = np.where(dataset['MC Group']==1,1,0)
def business_metrics_resource(dataset):
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
                                                            'Missed Cleanings':'sum'
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
    return bm
df_res = business_metrics_resource(dataset)
df_missed = df_res[df_res['Missed Cleanings'] > 0]
if len(df_missed)!=0:
    fig_missed = px.bar(df_missed,
                y=df_missed['Missed Cleanings'],
                x=df_missed['Resource'], labels={'x':'Machine','y':'# Re-Cleanings'}, text=df_missed['Missed Cleanings'],
                title='# MISSED CLEANINGS'
                ).update_xaxes(categoryorder="total descending")
    fig_missed.update_traces(texttemplate='%{text:0f}',textposition='outside',width=0.3,cliponaxis = False)
    fig_missed.update_yaxes(nticks=3)
    fig_missed.show()
else:
    fig_missed={}
dynamic_outputs = plotly.io.to_json(fig_missed)
"""


# In[13]:


kpi_cip_util="""
import pandas as pd
from pathlib import Path
from azure.storage.blob import BlockBlobService
from io import StringIO
import json
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly


from azure.cosmosdb.table.tableservice import TableService
from pathlib import Path
from azure.storage.blob import BlockBlobService
from io import StringIO
from datetime import datetime, timedelta
import os
import pandas as pd
import json
from dateutil import parser
# from azure.identity import DefaultAzureCredential

key_value = 's94v9Q0+SXPPRagBKJUfB2WOA2Ts5mFPUb4sUg+C0tRzagJel6eYeeyrA7Dlz7s2mmPorbtOJQohLgjefo++Vw=='
connection_string = 'DefaultEndpointsProtocol=https;AccountName=mathcotakedastorage;AccountKey=s94v9Q0+SXPPRagBKJUfB2WOA2Ts5mFPUb4sUg+C0tRzagJel6eYeeyrA7Dlz7s2mmPorbtOJQohLgjefo++Vw==;EndpointSuffix=core.windows.net'
accountname = 'mathcotakedastorage'
data_source = 'azure_blob_storage'

def get_ingested_data(file_path, datasource_type, connection_uri, container_name):
    path = Path(file_path)
    block_blob_service = BlockBlobService(connection_string=connection_uri)
    blob_data = block_blob_service.get_blob_to_text(container_name=container_name, 
                                                    blob_name=file_path)
    ingested_df = pd.read_csv(StringIO(blob_data.content))
    return ingested_df
block_blob_service = BlockBlobService(connection_string=connection_string)
blob_data = block_blob_service.get_blob_to_text(container_name='input1', blob_name='Scheduler_Input/schedule_input.json')
f = pd.read_json(StringIO(blob_data.content))

#Start Date UTC and round handling
Start_Date = f['data']['Start Date'][0:19]
Start_Date=parser.parse(datetime.strptime(Start_Date ,'%Y-%m-%dT%H:%M:%S').strftime('%Y-%m-%dT%H:%M:%S'))
Start_Date=datetime.strftime(Start_Date ,'%Y-%m-%d')
Start_Date=parser.parse(datetime.strptime(Start_Date ,'%Y-%m-%d').strftime('%Y-%m-%d'))
#End Date UTC and round handling
End_Date = f['data']['End Date'][0:19]
End_Date=parser.parse(datetime.strptime(End_Date ,'%Y-%m-%dT%H:%M:%S').strftime('%Y-%m-%dT%H:%M:%S'))
End_Date=datetime.strftime(End_Date ,'%Y-%m-%d')
End_Date=parser.parse(datetime.strptime(End_Date ,'%Y-%m-%d').strftime('%Y-%m-%d'))+timedelta(hours=23.99)
# All the metadata is stored into table
table_service= TableService(account_name='mathcotakedastorage', account_key='s94v9Q0+SXPPRagBKJUfB2WOA2Ts5mFPUb4sUg+C0tRzagJel6eYeeyrA7Dlz7s2mmPorbtOJQohLgjefo++Vw==')
tasks=table_service.query_entities('schedule', f"PartitionKey eq 'schedule' and type eq 'Planned'")
table=pd.DataFrame(tasks)
# Converting start date and end date in metadata df to datetime
table['start_at'] = pd.to_datetime(table['start_at'],utc=True).dt.date
table['end_at'] = pd.to_datetime(table['end_at'],utc=True).dt.date
table['start_at']=pd.to_datetime(table['start_at'])
table['end_at']=pd.to_datetime(table['end_at'])
# Query df such that we get file path of data with schedule date between the given date on UI
# path eg = 'Src1/300921_Schedule/2021-09-04T04-10/output/20210930_2021-09-04T04-10_Optimized_Schedule_Output_(R).csv'
if f['data']['use_latest_schedule']==True:
    queried_table = table[(table['start_at'] <= datetime.strftime(Start_Date,'%Y-%m-%d')) & (table['end_at'] >= datetime.strftime(End_Date ,'%Y-%m-%d'))]
    queried_table = queried_table.tail(1)
    queried_table.reset_index(inplace = True)
    queried_path = queried_table['result_file'][0]
    queried_path = queried_path[6:76] + '_Optimized_Schedule_Output_(R).csv'
    condition1=[(queried_table['start_at'] == datetime.strftime(Start_Date,'%Y-%m-%d')) & (queried_table['end_at'] == datetime.strftime(End_Date ,'%Y-%m-%d'))][0]
    condition2=[(queried_table['start_at'] < datetime.strftime(Start_Date,'%Y-%m-%d')) | (queried_table['end_at'] > datetime.strftime(End_Date ,'%Y-%m-%d'))][0]
    if condition1[0]==True:
      schedule = get_ingested_data(file_path=queried_path,datasource_type=data_source,connection_uri=connection_string,container_name='input')
    elif condition2[0]==True:
      temp_schedule = queried_path
      temp_schedule = get_ingested_data(file_path=queried_path,datasource_type=data_source,connection_uri=connection_string,container_name='input')
      temp_schedule['Usage_Start'] = pd.to_datetime(temp_schedule['Usage_Start'])
      temp_schedule['Usage_End'] = pd.to_datetime(temp_schedule['Usage_End'])
      temp_schedule.sort_values(by='Usage_Start')
      schedule = temp_schedule[(temp_schedule['Usage_Start']>=Start_Date) & (temp_schedule['Usage_End']<=End_Date)]
    else:
      temp_schedule = get_ingested_data(file_path='merged_output_repository/merged_outputs.csv',datasource_type=data_source,connection_uri=connection_string,container_name='input')
      temp_schedule['Usage_Start'] = pd.to_datetime(temp_schedule['Usage_Start'])
      temp_schedule['Usage_End'] = pd.to_datetime(temp_schedule['Usage_End'])
      schedule = temp_schedule[(temp_schedule['Usage_Start']>=Start_Date) & (temp_schedule['Usage_End']<=End_Date)]
else:
    schedule =  get_ingested_data(file_path='merged_output_repository/latest_output.csv',datasource_type=data_source,connection_uri=connection_string,container_name='input')

dataset=schedule.reset_index(drop=True)
dataset['Clean_Start_Time']=pd.to_datetime(dataset['Clean_Start_Time'],format='%Y/%m/%d %H:%M:%S')
dataset['Clean_End_Time']=pd.to_datetime(dataset['Clean_End_Time'],format='%Y/%m/%d %H:%M:%S')
dataset['Clean_Start_Time']= pd.to_datetime(dataset['Clean_Start_Time'],infer_datetime_format=True,utc=True)
dataset['Clean_End_Time']= pd.to_datetime(dataset['Clean_End_Time'],infer_datetime_format=True,utc=True)

working_hours = (dataset['Clean_End_Time'].max()-dataset['Clean_Start_Time'].min())/ (np.timedelta64(1, 'h'))
dataset['Cleaning Duration']=dataset['Cleaning Duration']/60
metrics_cip = dataset.groupby(['MC Group','Constraint','Resource'], as_index = False).agg({
                                                'clean_flag': 'sum',
                                            'Cleaning Duration':'sum'
                                                })
metrics_cip = metrics_cip.rename(columns = {'clean_flag':'No. of cleanings','Cleaning Duration':'CIP used Duration',
                        })
metrics_cip['Available Hours']= working_hours
metrics_cip['CIP idle time'] = (metrics_cip['Available Hours'] - metrics_cip['CIP used Duration']).round(1)
metrics_cip['CIP idle time%'] = ((metrics_cip['CIP idle time']/(metrics_cip['CIP used Duration']+metrics_cip['CIP idle time']))*100).round(1)
metrics_cip['CIP Usage%'] = (((metrics_cip['CIP used Duration'])/(metrics_cip['CIP used Duration']+metrics_cip['CIP idle time']))*100).round(1)
total_cip = metrics_cip.groupby(['MC Group'], as_index = False).agg({
                                                'CIP used Duration': 'sum',
                                            'Available Hours':'min'
                                                })
total_cip['CIP Usage%'] = (((total_cip['CIP used Duration'])/(total_cip['Available Hours']))*100).round()
fig_cip =px.sunburst(metrics_cip,path=['MC Group','Resource'],values='No. of cleanings')
fig_cip.update_traces(textfont=dict(color="white"))
fig_cip.update_layout(hoverlabel=dict(font_color="white",font_size=16))

labels = ['MC1','MC2','MC3','MC4']
fig = go.Figure(data = go.Pie(labels=labels, values=total_cip['CIP Usage%'].tolist(),sort=False))
fig.update_traces(hole=.4, hoverinfo="label+percent+name",textfont=dict(color="white"),texttemplate="%{percent:0%f}")
fig.show()
fig_cip.show()
fig_month=json.loads(plotly.io.to_json(fig_cip))
fig_week=json.loads(plotly.io.to_json(fig))

util_flip= {
  "front":{
    "data":{
      "value":fig_month
      }
  },
  "back":{
    "data":{
      "value":fig_week
    }
  },
  "is_flip":True
}

dynamic_outputs=json.dumps(util_flip)

"""


# In[14]:


kpi_pipe_util="""
import pandas as pd
from pathlib import Path
from azure.storage.blob import BlockBlobService
from io import StringIO
import json
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly


from azure.cosmosdb.table.tableservice import TableService
from pathlib import Path
from azure.storage.blob import BlockBlobService
from io import StringIO
from datetime import datetime, timedelta
import os
import pandas as pd
import json
from dateutil import parser
# from azure.identity import DefaultAzureCredential

key_value = 's94v9Q0+SXPPRagBKJUfB2WOA2Ts5mFPUb4sUg+C0tRzagJel6eYeeyrA7Dlz7s2mmPorbtOJQohLgjefo++Vw=='
connection_string = 'DefaultEndpointsProtocol=https;AccountName=mathcotakedastorage;AccountKey=s94v9Q0+SXPPRagBKJUfB2WOA2Ts5mFPUb4sUg+C0tRzagJel6eYeeyrA7Dlz7s2mmPorbtOJQohLgjefo++Vw==;EndpointSuffix=core.windows.net'
accountname = 'mathcotakedastorage'
data_source = 'azure_blob_storage'

def get_ingested_data(file_path, datasource_type, connection_uri, container_name):
    path = Path(file_path)
    block_blob_service = BlockBlobService(connection_string=connection_uri)
    blob_data = block_blob_service.get_blob_to_text(container_name=container_name, 
                                                    blob_name=file_path)
    ingested_df = pd.read_csv(StringIO(blob_data.content))
    return ingested_df
block_blob_service = BlockBlobService(connection_string=connection_string)
blob_data = block_blob_service.get_blob_to_text(container_name='input1', blob_name='Scheduler_Input/schedule_input.json')
f = pd.read_json(StringIO(blob_data.content))

#Start Date UTC and round handling
Start_Date = f['data']['Start Date'][0:19]
Start_Date=parser.parse(datetime.strptime(Start_Date ,'%Y-%m-%dT%H:%M:%S').strftime('%Y-%m-%dT%H:%M:%S'))
Start_Date=datetime.strftime(Start_Date ,'%Y-%m-%d')
Start_Date=parser.parse(datetime.strptime(Start_Date ,'%Y-%m-%d').strftime('%Y-%m-%d'))
#End Date UTC and round handling
End_Date = f['data']['End Date'][0:19]
End_Date=parser.parse(datetime.strptime(End_Date ,'%Y-%m-%dT%H:%M:%S').strftime('%Y-%m-%dT%H:%M:%S'))
End_Date=datetime.strftime(End_Date ,'%Y-%m-%d')
End_Date=parser.parse(datetime.strptime(End_Date ,'%Y-%m-%d').strftime('%Y-%m-%d'))+timedelta(hours=23.99)
# All the metadata is stored into table
table_service= TableService(account_name='mathcotakedastorage', account_key='s94v9Q0+SXPPRagBKJUfB2WOA2Ts5mFPUb4sUg+C0tRzagJel6eYeeyrA7Dlz7s2mmPorbtOJQohLgjefo++Vw==')
tasks=table_service.query_entities('schedule', f"PartitionKey eq 'schedule' and type eq 'Planned'")
table=pd.DataFrame(tasks)
# Converting start date and end date in metadata df to datetime
table['start_at'] = pd.to_datetime(table['start_at'],utc=True).dt.date
table['end_at'] = pd.to_datetime(table['end_at'],utc=True).dt.date
table['start_at']=pd.to_datetime(table['start_at'])
table['end_at']=pd.to_datetime(table['end_at'])
# Query df such that we get file path of data with schedule date between the given date on UI
# path eg = 'Src1/300921_Schedule/2021-09-04T04-10/output/20210930_2021-09-04T04-10_Optimized_Schedule_Output_(R).csv'
if f['data']['use_latest_schedule']==True:
    queried_table = table[(table['start_at'] <= datetime.strftime(Start_Date,'%Y-%m-%d')) & (table['end_at'] >= datetime.strftime(End_Date ,'%Y-%m-%d'))]
    queried_table = queried_table.tail(1)
    queried_table.reset_index(inplace = True)
    queried_path = queried_table['result_file'][0]
    queried_path = queried_path[6:76] + '_Optimized_Schedule_Output_(R).csv'
    condition1=[(queried_table['start_at'] == datetime.strftime(Start_Date,'%Y-%m-%d')) & (queried_table['end_at'] == datetime.strftime(End_Date ,'%Y-%m-%d'))][0]
    condition2=[(queried_table['start_at'] < datetime.strftime(Start_Date,'%Y-%m-%d')) | (queried_table['end_at'] > datetime.strftime(End_Date ,'%Y-%m-%d'))][0]
    if condition1[0]==True:
      schedule = get_ingested_data(file_path=queried_path,datasource_type=data_source,connection_uri=connection_string,container_name='input')
    elif condition2[0]==True:
      temp_schedule = queried_path
      temp_schedule = get_ingested_data(file_path=queried_path,datasource_type=data_source,connection_uri=connection_string,container_name='input')
      temp_schedule['Usage_Start'] = pd.to_datetime(temp_schedule['Usage_Start'])
      temp_schedule['Usage_End'] = pd.to_datetime(temp_schedule['Usage_End'])
      temp_schedule.sort_values(by='Usage_Start')
      schedule = temp_schedule[(temp_schedule['Usage_Start']>=Start_Date) & (temp_schedule['Usage_End']<=End_Date)]
    else:
      temp_schedule = get_ingested_data(file_path='merged_output_repository/merged_outputs.csv',datasource_type=data_source,connection_uri=connection_string,container_name='input')
      temp_schedule['Usage_Start'] = pd.to_datetime(temp_schedule['Usage_Start'])
      temp_schedule['Usage_End'] = pd.to_datetime(temp_schedule['Usage_End'])
      schedule = temp_schedule[(temp_schedule['Usage_Start']>=Start_Date) & (temp_schedule['Usage_End']<=End_Date)]
else:
    schedule =  get_ingested_data(file_path='merged_output_repository/latest_output.csv',datasource_type=data_source,connection_uri=connection_string,container_name='input')

dataset=schedule.reset_index(drop=True)
dataset['Clean_Start_Time']=pd.to_datetime(dataset['Clean_Start_Time'],format='%Y/%m/%d %H:%M:%S')
dataset['Clean_End_Time']=pd.to_datetime(dataset['Clean_End_Time'],format='%Y/%m/%d %H:%M:%S')
dataset['Clean_Start_Time']= pd.to_datetime(dataset['Clean_Start_Time'],infer_datetime_format=True,utc=True)
dataset['Clean_End_Time']= pd.to_datetime(dataset['Clean_End_Time'],infer_datetime_format=True,utc=True)

working_hours = (dataset['Clean_End_Time'].max()-dataset['Clean_Start_Time'].min())/ (np.timedelta64(1, 'h'))
dataset['Cleaning Duration']=dataset['Cleaning Duration']/60
metrics_cip = dataset.groupby(['MC Group','Constraint','Resource'], as_index = False).agg({
                                                'clean_flag': 'sum',
                                            'Cleaning Duration':'sum'
                                                })
metrics_cip = metrics_cip.rename(columns = {'clean_flag':'No. of cleanings','Cleaning Duration':'CIP used Duration',
                        })
metrics_cip['Available Hours']= working_hours
metrics_cip['CIP idle time'] = (metrics_cip['Available Hours'] - metrics_cip['CIP used Duration']).round(1)
metrics_cip['CIP idle time%'] = ((metrics_cip['CIP idle time']/(metrics_cip['CIP used Duration']+metrics_cip['CIP idle time']))*100).round(1)
metrics_cip['CIP Usage%'] = (((metrics_cip['CIP used Duration'])/(metrics_cip['CIP used Duration']+metrics_cip['CIP idle time']))*100).round(1)
total_pipe = metrics_cip.groupby(['Constraint'], as_index = False).agg({
                                                'CIP used Duration': 'sum',
                                            'Available Hours':'min'
                                                })
total_pipe['CIP Usage%'] = (((total_pipe['CIP used Duration'])/(total_pipe['Available Hours']))*100).round()
fig_cip =px.sunburst(metrics_cip,path=['Constraint','Resource'],values='No. of cleanings')
fig_cip.update_traces(textfont=dict(color="white"))
fig_cip.update_layout(hoverlabel=dict(font_color="white",font_size=16))
fig_cip.show()
# fig_pipe_cycles = px.pie(total_pipe, values='CIP Usage%', names='Constraint', title='CONSTRAINT USAGE DISTRIBUTION')
# fig_pipe_cycles.update_traces(texttemplate="%{percent:0%f}",textfont=dict(color="white"))
# fig_pipe_cycles.update_traces(hole=.4, hoverinfo="label+percent+name",textfont=dict(color="white"),texttemplate="%{percent:0%f}")
# fig_pipe_cycles.show()
labels = total_pipe['Constraint'].tolist()
fig = go.Figure(data = go.Bar(x=labels, y=total_pipe['CIP Usage%'].tolist(),text=total_pipe['CIP Usage%'],textposition='outside'))
fig.update_traces(texttemplate='%{text:0f}',cliponaxis = False)
fig.update_layout(xaxis_title='Constraint',
                  yaxis_title='Usage %')
fig.show()
fig_month=json.loads(plotly.io.to_json(fig_cip))
fig_week=json.loads(plotly.io.to_json(fig))

util_flip= {
  "front":{
    "data":{
      "value":fig_month
      }
  },
  "back":{
    "data":{
      "value":fig_week
    }
  },
  "is_flip":True
}
dynamic_outputs=json.dumps(util_flip)
"""


# # Resource Summary Code String #

# In[15]:


kpi_res_util="""
import pandas as pd
from pathlib import Path
from azure.storage.blob import BlockBlobService
from io import StringIO
import json
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly



from azure.cosmosdb.table.tableservice import TableService
from pathlib import Path
from azure.storage.blob import BlockBlobService
from io import StringIO
from datetime import datetime, timedelta
import os
import pandas as pd
import json
from dateutil import parser
# from azure.identity import DefaultAzureCredential

key_value = 's94v9Q0+SXPPRagBKJUfB2WOA2Ts5mFPUb4sUg+C0tRzagJel6eYeeyrA7Dlz7s2mmPorbtOJQohLgjefo++Vw=='
connection_string = 'DefaultEndpointsProtocol=https;AccountName=mathcotakedastorage;AccountKey=s94v9Q0+SXPPRagBKJUfB2WOA2Ts5mFPUb4sUg+C0tRzagJel6eYeeyrA7Dlz7s2mmPorbtOJQohLgjefo++Vw==;EndpointSuffix=core.windows.net'
accountname = 'mathcotakedastorage'
data_source = 'azure_blob_storage'

def get_ingested_data(file_path, datasource_type, connection_uri, container_name):
    path = Path(file_path)
    block_blob_service = BlockBlobService(connection_string=connection_uri)
    blob_data = block_blob_service.get_blob_to_text(container_name=container_name, 
                                                    blob_name=file_path)
    ingested_df = pd.read_csv(StringIO(blob_data.content))
    return ingested_df
block_blob_service = BlockBlobService(connection_string=connection_string)
blob_data = block_blob_service.get_blob_to_text(container_name='input1', blob_name='Scheduler_Input/schedule_input.json')
f = pd.read_json(StringIO(blob_data.content))

#Start Date UTC and round handling
Start_Date = f['data']['Start Date'][0:19]
Start_Date=parser.parse(datetime.strptime(Start_Date ,'%Y-%m-%dT%H:%M:%S').strftime('%Y-%m-%dT%H:%M:%S'))
Start_Date=datetime.strftime(Start_Date ,'%Y-%m-%d')
Start_Date=parser.parse(datetime.strptime(Start_Date ,'%Y-%m-%d').strftime('%Y-%m-%d'))
#End Date UTC and round handling
End_Date = f['data']['End Date'][0:19]
End_Date=parser.parse(datetime.strptime(End_Date ,'%Y-%m-%dT%H:%M:%S').strftime('%Y-%m-%dT%H:%M:%S'))
End_Date=datetime.strftime(End_Date ,'%Y-%m-%d')
End_Date=parser.parse(datetime.strptime(End_Date ,'%Y-%m-%d').strftime('%Y-%m-%d'))+timedelta(hours=23.99)
# All the metadata is stored into table
table_service= TableService(account_name='mathcotakedastorage', account_key='s94v9Q0+SXPPRagBKJUfB2WOA2Ts5mFPUb4sUg+C0tRzagJel6eYeeyrA7Dlz7s2mmPorbtOJQohLgjefo++Vw==')
tasks=table_service.query_entities('schedule', f"PartitionKey eq 'schedule' and type eq 'Planned'")
table=pd.DataFrame(tasks)
# Converting start date and end date in metadata df to datetime
table['start_at'] = pd.to_datetime(table['start_at'],utc=True).dt.date
table['end_at'] = pd.to_datetime(table['end_at'],utc=True).dt.date
table['start_at']=pd.to_datetime(table['start_at'])
table['end_at']=pd.to_datetime(table['end_at'])
# Query df such that we get file path of data with schedule date between the given date on UI
# path eg = 'Src1/300921_Schedule/2021-09-04T04-10/output/20210930_2021-09-04T04-10_Optimized_Schedule_Output_(R).csv'
if f['data']['use_latest_schedule']==True:
    queried_table = table[(table['start_at'] <= datetime.strftime(Start_Date,'%Y-%m-%d')) & (table['end_at'] >= datetime.strftime(End_Date ,'%Y-%m-%d'))]
    queried_table = queried_table.tail(1)
    queried_table.reset_index(inplace = True)
    queried_path = queried_table['result_file'][0]
    queried_path = queried_path[6:76] + '_Optimized_Schedule_Output_(R).csv'
    condition1=[(queried_table['start_at'] == datetime.strftime(Start_Date,'%Y-%m-%d')) & (queried_table['end_at'] == datetime.strftime(End_Date ,'%Y-%m-%d'))][0]
    condition2=[(queried_table['start_at'] < datetime.strftime(Start_Date,'%Y-%m-%d')) | (queried_table['end_at'] > datetime.strftime(End_Date ,'%Y-%m-%d'))][0]
    if condition1[0]==True:
      schedule = get_ingested_data(file_path=queried_path,datasource_type=data_source,connection_uri=connection_string,container_name='input')
    elif condition2[0]==True:
      temp_schedule = queried_path
      temp_schedule = get_ingested_data(file_path=queried_path,datasource_type=data_source,connection_uri=connection_string,container_name='input')
      temp_schedule['Usage_Start'] = pd.to_datetime(temp_schedule['Usage_Start'])
      temp_schedule['Usage_End'] = pd.to_datetime(temp_schedule['Usage_End'])
      temp_schedule.sort_values(by='Usage_Start')
      schedule = temp_schedule[(temp_schedule['Usage_Start']>=Start_Date) & (temp_schedule['Usage_End']<=End_Date)]
    else:
      temp_schedule = get_ingested_data(file_path='merged_output_repository/merged_outputs.csv',datasource_type=data_source,connection_uri=connection_string,container_name='input')
      temp_schedule['Usage_Start'] = pd.to_datetime(temp_schedule['Usage_Start'])
      temp_schedule['Usage_End'] = pd.to_datetime(temp_schedule['Usage_End'])
      schedule = temp_schedule[(temp_schedule['Usage_Start']>=Start_Date) & (temp_schedule['Usage_End']<=End_Date)]
else:
    schedule =  get_ingested_data(file_path='merged_output_repository/latest_output.csv',datasource_type=data_source,connection_uri=connection_string,container_name='input')

dataset=schedule.reset_index(drop=True)
dataset['Clean_Start_Time']=pd.to_datetime(dataset['Clean_Start_Time'],format='%Y/%m/%d %H:%M:%S')
dataset['Usage_End']=pd.to_datetime(dataset['Usage_End'],format='%Y/%m/%d %H:%M:%S')
dataset['Clean_End_Time']=pd.to_datetime(dataset['Clean_End_Time'],format='%Y/%m/%d %H:%M:%S')
dataset['Usage_Start']=pd.to_datetime(dataset['Usage_Start'],format='%Y/%m/%d %H:%M:%S')
dataset['Usage_Start']= pd.to_datetime(dataset['Usage_Start'],infer_datetime_format=True,utc=True)
dataset['Usage_End']= pd.to_datetime(dataset['Usage_End'],infer_datetime_format=True,utc=True)
dataset['Clean_Start_Time']= pd.to_datetime(dataset['Clean_Start_Time'],infer_datetime_format=True,utc=True)
dataset['Clean_End_Time']= pd.to_datetime(dataset['Clean_End_Time'],infer_datetime_format=True,utc=True)
dataset['MC Group'] = dataset['MC Group'].fillna(1)
dataset['Missed Cleanings'] = np.where(dataset['MC Group']==1,1,0)
def business_metrics_resource(dataset):
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
                                                            'Missed Cleanings':'sum'
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
    return bm
df_res=business_metrics_resource(dataset)
df_res['Resource']=df_res['Resource'].replace({'Long Transfer Line - L717L':'L717L'})
fig_util = px.bar(df_res,
             y=df_res['No. of Utilizations'],
             x=df_res['Resource'], text=df_res['No. of Utilizations'],title='Frequency of Utilizations'
             ).update_xaxes(categoryorder="total descending")
fig_util.update_traces(texttemplate='%{text:0f}',textposition='outside',width=0.3,cliponaxis = False)
fig_util.show()
dynamic_outputs=plotly.io.to_json(fig_util)
"""


# In[16]:


kpi_res_usage="""
import pandas as pd
from pathlib import Path
from azure.storage.blob import BlockBlobService
from io import StringIO
import json
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly



from azure.cosmosdb.table.tableservice import TableService
from pathlib import Path
from azure.storage.blob import BlockBlobService
from io import StringIO
from datetime import datetime, timedelta
import os
import pandas as pd
import json
from dateutil import parser
# from azure.identity import DefaultAzureCredential

key_value = 's94v9Q0+SXPPRagBKJUfB2WOA2Ts5mFPUb4sUg+C0tRzagJel6eYeeyrA7Dlz7s2mmPorbtOJQohLgjefo++Vw=='
connection_string = 'DefaultEndpointsProtocol=https;AccountName=mathcotakedastorage;AccountKey=s94v9Q0+SXPPRagBKJUfB2WOA2Ts5mFPUb4sUg+C0tRzagJel6eYeeyrA7Dlz7s2mmPorbtOJQohLgjefo++Vw==;EndpointSuffix=core.windows.net'
accountname = 'mathcotakedastorage'
data_source = 'azure_blob_storage'

def get_ingested_data(file_path, datasource_type, connection_uri, container_name):
    path = Path(file_path)
    block_blob_service = BlockBlobService(connection_string=connection_uri)
    blob_data = block_blob_service.get_blob_to_text(container_name=container_name, 
                                                    blob_name=file_path)
    ingested_df = pd.read_csv(StringIO(blob_data.content))
    return ingested_df
block_blob_service = BlockBlobService(connection_string=connection_string)
blob_data = block_blob_service.get_blob_to_text(container_name='input1', blob_name='Scheduler_Input/schedule_input.json')
f = pd.read_json(StringIO(blob_data.content))

#Start Date UTC and round handling
Start_Date = f['data']['Start Date'][0:19]
Start_Date=parser.parse(datetime.strptime(Start_Date ,'%Y-%m-%dT%H:%M:%S').strftime('%Y-%m-%dT%H:%M:%S'))
Start_Date=datetime.strftime(Start_Date ,'%Y-%m-%d')
Start_Date=parser.parse(datetime.strptime(Start_Date ,'%Y-%m-%d').strftime('%Y-%m-%d'))
#End Date UTC and round handling
End_Date = f['data']['End Date'][0:19]
End_Date=parser.parse(datetime.strptime(End_Date ,'%Y-%m-%dT%H:%M:%S').strftime('%Y-%m-%dT%H:%M:%S'))
End_Date=datetime.strftime(End_Date ,'%Y-%m-%d')
End_Date=parser.parse(datetime.strptime(End_Date ,'%Y-%m-%d').strftime('%Y-%m-%d'))+timedelta(hours=23.99)
# All the metadata is stored into table
table_service= TableService(account_name='mathcotakedastorage', account_key='s94v9Q0+SXPPRagBKJUfB2WOA2Ts5mFPUb4sUg+C0tRzagJel6eYeeyrA7Dlz7s2mmPorbtOJQohLgjefo++Vw==')
tasks=table_service.query_entities('schedule', f"PartitionKey eq 'schedule' and type eq 'Planned'")
table=pd.DataFrame(tasks)
# Converting start date and end date in metadata df to datetime
table['start_at'] = pd.to_datetime(table['start_at'],utc=True).dt.date
table['end_at'] = pd.to_datetime(table['end_at'],utc=True).dt.date
table['start_at']=pd.to_datetime(table['start_at'])
table['end_at']=pd.to_datetime(table['end_at'])
# Query df such that we get file path of data with schedule date between the given date on UI
# path eg = 'Src1/300921_Schedule/2021-09-04T04-10/output/20210930_2021-09-04T04-10_Optimized_Schedule_Output_(R).csv'
if f['data']['use_latest_schedule']==True:
    queried_table = table[(table['start_at'] <= datetime.strftime(Start_Date,'%Y-%m-%d')) & (table['end_at'] >= datetime.strftime(End_Date ,'%Y-%m-%d'))]
    queried_table = queried_table.tail(1)
    queried_table.reset_index(inplace = True)
    queried_path = queried_table['result_file'][0]
    queried_path = queried_path[6:76] + '_Optimized_Schedule_Output_(R).csv'
    condition1=[(queried_table['start_at'] == datetime.strftime(Start_Date,'%Y-%m-%d')) & (queried_table['end_at'] == datetime.strftime(End_Date ,'%Y-%m-%d'))][0]
    condition2=[(queried_table['start_at'] < datetime.strftime(Start_Date,'%Y-%m-%d')) | (queried_table['end_at'] > datetime.strftime(End_Date ,'%Y-%m-%d'))][0]
    if condition1[0]==True:
      schedule = get_ingested_data(file_path=queried_path,datasource_type=data_source,connection_uri=connection_string,container_name='input')
    elif condition2[0]==True:
      temp_schedule = queried_path
      temp_schedule = get_ingested_data(file_path=queried_path,datasource_type=data_source,connection_uri=connection_string,container_name='input')
      temp_schedule['Usage_Start'] = pd.to_datetime(temp_schedule['Usage_Start'])
      temp_schedule['Usage_End'] = pd.to_datetime(temp_schedule['Usage_End'])
      temp_schedule.sort_values(by='Usage_Start')
      schedule = temp_schedule[(temp_schedule['Usage_Start']>=Start_Date) & (temp_schedule['Usage_End']<=End_Date)]
    else:
      temp_schedule = get_ingested_data(file_path='merged_output_repository/merged_outputs.csv',datasource_type=data_source,connection_uri=connection_string,container_name='input')
      temp_schedule['Usage_Start'] = pd.to_datetime(temp_schedule['Usage_Start'])
      temp_schedule['Usage_End'] = pd.to_datetime(temp_schedule['Usage_End'])
      schedule = temp_schedule[(temp_schedule['Usage_Start']>=Start_Date) & (temp_schedule['Usage_End']<=End_Date)]
else:
    schedule =  get_ingested_data(file_path='merged_output_repository/latest_output.csv',datasource_type=data_source,connection_uri=connection_string,container_name='input')

dataset=schedule.reset_index(drop=True)
dataset['Clean_Start_Time']=pd.to_datetime(dataset['Clean_Start_Time'],format='%Y/%m/%d %H:%M:%S')
dataset['Usage_End']=pd.to_datetime(dataset['Usage_End'],format='%Y/%m/%d %H:%M:%S')
dataset['Clean_End_Time']=pd.to_datetime(dataset['Clean_End_Time'],format='%Y/%m/%d %H:%M:%S')
dataset['Usage_Start']=pd.to_datetime(dataset['Usage_Start'],format='%Y/%m/%d %H:%M:%S')
dataset['MC Group'] = dataset['MC Group'].fillna(1)
dataset['Missed Cleanings'] = np.where(dataset['MC Group']==1,1,0)
def business_metrics_resource(dataset):
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
                                                            'Missed Cleanings':'sum'
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
    return bm
df_res=business_metrics_resource(dataset)
df_res['Resource']=df_res['Resource'].replace({'Long Transfer Line - L717L':'L717L'})
fig_res_usage = px.bar(df_res,
             y=df_res['Resource Usage %'],
             x=df_res['Resource'], text=df_res['Resource Usage %'],title='RESOURCE USAGE %',labels={'y':'USAGE %','x':'Resource'}
             ).update_xaxes(categoryorder="total descending")
fig_res_usage.update_traces(texttemplate='%{text:0f}',textposition='outside',width=0.3,cliponaxis = False)
fig_res_usage.show()
dynamic_outputs=plotly.io.to_json(fig_res_usage)
"""


# In[17]:


kpi_res_preclean="""
import pandas as pd
from pathlib import Path
from azure.storage.blob import BlockBlobService
from io import StringIO
import json
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly


from azure.cosmosdb.table.tableservice import TableService
from pathlib import Path
from azure.storage.blob import BlockBlobService
from io import StringIO
from datetime import datetime, timedelta
import os
import pandas as pd
import json
from dateutil import parser
# from azure.identity import DefaultAzureCredential

key_value = 's94v9Q0+SXPPRagBKJUfB2WOA2Ts5mFPUb4sUg+C0tRzagJel6eYeeyrA7Dlz7s2mmPorbtOJQohLgjefo++Vw=='
connection_string = 'DefaultEndpointsProtocol=https;AccountName=mathcotakedastorage;AccountKey=s94v9Q0+SXPPRagBKJUfB2WOA2Ts5mFPUb4sUg+C0tRzagJel6eYeeyrA7Dlz7s2mmPorbtOJQohLgjefo++Vw==;EndpointSuffix=core.windows.net'
accountname = 'mathcotakedastorage'
data_source = 'azure_blob_storage'

def get_ingested_data(file_path, datasource_type, connection_uri, container_name):
    path = Path(file_path)
    block_blob_service = BlockBlobService(connection_string=connection_uri)
    blob_data = block_blob_service.get_blob_to_text(container_name=container_name, 
                                                    blob_name=file_path)
    ingested_df = pd.read_csv(StringIO(blob_data.content))
    return ingested_df
block_blob_service = BlockBlobService(connection_string=connection_string)
blob_data = block_blob_service.get_blob_to_text(container_name='input1', blob_name='Scheduler_Input/schedule_input.json')
f = pd.read_json(StringIO(blob_data.content))

#Start Date UTC and round handling
Start_Date = f['data']['Start Date'][0:19]
Start_Date=parser.parse(datetime.strptime(Start_Date ,'%Y-%m-%dT%H:%M:%S').strftime('%Y-%m-%dT%H:%M:%S'))
Start_Date=datetime.strftime(Start_Date ,'%Y-%m-%d')
Start_Date=parser.parse(datetime.strptime(Start_Date ,'%Y-%m-%d').strftime('%Y-%m-%d'))
#End Date UTC and round handling
End_Date = f['data']['End Date'][0:19]
End_Date=parser.parse(datetime.strptime(End_Date ,'%Y-%m-%dT%H:%M:%S').strftime('%Y-%m-%dT%H:%M:%S'))
End_Date=datetime.strftime(End_Date ,'%Y-%m-%d')
End_Date=parser.parse(datetime.strptime(End_Date ,'%Y-%m-%d').strftime('%Y-%m-%d'))+timedelta(hours=23.99)
# All the metadata is stored into table
table_service= TableService(account_name='mathcotakedastorage', account_key='s94v9Q0+SXPPRagBKJUfB2WOA2Ts5mFPUb4sUg+C0tRzagJel6eYeeyrA7Dlz7s2mmPorbtOJQohLgjefo++Vw==')
tasks=table_service.query_entities('schedule', f"PartitionKey eq 'schedule' and type eq 'Planned'")
table=pd.DataFrame(tasks)
# Converting start date and end date in metadata df to datetime
table['start_at'] = pd.to_datetime(table['start_at'],utc=True).dt.date
table['end_at'] = pd.to_datetime(table['end_at'],utc=True).dt.date
table['start_at']=pd.to_datetime(table['start_at'])
table['end_at']=pd.to_datetime(table['end_at'])
# Query df such that we get file path of data with schedule date between the given date on UI
# path eg = 'Src1/300921_Schedule/2021-09-04T04-10/output/20210930_2021-09-04T04-10_Optimized_Schedule_Output_(R).csv'
if f['data']['use_latest_schedule']==True:
    queried_table = table[(table['start_at'] <= datetime.strftime(Start_Date,'%Y-%m-%d')) & (table['end_at'] >= datetime.strftime(End_Date ,'%Y-%m-%d'))]
    queried_table = queried_table.tail(1)
    queried_table.reset_index(inplace = True)
    queried_path = queried_table['result_file'][0]
    queried_path = queried_path[6:76] + '_Optimized_Schedule_Output_(R).csv'
    condition1=[(queried_table['start_at'] == datetime.strftime(Start_Date,'%Y-%m-%d')) & (queried_table['end_at'] == datetime.strftime(End_Date ,'%Y-%m-%d'))][0]
    condition2=[(queried_table['start_at'] < datetime.strftime(Start_Date,'%Y-%m-%d')) | (queried_table['end_at'] > datetime.strftime(End_Date ,'%Y-%m-%d'))][0]
    if condition1[0]==True:
      schedule = get_ingested_data(file_path=queried_path,datasource_type=data_source,connection_uri=connection_string,container_name='input')
    elif condition2[0]==True:
      temp_schedule = queried_path
      temp_schedule = get_ingested_data(file_path=queried_path,datasource_type=data_source,connection_uri=connection_string,container_name='input')
      temp_schedule['Usage_Start'] = pd.to_datetime(temp_schedule['Usage_Start'])
      temp_schedule['Usage_End'] = pd.to_datetime(temp_schedule['Usage_End'])
      temp_schedule.sort_values(by='Usage_Start')
      schedule = temp_schedule[(temp_schedule['Usage_Start']>=Start_Date) & (temp_schedule['Usage_End']<=End_Date)]
    else:
      temp_schedule = get_ingested_data(file_path='merged_output_repository/merged_outputs.csv',datasource_type=data_source,connection_uri=connection_string,container_name='input')
      temp_schedule['Usage_Start'] = pd.to_datetime(temp_schedule['Usage_Start'])
      temp_schedule['Usage_End'] = pd.to_datetime(temp_schedule['Usage_End'])
      schedule = temp_schedule[(temp_schedule['Usage_Start']>=Start_Date) & (temp_schedule['Usage_End']<=End_Date)]
else:
    schedule =  get_ingested_data(file_path='merged_output_repository/latest_output.csv',datasource_type=data_source,connection_uri=connection_string,container_name='input')

dataset=schedule.reset_index(drop=True)
dataset['Clean_Start_Time']=pd.to_datetime(dataset['Clean_Start_Time'],format='%Y/%m/%d %H:%M:%S')
dataset['Usage_End']=pd.to_datetime(dataset['Usage_End'],format='%Y/%m/%d %H:%M:%S')
dataset['Clean_End_Time']=pd.to_datetime(dataset['Clean_End_Time'],format='%Y/%m/%d %H:%M:%S')
dataset['Usage_Start']=pd.to_datetime(dataset['Usage_Start'],format='%Y/%m/%d %H:%M:%S')
dataset['MC Group'] = dataset['MC Group'].fillna(1)
dataset['Missed Cleanings'] = np.where(dataset['MC Group']==1,1,0)
def business_metrics_resource(dataset):
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
                                                            'Missed Cleanings':'sum'
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
    return bm
def business_metrics_CIP(dataset):
    if dataset['Clean_Start_Time'].min() > dataset['Usage_Start'].min():
        working_hours = (dataset['Clean_End_Time'].max()-dataset['Usage_Start'].min())/ (np.timedelta64(1, 'h'))
    else:
        working_hours = (dataset['Clean_End_Time'].max()-dataset['Clean_Start_Time'].min())/ (np.timedelta64(1, 'h'))
    dataset['timebw'] = dataset['Clean_Start_Time'] - dataset['Usage_End']
    dataset['timebw'] = dataset['timebw'] / np.timedelta64(1, 'h')
    dataset['setup_flag'] = np.where(dataset['timebw'] > 0.166 , 1, 0)
    dataset['setup_flag'] = dataset['setup_flag']*10
    dataset = dataset.rename(columns = {'setup_flag': 'Setup time'})
    dataset['Setup time'] = (dataset['Setup time']/60).round(1)
    metrics_cip = dataset.groupby(['MC Group','Constraint'], as_index = False).agg({
                                                   'Resource': pd.Series.nunique, 
                                                    'clean_flag': 'sum', 
                                                   'Cleaning Duration':'sum',
                                                    'Setup time':'sum',
                                                    'Parallel Clean Flag':pd.Series.nunique
                                                    })
    metrics_cip = metrics_cip.rename(columns = {'Resource': 'Resources cleaned',
                               'clean_flag':'No. of cleanings','Cleaning Duration':'CIP used Duration',
                               })
    metrics_cip['No. of cleanings'] = metrics_cip['No. of cleanings'] - metrics_cip['Parallel Clean Flag']
    metrics_cip['Available Hours']= working_hours
    metrics_cip['CIP idle time'] = (metrics_cip['Available Hours'] - metrics_cip['CIP used Duration']).round(1)
    metrics_cip['CIP idle time%'] = ((metrics_cip['CIP idle time']/(metrics_cip['CIP used Duration']+metrics_cip['Setup time']+metrics_cip['CIP idle time']))*100).round(1)
    metrics_cip['CIP Usage%'] = (((metrics_cip['CIP used Duration']+metrics_cip['Setup time'])/(metrics_cip['CIP used Duration']+metrics_cip['Setup time']+metrics_cip['CIP idle time']))*100).round(1)

    metrics_cip_total = metrics_cip.groupby(['MC Group'], as_index = False).agg({
                                                   'Resources cleaned': 'sum',
                                                    'No. of cleanings': 'sum', #No of cleanings
                                                   'CIP used Duration':'sum',#CIP used duration
                                                    'Setup time':'sum',
                                                    'Available Hours':'max',
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
        df1=df1.append({'MC Group':i,'Min Downtime between 2 CIP cleanings':inter_arrival_time},ignore_index=True)
        df1['Min Downtime between 2 CIP cleanings']=df1['Min Downtime between 2 CIP cleanings'].round(2)
        df1 = df1.dropna()
    metrics_cip_total = pd.merge(metrics_cip_total,df1,on=['MC Group'])
    return metrics_cip,metrics_cip_total
df_res = business_metrics_resource(dataset)
df_pipe,df_cip = business_metrics_CIP(dataset)
non_missed = dataset[dataset['MC Group'].isin(['MC1','MC2','MC3','MC4'])]
def pre_re_cleaning(dataframe):   
    df1 = dataframe[(dataframe['preclean']==1)]
    df1['Cleaning Duration'] = df1['Cleaning Duration']/60
    data_temp = pd.DataFrame()
    for res in dataframe.Resource.unique():
        temp_df = dataframe[dataframe['Resource']==res]
        y=temp_df.min()['Usage_Start']
        data_temp = data_temp.append({"Resource":res,'Min_Usage_Start':y},ignore_index=True)
    new = pd.merge(df1,data_temp,on=['Resource'])
    new['CHT 1st U']=np.where((new['Usage_Start']== new['Min_Usage_Start']),'Pre-cleaning','Re-cleaning')
    x = new[new['CHT 1st U']=='Pre-cleaning'].shape[0]
    y =new[new['CHT 1st U']=='Re-cleaning'].shape[0]
    pre_re = pd.DataFrame([['Total no. of Pre-cleanings ',x],['Total no. of Re-cleanings',y]],columns=['Insights','Count'])
    return pre_re,new 
m,new1 = pre_re_cleaning(non_missed)
new=new1[new1['CHT 1st U']=='Pre-cleaning']
new= new.groupby('Resource',as_index = False).agg({'Cleaning Duration':'count'})
new= new.rename(columns = {'Cleaning Duration': 'Count'})
new['Resource'] = np.where( ((new['Resource'].str[:2].isin(['AS']))== True) , new['Resource'].str[-5:],new['Resource'])
new['Resource'] = np.where( ((new['Resource'].str[:4].isin(['Long']))== True) , new['Resource'].str[-5:],new['Resource'])
fig_pre_clean = px.bar(new,
             y='Count',
             x='Resource', labels={'x':'Resource','y':'# Pre-Cleanings'}, text= 'Count',height=700).update_xaxes(categoryorder="total descending")
fig_pre_clean.update_traces(texttemplate='%{text:0f}',textposition='outside',width=0.3,cliponaxis = False)
fig_pre_clean.update_yaxes(nticks=3)
fig_pre_clean.update_layout(xaxis_title='Resource',yaxis_title='# Pre-Cleanings')
fig_pre_clean.update_yaxes(nticks=3)
dynamic_outputs=plotly.io.to_json(fig_pre_clean)
"""


# In[18]:


kpi_res_reclean="""
import pandas as pd
from pathlib import Path
from azure.storage.blob import BlockBlobService
from io import StringIO
import json
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly


from azure.cosmosdb.table.tableservice import TableService
from pathlib import Path
from azure.storage.blob import BlockBlobService
from io import StringIO
from datetime import datetime, timedelta
import os
import pandas as pd
import json
from dateutil import parser
# from azure.identity import DefaultAzureCredential

key_value = 's94v9Q0+SXPPRagBKJUfB2WOA2Ts5mFPUb4sUg+C0tRzagJel6eYeeyrA7Dlz7s2mmPorbtOJQohLgjefo++Vw=='
connection_string = 'DefaultEndpointsProtocol=https;AccountName=mathcotakedastorage;AccountKey=s94v9Q0+SXPPRagBKJUfB2WOA2Ts5mFPUb4sUg+C0tRzagJel6eYeeyrA7Dlz7s2mmPorbtOJQohLgjefo++Vw==;EndpointSuffix=core.windows.net'
accountname = 'mathcotakedastorage'
data_source = 'azure_blob_storage'

def get_ingested_data(file_path, datasource_type, connection_uri, container_name):
    path = Path(file_path)
    block_blob_service = BlockBlobService(connection_string=connection_uri)
    blob_data = block_blob_service.get_blob_to_text(container_name=container_name, 
                                                    blob_name=file_path)
    ingested_df = pd.read_csv(StringIO(blob_data.content))
    return ingested_df
block_blob_service = BlockBlobService(connection_string=connection_string)
blob_data = block_blob_service.get_blob_to_text(container_name='input1', blob_name='Scheduler_Input/schedule_input.json')
f = pd.read_json(StringIO(blob_data.content))

#Start Date UTC and round handling
Start_Date = f['data']['Start Date'][0:19]
Start_Date=parser.parse(datetime.strptime(Start_Date ,'%Y-%m-%dT%H:%M:%S').strftime('%Y-%m-%dT%H:%M:%S'))
Start_Date=datetime.strftime(Start_Date ,'%Y-%m-%d')
Start_Date=parser.parse(datetime.strptime(Start_Date ,'%Y-%m-%d').strftime('%Y-%m-%d'))
#End Date UTC and round handling
End_Date = f['data']['End Date'][0:19]
End_Date=parser.parse(datetime.strptime(End_Date ,'%Y-%m-%dT%H:%M:%S').strftime('%Y-%m-%dT%H:%M:%S'))
End_Date=datetime.strftime(End_Date ,'%Y-%m-%d')
End_Date=parser.parse(datetime.strptime(End_Date ,'%Y-%m-%d').strftime('%Y-%m-%d'))+timedelta(hours=23.99)
# All the metadata is stored into table
table_service= TableService(account_name='mathcotakedastorage', account_key='s94v9Q0+SXPPRagBKJUfB2WOA2Ts5mFPUb4sUg+C0tRzagJel6eYeeyrA7Dlz7s2mmPorbtOJQohLgjefo++Vw==')
tasks=table_service.query_entities('schedule', f"PartitionKey eq 'schedule' and type eq 'Planned'")
table=pd.DataFrame(tasks)
# Converting start date and end date in metadata df to datetime
table['start_at'] = pd.to_datetime(table['start_at'],utc=True).dt.date
table['end_at'] = pd.to_datetime(table['end_at'],utc=True).dt.date
table['start_at']=pd.to_datetime(table['start_at'])
table['end_at']=pd.to_datetime(table['end_at'])
# Query df such that we get file path of data with schedule date between the given date on UI
# path eg = 'Src1/300921_Schedule/2021-09-04T04-10/output/20210930_2021-09-04T04-10_Optimized_Schedule_Output_(R).csv'
if f['data']['use_latest_schedule']==True:
    queried_table = table[(table['start_at'] <= datetime.strftime(Start_Date,'%Y-%m-%d')) & (table['end_at'] >= datetime.strftime(End_Date ,'%Y-%m-%d'))]
    queried_table = queried_table.tail(1)
    queried_table.reset_index(inplace = True)
    queried_path = queried_table['result_file'][0]
    queried_path = queried_path[6:76] + '_Optimized_Schedule_Output_(R).csv'
    condition1=[(queried_table['start_at'] == datetime.strftime(Start_Date,'%Y-%m-%d')) & (queried_table['end_at'] == datetime.strftime(End_Date ,'%Y-%m-%d'))][0]
    condition2=[(queried_table['start_at'] < datetime.strftime(Start_Date,'%Y-%m-%d')) | (queried_table['end_at'] > datetime.strftime(End_Date ,'%Y-%m-%d'))][0]
    if condition1[0]==True:
      schedule = get_ingested_data(file_path=queried_path,datasource_type=data_source,connection_uri=connection_string,container_name='input')
    elif condition2[0]==True:
      temp_schedule = queried_path
      temp_schedule = get_ingested_data(file_path=queried_path,datasource_type=data_source,connection_uri=connection_string,container_name='input')
      temp_schedule['Usage_Start'] = pd.to_datetime(temp_schedule['Usage_Start'])
      temp_schedule['Usage_End'] = pd.to_datetime(temp_schedule['Usage_End'])
      temp_schedule.sort_values(by='Usage_Start')
      schedule = temp_schedule[(temp_schedule['Usage_Start']>=Start_Date) & (temp_schedule['Usage_End']<=End_Date)]
    else:
      temp_schedule = get_ingested_data(file_path='merged_output_repository/merged_outputs.csv',datasource_type=data_source,connection_uri=connection_string,container_name='input')
      temp_schedule['Usage_Start'] = pd.to_datetime(temp_schedule['Usage_Start'])
      temp_schedule['Usage_End'] = pd.to_datetime(temp_schedule['Usage_End'])
      schedule = temp_schedule[(temp_schedule['Usage_Start']>=Start_Date) & (temp_schedule['Usage_End']<=End_Date)]
else:
    schedule =  get_ingested_data(file_path='merged_output_repository/latest_output.csv',datasource_type=data_source,connection_uri=connection_string,container_name='input')

dataset=schedule.reset_index(drop=True)
dataset['Clean_Start_Time']=pd.to_datetime(dataset['Clean_Start_Time'],format='%Y/%m/%d %H:%M:%S')
dataset['Usage_End']=pd.to_datetime(dataset['Usage_End'],format='%Y/%m/%d %H:%M:%S')
dataset['Clean_End_Time']=pd.to_datetime(dataset['Clean_End_Time'],format='%Y/%m/%d %H:%M:%S')
dataset['Usage_Start']=pd.to_datetime(dataset['Usage_Start'],format='%Y/%m/%d %H:%M:%S')
dataset['MC Group'] = dataset['MC Group'].fillna(1)
dataset['Missed Cleanings'] = np.where(dataset['MC Group']==1,1,0)
def business_metrics_resource(dataset):
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
                                                            'Missed Cleanings':'sum'
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
    return bm
def business_metrics_CIP(dataset):
    if dataset['Clean_Start_Time'].min() > dataset['Usage_Start'].min():
        working_hours = (dataset['Clean_End_Time'].max()-dataset['Usage_Start'].min())/ (np.timedelta64(1, 'h'))
    else:
        working_hours = (dataset['Clean_End_Time'].max()-dataset['Clean_Start_Time'].min())/ (np.timedelta64(1, 'h'))
    dataset['timebw'] = dataset['Clean_Start_Time'] - dataset['Usage_End']
    dataset['timebw'] = dataset['timebw'] / np.timedelta64(1, 'h')
    dataset['setup_flag'] = np.where(dataset['timebw'] > 0.166 , 1, 0)
    dataset['setup_flag'] = dataset['setup_flag']*10
    dataset = dataset.rename(columns = {'setup_flag': 'Setup time'})
    dataset['Setup time'] = (dataset['Setup time']/60).round(1)
    metrics_cip = dataset.groupby(['MC Group','Constraint'], as_index = False).agg({
                                                   'Resource': pd.Series.nunique, 
                                                    'clean_flag': 'sum', 
                                                   'Cleaning Duration':'sum',
                                                    'Setup time':'sum',
                                                    'Parallel Clean Flag':pd.Series.nunique
                                                    })
    metrics_cip = metrics_cip.rename(columns = {'Resource': 'Resources cleaned',
                               'clean_flag':'No. of cleanings','Cleaning Duration':'CIP used Duration',
                               })
    metrics_cip['No. of cleanings'] = metrics_cip['No. of cleanings'] - metrics_cip['Parallel Clean Flag']
    metrics_cip['Available Hours']= working_hours
    metrics_cip['CIP idle time'] = (metrics_cip['Available Hours'] - metrics_cip['CIP used Duration']).round(1)
    metrics_cip['CIP idle time%'] = ((metrics_cip['CIP idle time']/(metrics_cip['CIP used Duration']+metrics_cip['Setup time']+metrics_cip['CIP idle time']))*100).round(1)
    metrics_cip['CIP Usage%'] = (((metrics_cip['CIP used Duration']+metrics_cip['Setup time'])/(metrics_cip['CIP used Duration']+metrics_cip['Setup time']+metrics_cip['CIP idle time']))*100).round(1)

    metrics_cip_total = metrics_cip.groupby(['MC Group'], as_index = False).agg({
                                                   'Resources cleaned': 'sum',
                                                    'No. of cleanings': 'sum', #No of cleanings
                                                   'CIP used Duration':'sum',#CIP used duration
                                                    'Setup time':'sum',
                                                    'Available Hours':'max',
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
        df1=df1.append({'MC Group':i,'Min Downtime between 2 CIP cleanings':inter_arrival_time},ignore_index=True)
        df1['Min Downtime between 2 CIP cleanings']=df1['Min Downtime between 2 CIP cleanings'].round(2)
        df1 = df1.dropna()
    metrics_cip_total = pd.merge(metrics_cip_total,df1,on=['MC Group'])
    return metrics_cip,metrics_cip_total
df_res = business_metrics_resource(dataset)
df_pipe,df_cip = business_metrics_CIP(dataset)
non_missed = dataset[dataset['MC Group'].isin(['MC1','MC2','MC3','MC4'])]
def pre_re_cleaning(dataframe):   
    df1 = dataframe[(dataframe['preclean']==1)]
    df1['Cleaning Duration'] = df1['Cleaning Duration']/60
    data_temp = pd.DataFrame()
    for res in dataframe.Resource.unique():
        temp_df = dataframe[dataframe['Resource']==res]
        y=temp_df.min()['Usage_Start']
        data_temp = data_temp.append({"Resource":res,'Min_Usage_Start':y},ignore_index=True)
    new = pd.merge(df1,data_temp,on=['Resource'])
    new['CHT 1st U']=np.where((new['Usage_Start']== new['Min_Usage_Start']),'Pre-cleaning','Re-cleaning')
    x = new[new['CHT 1st U']=='Pre-cleaning'].shape[0]
    y =new[new['CHT 1st U']=='Re-cleaning'].shape[0]
    pre_re = pd.DataFrame([['Total no. of Pre-cleanings ',x],['Total no. of Re-cleanings',y]],columns=['Insights','Count'])
    return pre_re,new 
m,new1 = pre_re_cleaning(non_missed)
new=new1[new1['CHT 1st U']=='Re-cleaning']
new= new.groupby('Resource',as_index = False).agg({'Cleaning Duration':'count'})
new= new.rename(columns = {'Cleaning Duration': 'Count'})
new['Resource'] = np.where( ((new['Resource'].str[:2].isin(['AS']))== True) , new['Resource'].str[-5:],new['Resource'])
new['Resource'] = np.where( ((new['Resource'].str[:4].isin(['Long']))== True) , new['Resource'].str[-5:],new['Resource'])
fig_pre_clean = px.bar(new,
             y='Count',
             x='Resource', labels={'x':'Resource','y':'# Pre-Cleanings'}, text= 'Count',height=700).update_xaxes(categoryorder="total descending")
fig_pre_clean.update_traces(texttemplate='%{text:0f}',textposition='outside',width=0.3,cliponaxis = False)
fig_pre_clean.update_yaxes(nticks=3)
fig_pre_clean.update_layout(xaxis_title='Resource',yaxis_title='# Re-Cleanings')
dynamic_outputs=plotly.io.to_json(fig_pre_clean)
"""


# # CIP Utilization Metrics Code String #

# In[19]:


filter_code_string_CIP_Metrics="""
import pandas as pd
from pathlib import Path
from azure.storage.blob import BlockBlobService
from io import StringIO
import json

data_source = 'azure_blob_storage'
connection_string = 'DefaultEndpointsProtocol=https;AccountName=mathcotakedastorage;AccountKey=s94v9Q0+SXPPRagBKJUfB2WOA2Ts5mFPUb4sUg+C0tRzagJel6eYeeyrA7Dlz7s2mmPorbtOJQohLgjefo++Vw==;EndpointSuffix=core.windows.net'

def get_ingested_data(file_path, datasource_type, connection_uri, container_name):
    path = Path(file_path)
    block_blob_service = BlockBlobService(connection_string=connection_uri)
    blob_data = block_blob_service.get_blob_to_text(container_name=container_name, 
                                                    blob_name=file_path)
    ingested_df = pd.read_csv(StringIO(blob_data.content))

    return ingested_df


def get_filter_options(df= pd.DataFrame(), on_col="", using_cols={}, default_val='All'):
    cols_in_df = list(df.columns)

    filter_options = []
    if on_col in cols_in_df:
        if len(using_cols) == 0:
            filter_options = list(df[on_col].unique())
        else:
            for key in using_cols.keys():
                if key in cols_in_df:
                    if type(using_cols[key]) == str or type(using_cols[key]) == int :
                        if using_cols[key] == 'All':
                            continue
                        else:
                            df = df[df[key] == using_cols[key]]
                    else:
                        df = df[df[key].isin(using_cols[key])]
                else:
                    continue
        filter_options = list(df[on_col].unique())
    if default_val:
        filter_options.insert(0, default_val)
    return filter_options

# Reference
list_of_cols = ['MC Group','Date']
using_cols_for_test = {}

schedule=get_ingested_data(file_path='merged_output_repository/merged_outputs.csv',datasource_type=data_source,connection_uri=connection_string,container_name='input')
schedule.dropna(subset = ["MC Group"], inplace=True)
def_options_for_filters= {}
for x in list_of_cols:
       #\print('------------------'+x+'---------------------------')
    if x == 'MC Group':
        def_options_for_filters[x]=get_filter_options(schedule, on_col=x, using_cols=using_cols_for_test)
    elif x== 'Date':
        def_options_for_filters[x]={'start_date':str(pd.to_datetime(schedule['Clean_Start_Time'],infer_datetime_format=True,utc=True).min()), 'end_date': str(pd.to_datetime(schedule['Clean_End_Time'],infer_datetime_format=True,utc=True).max())}
    else:
        def_options_for_filters[x]={}


fil = {
    'Task': {
        'index': 0,
        'label': 'MC Group',
        'type': 'multiple',
        'options': def_options_for_filters['MC Group']
    },
    'Date':{
        'index':1,
        'label':'Date Range',
        'type':'date_range',
        'options':[]
    }
}



def generate_filter_json(current_filter_params,filter_options=fil, default_values=def_options_for_filters):
    basic_filter_dict = {
        "widget_filter_index": 0,
        "widget_filter_function": False,
        "widget_filter_function_parameter": False,
        "widget_filter_hierarchy_key": False,
        "widget_filter_isall": False,
        "widget_filter_multiselect": False,
        "widget_tag_input_type": "select",
        "widget_tag_key": "",
        "widget_tag_label": "",
        "widget_tag_value": [],
        "widget_filter_type": "",
        "widget_filter_params":None
    }
    dataValues = []
    defaultValues = {}

    for filter in filter_options.keys():
        instance_dict = dict(basic_filter_dict)
        instance_dict['widget_tag_key'] = filter
        instance_dict['widget_filter_index'] = filter_options[filter]['index']
        instance_dict['widget_tag_label'] = filter_options[filter]['label']
        instance_dict['widget_tag_value'] = filter_options[filter]['options']
        instance_dict['widget_filter_multiselect'] = True if filter_options[filter]['type'] == 'multiple' else False
        dataValues.append(instance_dict)
        if filter_options[filter]['type']=='date_range':
            instance_dict['widget_filter_type']='date_range'
            instance_dict['widget_filter_params']={'start_date':{'format':"DD/MM/yyyy"},'end_date':{'format':"DD/MM/yyyy"}}
        if current_filter_params=={}:
          defaultValues[filter] = default_values.get(
            filter, ['All'] if instance_dict['widget_filter_multiselect'] else 'All')
        else:
          defaultValues[filter] = current_filter_params['selected'].get(
            filter, ['All'] if instance_dict['widget_filter_multiselect'] else 'All')
    final_json = {'dataValues': dataValues, 'defaultValues': defaultValues}
    return final_json

dynamic_outputs=json.dumps(generate_filter_json(current_filter_params,filter_options=fil, default_values=def_options_for_filters))
print(dynamic_outputs)
"""


# In[20]:


kpi_cip_utilization_overall="""
import pandas as pd
from pathlib import Path
from azure.storage.blob import BlockBlobService
from io import StringIO
import json
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly

data_source = 'azure_blob_storage'
connection_string = 'DefaultEndpointsProtocol=https;AccountName=mathcotakedastorage;AccountKey=s94v9Q0+SXPPRagBKJUfB2WOA2Ts5mFPUb4sUg+C0tRzagJel6eYeeyrA7Dlz7s2mmPorbtOJQohLgjefo++Vw==;EndpointSuffix=core.windows.net'

def get_ingested_data(file_path, datasource_type, connection_uri, container_name):
    path = Path(file_path)
    block_blob_service = BlockBlobService(connection_string=connection_uri)
    blob_data = block_blob_service.get_blob_to_text(container_name=container_name, 
                                                    blob_name=file_path)
    ingested_df = pd.read_csv(StringIO(blob_data.content))

    return ingested_df

 
data_source = 'azure_blob_storage'
connection_string = 'DefaultEndpointsProtocol=https;AccountName=mathcotakedastorage;AccountKey=s94v9Q0+SXPPRagBKJUfB2WOA2Ts5mFPUb4sUg+C0tRzagJel6eYeeyrA7Dlz7s2mmPorbtOJQohLgjefo++Vw==;EndpointSuffix=core.windows.net' 
def get_ingested_data(file_path, datasource_type, connection_uri, container_name):
    path = Path(file_path)
    block_blob_service = BlockBlobService(connection_string=connection_uri)
    blob_data = block_blob_service.get_blob_to_text(container_name=container_name, 
                                                    blob_name=file_path)
    ingested_df = pd.read_csv(StringIO(blob_data.content))
 
    return ingested_df

schedule=get_ingested_data(file_path='merged_output_repository/merged_outputs.csv',datasource_type=data_source,connection_uri=connection_string,container_name='input')
actuals=get_ingested_data(file_path='merged_output_repository/merged_actual_outputs.csv',datasource_type=data_source,connection_uri=connection_string,container_name='input')
def filter_data(selected_filter, df):
    cols_in_df = list(df.columns)
    # Filtering data
    if len(selected_filters) > 0:
        for key in selected_filters.keys():
            if type(selected_filters[key])==dict and selected_filters[key].get('start_date',False) and selected_filters[key].get('end_date',False):
                selected_filters[key]['start_date']=pd.to_datetime(selected_filters[key]['start_date'],infer_datetime_format=True,utc=True)
                selected_filters[key]['end_date']=pd.to_datetime(selected_filters[key]['end_date'],infer_datetime_format=True,utc=True)
                print(selected_filters)
                df=df[(df['Clean_Start_Time']>selected_filters[key]['start_date']) & (df['Clean_End_Time']<selected_filters[key]['end_date'])]
            elif key in cols_in_df:
                if type(selected_filter[key]) == str or type(selected_filter[key]) == int:
                    if selected_filter[key] == 'All':
                        continue
                    else:
                        df = df[df[key] == selected_filter[key]]
                else:
                    if isinstance(selected_filter[key],list) and 'All' in selected_filter[key]:
                        continue
                    else:
                        df = df[df[key].isin(selected_filter[key])]
            else:
                continue
    return df
schedule.Usage_Start= pd.to_datetime(schedule.Usage_Start, format = '%Y/%m/%d %H:%M:%S',utc=True)
schedule.Usage_End= pd.to_datetime(schedule.Usage_End, format = '%Y/%m/%d %H:%M:%S')
schedule['Next Usage']= pd.to_datetime(schedule['Next Usage'], format = '%Y/%m/%d %H:%M:%S')
schedule['End DHT Time']= pd.to_datetime(schedule['End DHT Time'], format = '%Y/%m/%d %H:%M:%S')
schedule['End CHT Time']= pd.to_datetime(schedule['End CHT Time'], format = '%Y/%m/%d %H:%M:%S')
schedule.Clean_Start_Time= pd.to_datetime(schedule.Clean_Start_Time, format = '%Y/%m/%d %H:%M:%S',utc=True)
schedule.Clean_End_Time= pd.to_datetime(schedule.Clean_End_Time, format = '%Y/%m/%d %H:%M:%S',utc=True)
actuals['Cleaning Duration'] = actuals['Cleaning Duration'] / 60
actuals.Clean_Start_Time= pd.to_datetime(actuals.Clean_Start_Time, format = '%Y/%m/%d %H:%M:%S',utc=True)
actuals.Clean_End_Time= pd.to_datetime(actuals.Clean_End_Time, format = '%Y/%m/%d %H:%M:%S',utc=True)
schedule=filter_data(selected_filters,schedule)
actuals=filter_data(selected_filters,actuals)
actuals = actuals[actuals['Cleaning_Type']=='Standard']
schedule['Cleaning Duration'] = schedule['Cleaning Duration'] / 60

if schedule['Clean_Start_Time'].min() > schedule['Usage_Start'].min():
        working_hours_schedule = (schedule['Clean_End_Time'].max()-schedule['Usage_Start'].min())/ (np.timedelta64(1, 'h'))
else:
        working_hours_schedule = (schedule['Clean_End_Time'].max()-schedule['Clean_Start_Time'].min())/ (np.timedelta64(1, 'h'))
actuals = actuals[['MC Group','Clean_Start_Time','Clean_End_Time','clean_flag','Cleaning Duration']]
working_hours_actuals = (actuals['Clean_End_Time'].max()-actuals['Clean_Start_Time'].min())/ (np.timedelta64(1, 'h'))
metrics_cip_schedule = schedule.groupby(['MC Group'], as_index = False).agg({
                                            
                                                'clean_flag': 'sum',
                                            'Cleaning Duration':'sum'
                                                })
metrics_cip_schedule = metrics_cip_schedule.rename(columns = {'Resource': 'Resources cleaned',
                        'clean_flag':'No. of cleanings','Cleaning Duration':'CIP used Duration',
                        })
metrics_cip_actuals = actuals.groupby(['MC Group'], as_index = False).agg({
                                            
                                                'clean_flag': 'sum',
                                            'Cleaning Duration':'sum'
                                                })
metrics_cip_actuals = metrics_cip_actuals.rename(columns = {'Resource': 'Resources cleaned',
                        'clean_flag':'No. of cleanings','Cleaning Duration':'CIP used Duration',
                        })
metrics_cip_schedule['Total Available Hours'] = working_hours_schedule
metrics_cip_actuals['Total Available Hours'] = working_hours_actuals
metrics_cip_schedule['Usage %'] = round((metrics_cip_schedule['CIP used Duration']/ metrics_cip_schedule['Total Available Hours'])*100,0)
metrics_cip_actuals['Usage %'] = round((metrics_cip_actuals['CIP used Duration']/ metrics_cip_actuals['Total Available Hours'])*100,0)

#total cleaning cycles
total_cip_cleanings_schedule =round(metrics_cip_schedule['No. of cleanings'].sum(),0)
total_cip_cleanings_actuals =round(metrics_cip_actuals['No. of cleanings'].sum(),0)

#Cost for cleaning
cleaning_cost_planned = total_cip_cleanings_schedule* 25 
cleaning_cost_actuals = total_cip_cleanings_actuals* 25 

cip_cleaning_kpi= {
  "insight_data": [
    {
      "label": "Planned",
      "value": str(int(total_cip_cleanings_schedule))
    },
    {
      "label": "Actual",
      "severity": "success",
      "value": str(int(total_cip_cleanings_actuals))
    },
    { 
      "header":"Cost of Cleaning", 
      "header_style": {"textDecoration": "none"},     
      "label": "Planned",
      "value": "€ " + str(int(cleaning_cost_planned))
    },
    {
      "label": "Actual",
      "severity": "success",
      "value": "€ " + str(int(cleaning_cost_actuals))
    }
  ],
  "insight_label": "Total Cleaning Cycles"
}
dynamic_outputs=json.dumps(cip_cleaning_kpi)
print(dynamic_outputs)
"""


# In[21]:


kpi_cip_duration_overall="""
import pandas as pd
from pathlib import Path
from azure.storage.blob import BlockBlobService
from io import StringIO
import json
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly

data_source = 'azure_blob_storage'
connection_string = 'DefaultEndpointsProtocol=https;AccountName=mathcotakedastorage;AccountKey=s94v9Q0+SXPPRagBKJUfB2WOA2Ts5mFPUb4sUg+C0tRzagJel6eYeeyrA7Dlz7s2mmPorbtOJQohLgjefo++Vw==;EndpointSuffix=core.windows.net'

def get_ingested_data(file_path, datasource_type, connection_uri, container_name):
    path = Path(file_path)
    block_blob_service = BlockBlobService(connection_string=connection_uri)
    blob_data = block_blob_service.get_blob_to_text(container_name=container_name, 
                                                    blob_name=file_path)
    ingested_df = pd.read_csv(StringIO(blob_data.content))

    return ingested_df

 
data_source = 'azure_blob_storage'
connection_string = 'DefaultEndpointsProtocol=https;AccountName=mathcotakedastorage;AccountKey=s94v9Q0+SXPPRagBKJUfB2WOA2Ts5mFPUb4sUg+C0tRzagJel6eYeeyrA7Dlz7s2mmPorbtOJQohLgjefo++Vw==;EndpointSuffix=core.windows.net' 
def get_ingested_data(file_path, datasource_type, connection_uri, container_name):
    path = Path(file_path)
    block_blob_service = BlockBlobService(connection_string=connection_uri)
    blob_data = block_blob_service.get_blob_to_text(container_name=container_name, 
                                                    blob_name=file_path)
    ingested_df = pd.read_csv(StringIO(blob_data.content))
 
    return ingested_df

schedule=get_ingested_data(file_path='merged_output_repository/merged_outputs.csv',datasource_type=data_source,connection_uri=connection_string,container_name='input')
actuals=get_ingested_data(file_path='merged_output_repository/merged_actual_outputs.csv',datasource_type=data_source,connection_uri=connection_string,container_name='input')
def filter_data(selected_filter, df):
    cols_in_df = list(df.columns)
    # Filtering data
    if len(selected_filters) > 0:
        for key in selected_filters.keys():
            if type(selected_filters[key])==dict and selected_filters[key].get('start_date',False) and selected_filters[key].get('end_date',False):
                selected_filters[key]['start_date']=pd.to_datetime(selected_filters[key]['start_date'],infer_datetime_format=True,utc=True)
                selected_filters[key]['end_date']=pd.to_datetime(selected_filters[key]['end_date'],infer_datetime_format=True,utc=True)
                print(selected_filters)
                df=df[(df['Clean_Start_Time']>selected_filters[key]['start_date']) & (df['Clean_End_Time']<selected_filters[key]['end_date'])]
            elif key in cols_in_df:
                if type(selected_filter[key]) == str or type(selected_filter[key]) == int:
                    if selected_filter[key] == 'All':
                        continue
                    else:
                        df = df[df[key] == selected_filter[key]]
                else:
                    if isinstance(selected_filter[key],list) and 'All' in selected_filter[key]:
                        continue
                    else:
                        df = df[df[key].isin(selected_filter[key])]
            else:
                continue
    return df
schedule.Usage_Start= pd.to_datetime(schedule.Usage_Start, format = '%Y/%m/%d %H:%M:%S',utc=True)
schedule.Usage_End= pd.to_datetime(schedule.Usage_End, format = '%Y/%m/%d %H:%M:%S')
schedule['Next Usage']= pd.to_datetime(schedule['Next Usage'], format = '%Y/%m/%d %H:%M:%S')
schedule['End DHT Time']= pd.to_datetime(schedule['End DHT Time'], format = '%Y/%m/%d %H:%M:%S')
schedule['End CHT Time']= pd.to_datetime(schedule['End CHT Time'], format = '%Y/%m/%d %H:%M:%S')
schedule.Clean_Start_Time= pd.to_datetime(schedule.Clean_Start_Time, format = '%Y/%m/%d %H:%M:%S',utc=True)
schedule.Clean_End_Time= pd.to_datetime(schedule.Clean_End_Time, format = '%Y/%m/%d %H:%M:%S',utc=True)
actuals['Cleaning Duration'] = actuals['Cleaning Duration'] / 60
actuals.Clean_Start_Time= pd.to_datetime(actuals.Clean_Start_Time, format = '%Y/%m/%d %H:%M:%S',utc=True)
actuals.Clean_End_Time= pd.to_datetime(actuals.Clean_End_Time, format = '%Y/%m/%d %H:%M:%S',utc=True)
schedule=filter_data(selected_filters,schedule)
actuals=filter_data(selected_filters,actuals)
actuals = actuals[actuals['Cleaning_Type']=='Standard']
schedule['Cleaning Duration'] = schedule['Cleaning Duration'] / 60

if schedule['Clean_Start_Time'].min() > schedule['Usage_Start'].min():
        working_hours_schedule = (schedule['Clean_End_Time'].max()-schedule['Usage_Start'].min())/ (np.timedelta64(1, 'h'))
else:
        working_hours_schedule = (schedule['Clean_End_Time'].max()-schedule['Clean_Start_Time'].min())/ (np.timedelta64(1, 'h'))
actuals = actuals[['MC Group','Clean_Start_Time','Clean_End_Time','clean_flag','Cleaning Duration']]
working_hours_actuals = (actuals['Clean_End_Time'].max()-actuals['Clean_Start_Time'].min())/ (np.timedelta64(1, 'h'))
metrics_cip_schedule = schedule.groupby(['MC Group'], as_index = False).agg({
                                            
                                                'clean_flag': 'sum',
                                            'Cleaning Duration':'sum'
                                                })
metrics_cip_schedule = metrics_cip_schedule.rename(columns = {'Resource': 'Resources cleaned',
                        'clean_flag':'No. of cleanings','Cleaning Duration':'CIP used Duration',
                        })
metrics_cip_actuals = actuals.groupby(['MC Group'], as_index = False).agg({
                                            
                                                'clean_flag': 'sum',
                                            'Cleaning Duration':'sum'
                                                })
metrics_cip_actuals = metrics_cip_actuals.rename(columns = {'Resource': 'Resources cleaned',
                        'clean_flag':'No. of cleanings','Cleaning Duration':'CIP used Duration',
                        })
metrics_cip_schedule['Total Available Hours'] = working_hours_schedule
metrics_cip_actuals['Total Available Hours'] = working_hours_actuals
metrics_cip_schedule['Usage %'] = round((metrics_cip_schedule['CIP used Duration']/ metrics_cip_schedule['Total Available Hours'])*100,0)
metrics_cip_actuals['Usage %'] = round((metrics_cip_actuals['CIP used Duration']/ metrics_cip_actuals['Total Available Hours'])*100,0)

#average CIP utilization
if len(metrics_cip_schedule)!=0:
    cip_util_schedule = round((metrics_cip_schedule['CIP used Duration'].sum() /  metrics_cip_schedule['Total Available Hours'].sum())*100,0)
else:
    cip_util_schedule=0

if len(metrics_cip_actuals)!=0:
    cip_util_actuals = round((metrics_cip_actuals['CIP used Duration'].sum() /  metrics_cip_actuals['Total Available Hours'].sum())*100,0)
else:
    cip_util_actuals=0

#RO consumption

   #total cleaning cycles
total_cip_cleanings_schedule =round(metrics_cip_schedule['No. of cleanings'].sum(),0)
total_cip_cleanings_actuals =round(metrics_cip_actuals['No. of cleanings'].sum(),0)

ro_consumption_planned = total_cip_cleanings_schedule*3.2 
ro_consumption_actual = total_cip_cleanings_actuals*3.2

average_cip_kpi= {
  "insight_data": [
    {
      "label": "Planned",
      "value": str(int(cip_util_schedule))+"%"
    },
    {
      "label": "Actual",
      "severity": "success",
      "value": str(int(cip_util_actuals))+"%"
    },
    { 
      "header":"RO Consumption", 
      "header_style": {"textDecoration": "none"},     
      "label": "Planned",
      "value": str(int(ro_consumption_planned))+" kL"
    },
    {
      "label": "Actual",
      "severity": "success",
      "value": str(int(ro_consumption_actual))+" kL"
    }
  ],
  "insight_label": "Average CIP Utilization"
}
 
dynamic_outputs=json.dumps(average_cip_kpi)
print(dynamic_outputs)
"""


# In[22]:


kpi_clean_cycles_kpi="""
import pandas as pd
from pathlib import Path
from azure.storage.blob import BlockBlobService
from io import StringIO
import json
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly

data_source = 'azure_blob_storage'
connection_string = 'DefaultEndpointsProtocol=https;AccountName=mathcotakedastorage;AccountKey=s94v9Q0+SXPPRagBKJUfB2WOA2Ts5mFPUb4sUg+C0tRzagJel6eYeeyrA7Dlz7s2mmPorbtOJQohLgjefo++Vw==;EndpointSuffix=core.windows.net'

def get_ingested_data(file_path, datasource_type, connection_uri, container_name):
    path = Path(file_path)
    block_blob_service = BlockBlobService(connection_string=connection_uri)
    blob_data = block_blob_service.get_blob_to_text(container_name=container_name, 
                                                    blob_name=file_path)
    ingested_df = pd.read_csv(StringIO(blob_data.content))

    return ingested_df

 
data_source = 'azure_blob_storage'
connection_string = 'DefaultEndpointsProtocol=https;AccountName=mathcotakedastorage;AccountKey=s94v9Q0+SXPPRagBKJUfB2WOA2Ts5mFPUb4sUg+C0tRzagJel6eYeeyrA7Dlz7s2mmPorbtOJQohLgjefo++Vw==;EndpointSuffix=core.windows.net' 
def get_ingested_data(file_path, datasource_type, connection_uri, container_name):
    path = Path(file_path)
    block_blob_service = BlockBlobService(connection_string=connection_uri)
    blob_data = block_blob_service.get_blob_to_text(container_name=container_name, 
                                                    blob_name=file_path)
    ingested_df = pd.read_csv(StringIO(blob_data.content))
 
    return ingested_df

schedule=get_ingested_data(file_path='merged_output_repository/merged_outputs.csv',datasource_type=data_source,connection_uri=connection_string,container_name='input')
actuals=get_ingested_data(file_path='merged_output_repository/merged_actual_outputs.csv',datasource_type=data_source,connection_uri=connection_string,container_name='input')
def filter_data(selected_filter, df):
    cols_in_df = list(df.columns)
    # Filtering data
    if len(selected_filters) > 0:
        for key in selected_filters.keys():
            if type(selected_filters[key])==dict and selected_filters[key].get('start_date',False) and selected_filters[key].get('end_date',False):
                selected_filters[key]['start_date']=pd.to_datetime(selected_filters[key]['start_date'],infer_datetime_format=True,utc=True)
                selected_filters[key]['end_date']=pd.to_datetime(selected_filters[key]['end_date'],infer_datetime_format=True,utc=True)
                print(selected_filters)
                df=df[(df['Clean_Start_Time']>selected_filters[key]['start_date']) & (df['Clean_End_Time']<selected_filters[key]['end_date'])]
            elif key in cols_in_df:
                if type(selected_filter[key]) == str or type(selected_filter[key]) == int:
                    if selected_filter[key] == 'All':
                        continue
                    else:
                        df = df[df[key] == selected_filter[key]]
                else:
                    if isinstance(selected_filter[key],list) and 'All' in selected_filter[key]:
                        continue
                    else:
                        df = df[df[key].isin(selected_filter[key])]
            else:
                continue
    return df
schedule.Usage_Start= pd.to_datetime(schedule.Usage_Start, format = '%Y/%m/%d %H:%M:%S',utc=True)
schedule.Usage_End= pd.to_datetime(schedule.Usage_End, format = '%Y/%m/%d %H:%M:%S')
schedule['Next Usage']= pd.to_datetime(schedule['Next Usage'], format = '%Y/%m/%d %H:%M:%S')
schedule['End DHT Time']= pd.to_datetime(schedule['End DHT Time'], format = '%Y/%m/%d %H:%M:%S')
schedule['End CHT Time']= pd.to_datetime(schedule['End CHT Time'], format = '%Y/%m/%d %H:%M:%S')
schedule.Clean_Start_Time= pd.to_datetime(schedule.Clean_Start_Time, format = '%Y/%m/%d %H:%M:%S',utc=True)
schedule.Clean_End_Time= pd.to_datetime(schedule.Clean_End_Time, format = '%Y/%m/%d %H:%M:%S',utc=True)
actuals['Cleaning Duration'] = actuals['Cleaning Duration'] / 60
actuals.Clean_Start_Time= pd.to_datetime(actuals.Clean_Start_Time, format = '%Y/%m/%d %H:%M:%S',utc=True)
actuals.Clean_End_Time= pd.to_datetime(actuals.Clean_End_Time, format = '%Y/%m/%d %H:%M:%S',utc=True)
schedule=filter_data(selected_filters,schedule)
actuals=filter_data(selected_filters,actuals)
actuals = actuals[actuals['Cleaning_Type']=='Standard']
schedule['Cleaning Duration'] = schedule['Cleaning Duration'] / 60

if schedule['Clean_Start_Time'].min() > schedule['Usage_Start'].min():
        working_hours_schedule = (schedule['Clean_End_Time'].max()-schedule['Usage_Start'].min())/ (np.timedelta64(1, 'h'))
else:
        working_hours_schedule = (schedule['Clean_End_Time'].max()-schedule['Clean_Start_Time'].min())/ (np.timedelta64(1, 'h'))
actuals = actuals[['MC Group','Clean_Start_Time','Clean_End_Time','clean_flag','Cleaning Duration']]
working_hours_actuals = (actuals['Clean_End_Time'].max()-actuals['Clean_Start_Time'].min())/ (np.timedelta64(1, 'h'))
metrics_cip_schedule = schedule.groupby(['MC Group'], as_index = False).agg({
                                            
                                                'clean_flag': 'sum',
                                            'Cleaning Duration':'sum'
                                                })
metrics_cip_schedule = metrics_cip_schedule.rename(columns = {'Resource': 'Resources cleaned',
                        'clean_flag':'No. of cleanings','Cleaning Duration':'CIP used Duration',
                        })
metrics_cip_actuals = actuals.groupby(['MC Group'], as_index = False).agg({
                                            
                                                'clean_flag': 'sum',
                                            'Cleaning Duration':'sum'
                                                })
metrics_cip_actuals = metrics_cip_actuals.rename(columns = {'Resource': 'Resources cleaned',
                        'clean_flag':'No. of cleanings','Cleaning Duration':'CIP used Duration',
                        })
metrics_cip_schedule['Total Available Hours'] = working_hours_schedule
metrics_cip_actuals['Total Available Hours'] = working_hours_actuals
metrics_cip_schedule['Usage %'] = round((metrics_cip_schedule['CIP used Duration']/ metrics_cip_schedule['Total Available Hours'])*100,0)
metrics_cip_actuals['Usage %'] = round((metrics_cip_actuals['CIP used Duration']/ metrics_cip_actuals['Total Available Hours'])*100,0)

#total CIP utilization 
total_cip_util_schedule =round(metrics_cip_schedule['CIP used Duration'].sum(),0)
total_cip_util_actuals =round(metrics_cip_actuals['CIP used Duration'].sum(),0)

#WFI consumption
   #total cleaning cycles
total_cip_cleanings_schedule =round(metrics_cip_schedule['No. of cleanings'].sum(),0)
total_cip_cleanings_actuals =round(metrics_cip_actuals['No. of cleanings'].sum(),0)
wfi_consumption_planned = total_cip_cleanings_schedule*0.3 
wfi_consumption_actual = total_cip_cleanings_actuals*0.3

cip_duration_kpi= {
  "insight_data": [
    {
      "label": "Planned",
      "value": str(int(total_cip_util_schedule))
    },
    {
      "label": "Actual",
      "severity": "success",
      "value": str(int(total_cip_util_actuals))
    },
    { 
      "header":"WFI Consumption", 
      "header_style": {"textDecoration": "none"},     
      "label": "Planned",
      "value": str(int(wfi_consumption_planned))+" kL"
    },
    {
      "label": "Actual",
      "severity": "success",
      "value": str(int(wfi_consumption_actual))+" kL"
    }
  ],
  "insight_label": "CIP Usage Duration (Hours)"
}
dynamic_outputs=json.dumps(cip_duration_kpi)
print(dynamic_outputs)
"""


# In[23]:


kpi_possible_cycles_kpi="""
import pandas as pd
from pathlib import Path
from azure.storage.blob import BlockBlobService
from io import StringIO
import json
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly

data_source = 'azure_blob_storage'
connection_string = 'DefaultEndpointsProtocol=https;AccountName=mathcotakedastorage;AccountKey=s94v9Q0+SXPPRagBKJUfB2WOA2Ts5mFPUb4sUg+C0tRzagJel6eYeeyrA7Dlz7s2mmPorbtOJQohLgjefo++Vw==;EndpointSuffix=core.windows.net'

def get_ingested_data(file_path, datasource_type, connection_uri, container_name):
    path = Path(file_path)
    block_blob_service = BlockBlobService(connection_string=connection_uri)
    blob_data = block_blob_service.get_blob_to_text(container_name=container_name, 
                                                    blob_name=file_path)
    ingested_df = pd.read_csv(StringIO(blob_data.content))

    return ingested_df

 
data_source = 'azure_blob_storage'
connection_string = 'DefaultEndpointsProtocol=https;AccountName=mathcotakedastorage;AccountKey=s94v9Q0+SXPPRagBKJUfB2WOA2Ts5mFPUb4sUg+C0tRzagJel6eYeeyrA7Dlz7s2mmPorbtOJQohLgjefo++Vw==;EndpointSuffix=core.windows.net' 
def get_ingested_data(file_path, datasource_type, connection_uri, container_name):
    path = Path(file_path)
    block_blob_service = BlockBlobService(connection_string=connection_uri)
    blob_data = block_blob_service.get_blob_to_text(container_name=container_name, 
                                                    blob_name=file_path)
    ingested_df = pd.read_csv(StringIO(blob_data.content))
 
    return ingested_df

schedule=get_ingested_data(file_path='merged_output_repository/merged_outputs.csv',datasource_type=data_source,connection_uri=connection_string,container_name='input')
actuals=get_ingested_data(file_path='merged_output_repository/merged_actual_outputs.csv',datasource_type=data_source,connection_uri=connection_string,container_name='input')
def filter_data(selected_filter, df):
    cols_in_df = list(df.columns)
    # Filtering data
    if len(selected_filters) > 0:
        for key in selected_filters.keys():
            if type(selected_filters[key])==dict and selected_filters[key].get('start_date',False) and selected_filters[key].get('end_date',False):
                selected_filters[key]['start_date']=pd.to_datetime(selected_filters[key]['start_date'],infer_datetime_format=True,utc=True)
                selected_filters[key]['end_date']=pd.to_datetime(selected_filters[key]['end_date'],infer_datetime_format=True,utc=True)
                print(selected_filters)
                df=df[(df['Clean_Start_Time']>selected_filters[key]['start_date']) & (df['Clean_End_Time']<selected_filters[key]['end_date'])]
            elif key in cols_in_df:
                if type(selected_filter[key]) == str or type(selected_filter[key]) == int:
                    if selected_filter[key] == 'All':
                        continue
                    else:
                        df = df[df[key] == selected_filter[key]]
                else:
                    if isinstance(selected_filter[key],list) and 'All' in selected_filter[key]:
                        continue
                    else:
                        df = df[df[key].isin(selected_filter[key])]
            else:
                continue
    return df
schedule.Usage_Start= pd.to_datetime(schedule.Usage_Start, format = '%Y/%m/%d %H:%M:%S',utc=True)
schedule.Usage_End= pd.to_datetime(schedule.Usage_End, format = '%Y/%m/%d %H:%M:%S')
schedule['Next Usage']= pd.to_datetime(schedule['Next Usage'], format = '%Y/%m/%d %H:%M:%S')
schedule['End DHT Time']= pd.to_datetime(schedule['End DHT Time'], format = '%Y/%m/%d %H:%M:%S')
schedule['End CHT Time']= pd.to_datetime(schedule['End CHT Time'], format = '%Y/%m/%d %H:%M:%S')
schedule.Clean_Start_Time= pd.to_datetime(schedule.Clean_Start_Time, format = '%Y/%m/%d %H:%M:%S',utc=True)
schedule.Clean_End_Time= pd.to_datetime(schedule.Clean_End_Time, format = '%Y/%m/%d %H:%M:%S',utc=True)
actuals['Cleaning Duration'] = actuals['Cleaning Duration'] / 60
actuals.Clean_Start_Time= pd.to_datetime(actuals.Clean_Start_Time, format = '%Y/%m/%d %H:%M:%S',utc=True)
actuals.Clean_End_Time= pd.to_datetime(actuals.Clean_End_Time, format = '%Y/%m/%d %H:%M:%S',utc=True)
schedule=filter_data(selected_filters,schedule)
actuals=filter_data(selected_filters,actuals)
actuals = actuals[actuals['Cleaning_Type']=='Standard']
schedule['Cleaning Duration'] = schedule['Cleaning Duration'] / 60

if schedule['Clean_Start_Time'].min() > schedule['Usage_Start'].min():
        working_hours_schedule = (schedule['Clean_End_Time'].max()-schedule['Usage_Start'].min())/ (np.timedelta64(1, 'h'))
else:
        working_hours_schedule = (schedule['Clean_End_Time'].max()-schedule['Clean_Start_Time'].min())/ (np.timedelta64(1, 'h'))
actuals = actuals[['MC Group','Clean_Start_Time','Clean_End_Time','clean_flag','Cleaning Duration']]
working_hours_actuals = (actuals['Clean_End_Time'].max()-actuals['Clean_Start_Time'].min())/ (np.timedelta64(1, 'h'))
metrics_cip_schedule = schedule.groupby(['MC Group'], as_index = False).agg({
                                            
                                                'clean_flag': 'sum',
                                            'Cleaning Duration':'sum'
                                                })
metrics_cip_schedule = metrics_cip_schedule.rename(columns = {'Resource': 'Resources cleaned',
                        'clean_flag':'No. of cleanings','Cleaning Duration':'CIP used Duration',
                        })
metrics_cip_actuals = actuals.groupby(['MC Group'], as_index = False).agg({
                                            
                                                'clean_flag': 'sum',
                                            'Cleaning Duration':'sum'
                                                })
metrics_cip_actuals = metrics_cip_actuals.rename(columns = {'Resource': 'Resources cleaned',
                        'clean_flag':'No. of cleanings','Cleaning Duration':'CIP used Duration',
                        })
metrics_cip_schedule['Total Available Hours'] = working_hours_schedule
metrics_cip_actuals['Total Available Hours'] = working_hours_actuals
metrics_cip_schedule['Usage %'] = round((metrics_cip_schedule['CIP used Duration']/ metrics_cip_schedule['Total Available Hours'])*100,0)
metrics_cip_actuals['Usage %'] = round((metrics_cip_actuals['CIP used Duration']/ metrics_cip_actuals['Total Available Hours'])*100,0)

if len(schedule)!=0:
    non_missed = schedule[schedule['MC Group'].isin(['MC1','MC2','MC3','MC4'])]
    df2 = non_missed[['Resource','MC Group','Clean_Start_Time','Clean_End_Time','Cleaning Duration','Constraint']]
    df2 =df2.sort_values(by=['MC Group', 'Clean_Start_Time'])
    df2.drop_duplicates(subset=['Clean_Start_Time','Clean_End_Time','MC Group','Constraint'],keep = 'first', inplace = True)
    temp_df=pd.DataFrame()
    for mc in df2['MC Group'].unique():
        data_temp=df2[df2['MC Group']==mc]
        data_temp=data_temp.sort_values('Clean_Start_Time')
        data_temp['Idle Time']=data_temp['Clean_Start_Time']-data_temp['Clean_End_Time'].shift(1)
        data_temp['Idle Time']=data_temp['Idle Time']/(np.timedelta64(1,'s')*3600)
        data_temp['Average_Cleaning_Duration'] =data_temp['Cleaning Duration'].mean()
        temp_df=pd.concat([temp_df,data_temp])
    df2_x=temp_df.reset_index(drop=True)
    df2_x = df2_x.sort_values(by=['MC Group', 'Clean_Start_Time'])
    df2_x= df2_x.fillna(0)
    df2_x['Idle Time'] = np.where(df2_x['Idle Time']<0,0,df2_x['Idle Time'])
    cleaning_time =[]
    for i in range(len(df2_x['Idle Time'])):
        c= df2_x['Idle Time'][i]/df2_x['Average_Cleaning_Duration'][i]
        cleaning_time.append(int(c))
    df2_x['Cleaning_Possible_Time'] = cleaning_time 
    total_sum = df2_x.groupby(['MC Group'])['Cleaning_Possible_Time'].agg('sum')
    df3 = pd.DataFrame(total_sum, columns=['Cleaning_Possible_Time'] ).reset_index()
    metrics_cip_schedule['Cleaning Possible Time'] = df3['Cleaning_Possible_Time']
else:
    metrics_cip_schedule['Cleaning Possible Time']=0

if len(actuals)!=0:
    non_missed_actuals = actuals[actuals['MC Group'].isin(['MC1','MC2','MC3','MC4'])]
    df_actuals = non_missed_actuals[['MC Group','Clean_Start_Time','Clean_End_Time','Cleaning Duration', 'clean_flag']]
    df_actuals =df_actuals.sort_values(by=['MC Group', 'Clean_Start_Time'])
    df_actuals.drop_duplicates(subset=['Clean_Start_Time','Clean_End_Time','MC Group'],keep = 'first', inplace = True)
    temp_df_actuals=pd.DataFrame()
    for mc in df_actuals['MC Group'].unique():
        data_temp_actuals=df_actuals[df_actuals['MC Group']==mc]
        data_temp_actuals=data_temp_actuals.sort_values('Clean_Start_Time')
        data_temp_actuals['Idle Time']=data_temp_actuals['Clean_Start_Time']-data_temp_actuals['Clean_End_Time'].shift(1)
        data_temp_actuals['Idle Time']=data_temp_actuals['Idle Time']/(np.timedelta64(1,'s')*3600)
        data_temp_actuals['Average_Cleaning_Duration'] =data_temp_actuals['Cleaning Duration'].mean()
        temp_df_actuals=pd.concat([temp_df_actuals,data_temp_actuals])
    df_actuals=temp_df_actuals.reset_index(drop=True)
    df_actuals = df_actuals.sort_values(by=['MC Group', 'Clean_Start_Time'])
    df_actuals= df_actuals.fillna(0)
    df_actuals['Idle Time'] = np.where(df_actuals['Idle Time']<0,0,df_actuals['Idle Time'])
    cleaning_time =[]
    for i in range(len(df_actuals['Idle Time'])):
        c= df_actuals['Idle Time'][i]/df_actuals['Average_Cleaning_Duration'][i]
        cleaning_time.append(int(c))
    df_actuals['Cleaning_Possible_Time'] = cleaning_time 
    total_sum_actual = df_actuals.groupby(['MC Group'])['Cleaning_Possible_Time'].agg('sum')
    data_actual = pd.DataFrame(total_sum_actual, columns=['Cleaning_Possible_Time'] ).reset_index()
    metrics_cip_actuals['Cleaning Possible Time'] = data_actual['Cleaning_Possible_Time']
    
else:
    metrics_cip_actuals['Cleaning Possible Time']=0

#additional cleanings
additional_cleaning_schedule = round(metrics_cip_schedule['Cleaning Possible Time'].sum(),0)
additional_cleaning_actuals = round(metrics_cip_actuals['Cleaning Possible Time'].sum(),0)


#NaOH consumption
#total cleaning cycles
total_cip_cleanings_schedule =round(metrics_cip_schedule['No. of cleanings'].sum(),0)
total_cip_cleanings_actuals =round(metrics_cip_actuals['No. of cleanings'].sum(),0)
naoh_consumption_planned = round((total_cip_cleanings_schedule*0.016),2) 
naoh_consumption_actual = round((total_cip_cleanings_actuals*0.016),2)


cip_add_cleans_kpi= {
  "insight_data": [
    {
      "label": "Planned",
      "value": str(int(additional_cleaning_schedule))
    },
    {
      "label": "Actual",
      "severity": "success",
      "value": str(int(additional_cleaning_actuals))
    },
    { 
      "header":"NaOH Consumption", 
      "header_style": {"textDecoration": "none"},     
      "label": "Planned",
      "value": str(naoh_consumption_planned)+" kL"
    },
    {
      "label": "Actual",
      "severity": "success",
      "value": str(naoh_consumption_actual)+" kL"
    },
  ],
  "insight_label": "Additional Possible Cleanings"
}
dynamic_outputs=json.dumps(cip_add_cleans_kpi)
print(dynamic_outputs)
"""


# # CIP Utilization Visualization Code String #

# In[24]:


visual_cip_dur_overall="""
import pandas as pd
from pathlib import Path
from azure.storage.blob import BlockBlobService
from io import StringIO
import json
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly

data_source = 'azure_blob_storage'
connection_string = 'DefaultEndpointsProtocol=https;AccountName=mathcotakedastorage;AccountKey=s94v9Q0+SXPPRagBKJUfB2WOA2Ts5mFPUb4sUg+C0tRzagJel6eYeeyrA7Dlz7s2mmPorbtOJQohLgjefo++Vw==;EndpointSuffix=core.windows.net'

def get_ingested_data(file_path, datasource_type, connection_uri, container_name):
    path = Path(file_path)
    block_blob_service = BlockBlobService(connection_string=connection_uri)
    blob_data = block_blob_service.get_blob_to_text(container_name=container_name, 
                                                    blob_name=file_path)
    ingested_df = pd.read_csv(StringIO(blob_data.content))

    return ingested_df

 
data_source = 'azure_blob_storage'
connection_string = 'DefaultEndpointsProtocol=https;AccountName=mathcotakedastorage;AccountKey=s94v9Q0+SXPPRagBKJUfB2WOA2Ts5mFPUb4sUg+C0tRzagJel6eYeeyrA7Dlz7s2mmPorbtOJQohLgjefo++Vw==;EndpointSuffix=core.windows.net' 
def get_ingested_data(file_path, datasource_type, connection_uri, container_name):
    path = Path(file_path)
    block_blob_service = BlockBlobService(connection_string=connection_uri)
    blob_data = block_blob_service.get_blob_to_text(container_name=container_name, 
                                                    blob_name=file_path)
    ingested_df = pd.read_csv(StringIO(blob_data.content))
 
    return ingested_df

schedule=get_ingested_data(file_path='merged_output_repository/merged_outputs.csv',datasource_type=data_source,connection_uri=connection_string,container_name='input')
actuals=get_ingested_data(file_path='merged_output_repository/merged_actual_outputs.csv',datasource_type=data_source,connection_uri=connection_string,container_name='input')
def filter_data(selected_filter, df):
    cols_in_df = list(df.columns)
    # Filtering data
    if len(selected_filters) > 0:
        for key in selected_filters.keys():
            if type(selected_filters[key])==dict and selected_filters[key].get('start_date',False) and selected_filters[key].get('end_date',False):
                selected_filters[key]['start_date']=pd.to_datetime(selected_filters[key]['start_date'],infer_datetime_format=True,utc=True)
                selected_filters[key]['end_date']=pd.to_datetime(selected_filters[key]['end_date'],infer_datetime_format=True,utc=True)
                print(selected_filters)
                df=df[(df['Clean_Start_Time']>selected_filters[key]['start_date']) & (df['Clean_End_Time']<selected_filters[key]['end_date'])]
            elif key in cols_in_df:
                if type(selected_filter[key]) == str or type(selected_filter[key]) == int:
                    if selected_filter[key] == 'All':
                        continue
                    else:
                        df = df[df[key] == selected_filter[key]]
                else:
                    if isinstance(selected_filter[key],list) and 'All' in selected_filter[key]:
                        continue
                    else:
                        df = df[df[key].isin(selected_filter[key])]
            else:
                continue
    return df
schedule.Usage_Start= pd.to_datetime(schedule.Usage_Start, format = '%Y/%m/%d %H:%M:%S',utc=True)
schedule.Usage_End= pd.to_datetime(schedule.Usage_End, format = '%Y/%m/%d %H:%M:%S')
schedule['Next Usage']= pd.to_datetime(schedule['Next Usage'], format = '%Y/%m/%d %H:%M:%S')
schedule['End DHT Time']= pd.to_datetime(schedule['End DHT Time'], format = '%Y/%m/%d %H:%M:%S')
schedule['End CHT Time']= pd.to_datetime(schedule['End CHT Time'], format = '%Y/%m/%d %H:%M:%S')
schedule.Clean_Start_Time= pd.to_datetime(schedule.Clean_Start_Time, format = '%Y/%m/%d %H:%M:%S',utc=True)
schedule.Clean_End_Time= pd.to_datetime(schedule.Clean_End_Time, format = '%Y/%m/%d %H:%M:%S',utc=True)
actuals['Cleaning Duration'] = actuals['Cleaning Duration'] / 60
actuals.Clean_Start_Time= pd.to_datetime(actuals.Clean_Start_Time, format = '%Y/%m/%d %H:%M:%S',utc=True)
actuals.Clean_End_Time= pd.to_datetime(actuals.Clean_End_Time, format = '%Y/%m/%d %H:%M:%S',utc=True)
schedule=filter_data(selected_filters,schedule)
actuals=filter_data(selected_filters,actuals)
actuals = actuals[actuals['Cleaning_Type']=='Standard']
schedule['Cleaning Duration'] = schedule['Cleaning Duration'] / 60

if schedule['Clean_Start_Time'].min() > schedule['Usage_Start'].min():
        working_hours_schedule = (schedule['Clean_End_Time'].max()-schedule['Usage_Start'].min())/ (np.timedelta64(1, 'h'))
else:
        working_hours_schedule = (schedule['Clean_End_Time'].max()-schedule['Clean_Start_Time'].min())/ (np.timedelta64(1, 'h'))
actuals = actuals[['MC Group','Clean_Start_Time','Clean_End_Time','clean_flag','Cleaning Duration']]
working_hours_actuals = (actuals['Clean_End_Time'].max()-actuals['Clean_Start_Time'].min())/ (np.timedelta64(1, 'h'))
metrics_cip_schedule = schedule.groupby(['MC Group'], as_index = False).agg({
                                            
                                                'clean_flag': 'sum',
                                            'Cleaning Duration':'sum'
                                                })
metrics_cip_schedule = metrics_cip_schedule.rename(columns = {'Resource': 'Resources cleaned',
                        'clean_flag':'No. of cleanings','Cleaning Duration':'CIP used Duration',
                        })
metrics_cip_actuals = actuals.groupby(['MC Group'], as_index = False).agg({
                                            
                                                'clean_flag': 'sum',
                                            'Cleaning Duration':'sum'
                                                })
metrics_cip_actuals = metrics_cip_actuals.rename(columns = {'Resource': 'Resources cleaned',
                        'clean_flag':'No. of cleanings','Cleaning Duration':'CIP used Duration',
                        })
metrics_cip_schedule['Total Available Hours'] = working_hours_schedule
metrics_cip_actuals['Total Available Hours'] = working_hours_actuals
metrics_cip_schedule['Usage %'] = round((metrics_cip_schedule['CIP used Duration']/ metrics_cip_schedule['Total Available Hours'])*100,0)
metrics_cip_actuals['Usage %'] = round((metrics_cip_actuals['CIP used Duration']/ metrics_cip_actuals['Total Available Hours'])*100,0)
metrics_cip_schedule['CIP used Duration'] = metrics_cip_schedule['CIP used Duration'].round()
metrics_cip_actuals['CIP used Duration'] = metrics_cip_actuals['CIP used Duration'].round()
fig_duration = go.Figure(data=[go.Bar(name='Planned',x=metrics_cip_schedule['MC Group'],y =metrics_cip_schedule['CIP used Duration'], text =metrics_cip_schedule['CIP used Duration'], textposition= 'outside'),
                         go.Bar(name ='Actual',x=metrics_cip_actuals['MC Group'],y=metrics_cip_actuals['CIP used Duration'],text =metrics_cip_actuals['CIP used Duration'], textposition='outside')])
fig_duration.update_layout(barmode= 'group')
fig_duration.update_traces(texttemplate='%{text:0f}',textposition='outside',width=0.3,cliponaxis = False)
fig_duration.update_layout(title_text='CIP Usage Duration')
fig_duration.update_layout(xaxis_title='Machine',
                  yaxis_title='Duration (Hours)')
fig_duration.show()
average_cip_graph=json.loads(plotly.io.to_json(fig_duration))
dynamic_outputs=json.dumps(average_cip_graph)
print(dynamic_outputs)
"""


# In[25]:


visual_cip_usage_dist="""
import warnings
import pandas as pd
import numpy as np
import datetime as dt
import warnings
warnings.filterwarnings("ignore")
from pathlib import Path
from azure.storage.blob import BlockBlobService
from io import StringIO
import json
import plotly
from plotly.subplots import make_subplots
import plotly.graph_objects as go

data_source = 'azure_blob_storage'
connection_string = 'DefaultEndpointsProtocol=https;AccountName=mathcotakedastorage;AccountKey=s94v9Q0+SXPPRagBKJUfB2WOA2Ts5mFPUb4sUg+C0tRzagJel6eYeeyrA7Dlz7s2mmPorbtOJQohLgjefo++Vw==;EndpointSuffix=core.windows.net' 
def get_ingested_data(file_path, datasource_type, connection_uri, container_name):
    path = Path(file_path)
    block_blob_service = BlockBlobService(connection_string=connection_uri)
    blob_data = block_blob_service.get_blob_to_text(container_name=container_name, 
                                                    blob_name=file_path)
    ingested_df = pd.read_csv(StringIO(blob_data.content))
 
    return ingested_df
schedule=get_ingested_data(file_path='merged_output_repository/merged_outputs.csv',datasource_type=data_source,connection_uri=connection_string,container_name='input')
actuals=get_ingested_data(file_path='merged_output_repository/merged_actual_outputs.csv',datasource_type=data_source,connection_uri=connection_string,container_name='input')
schedule['Cleaning Duration'] = schedule['Cleaning Duration'] / 60
schedule.Usage_Start= pd.to_datetime(schedule.Usage_Start, format = '%Y/%m/%d %H:%M:%S',utc=True)
schedule.Usage_End= pd.to_datetime(schedule.Usage_End, format = '%Y/%m/%d %H:%M:%S',utc=True)
schedule['Next Usage']= pd.to_datetime(schedule['Next Usage'], format = '%Y/%m/%d %H:%M:%S',utc=True)
schedule['End DHT Time']= pd.to_datetime(schedule['End DHT Time'], format = '%Y/%m/%d %H:%M:%S',utc=True)
schedule['End CHT Time']= pd.to_datetime(schedule['End CHT Time'], format = '%Y/%m/%d %H:%M:%S',utc=True)
schedule.Clean_Start_Time= pd.to_datetime(schedule.Clean_Start_Time, format = '%Y/%m/%d %H:%M:%S',utc=True)
schedule.Clean_End_Time= pd.to_datetime(schedule.Clean_End_Time, format = '%Y/%m/%d %H:%M:%S',utc=True)
actuals['Cleaning Duration'] = actuals['Cleaning Duration'] / 60
actuals.Clean_Start_Time= pd.to_datetime(actuals.Clean_Start_Time, format = '%Y/%m/%d %H:%M:%S',utc=True)
actuals.Clean_End_Time= pd.to_datetime(actuals.Clean_End_Time, format = '%Y/%m/%d %H:%M:%S',utc=True)
def filter_data(selected_filter, df):
    cols_in_df = list(df.columns)
    # Filtering data
    if len(selected_filters) > 0:
        for key in selected_filters.keys():
            if type(selected_filters[key])==dict and selected_filters[key].get('start_date',False) and selected_filters[key].get('end_date',False):
                selected_filters[key]['start_date']=pd.to_datetime(selected_filters[key]['start_date'],infer_datetime_format=True,utc=True)
                selected_filters[key]['end_date']=pd.to_datetime(selected_filters[key]['end_date'],infer_datetime_format=True,utc=True)
                print(selected_filters)
                df=df[(df['Clean_Start_Time']>selected_filters[key]['start_date']) & (df['Clean_End_Time']<selected_filters[key]['end_date'])]
            elif key in cols_in_df:
                if type(selected_filter[key]) == str or type(selected_filter[key]) == int:
                    if selected_filter[key] == 'All':
                        continue
                    else:
                        df = df[df[key] == selected_filter[key]]
                else:
                    if isinstance(selected_filter[key],list) and 'All' in selected_filter[key]:
                        continue
                    else:
                        df = df[df[key].isin(selected_filter[key])]
            else:
                continue
    return df
schedule=filter_data(selected_filters,schedule)
actuals=filter_data(selected_filters,actuals)
actuals_full = actuals.copy()
actuals = actuals[actuals['Cleaning_Type']=='Standard']
if schedule['Clean_Start_Time'].min() > schedule['Usage_Start'].min():
        working_hours_schedule = (schedule['Clean_End_Time'].max()-schedule['Usage_Start'].min())/ (np.timedelta64(1, 'h'))
else:
        working_hours_schedule = (schedule['Clean_End_Time'].max()-schedule['Clean_Start_Time'].min())/ (np.timedelta64(1, 'h'))
actuals = actuals[['MC Group','Clean_Start_Time','Clean_End_Time','clean_flag','Cleaning Duration']]
working_hours_actuals = (actuals['Clean_End_Time'].max()-actuals['Clean_Start_Time'].min())/ (np.timedelta64(1, 'h'))
metrics_cip_schedule = schedule.groupby(['MC Group'], as_index = False).agg({
                                            
                                                'clean_flag': 'sum',
                                            'Cleaning Duration':'sum'
                                                })
metrics_cip_schedule = metrics_cip_schedule.rename(columns = {'Resource': 'Resources cleaned',
                        'clean_flag':'No. of cleanings','Cleaning Duration':'CIP used Duration',
                        })
metrics_cip_actuals = actuals.groupby(['MC Group'], as_index = False).agg({
                                            
                                                'clean_flag': 'sum',
                                            'Cleaning Duration':'sum'
                                                })
metrics_cip_actuals = metrics_cip_actuals.rename(columns = {'Resource': 'Resources cleaned',
                        'clean_flag':'No. of cleanings','Cleaning Duration':'CIP used Duration',
                        })
metrics_cip_schedule['Total Available Hours'] = working_hours_schedule
metrics_cip_actuals['Total Available Hours'] = working_hours_actuals
metrics_cip_schedule['Usage %'] = round((metrics_cip_schedule['CIP used Duration']/ metrics_cip_schedule['Total Available Hours'])*100,0)
metrics_cip_actuals['Usage %'] = round((metrics_cip_actuals['CIP used Duration']/ metrics_cip_actuals['Total Available Hours'])*100,0)
non_missed = schedule[schedule['MC Group'].isin(['MC1','MC2','MC3','MC4'])]
df2 = non_missed[['Resource','MC Group','Clean_Start_Time','Clean_End_Time','Cleaning Duration','Constraint']]
df2 =df2.sort_values(by=['MC Group', 'Clean_Start_Time'])
df2.drop_duplicates(subset=['Clean_Start_Time','Clean_End_Time','MC Group','Constraint'],keep = 'first', inplace = True)
temp_df=pd.DataFrame()
for mc in df2['MC Group'].unique():
    data_temp=df2[df2['MC Group']==mc]
    data_temp=data_temp.sort_values('Clean_Start_Time')
    data_temp['Idle Time']=data_temp['Clean_Start_Time']-data_temp['Clean_End_Time'].shift(1)
    data_temp['Idle Time']=data_temp['Idle Time']/(np.timedelta64(1,'s')*3600)
    data_temp['Average_Cleaning_Duration'] =data_temp['Cleaning Duration'].mean()
    temp_df=pd.concat([temp_df,data_temp])
df2=temp_df.reset_index(drop=True)
if len(df2)!=0:
    df2 = df2.sort_values(by=['MC Group', 'Clean_Start_Time'])
    df2= df2.fillna(0)
    cleaning_time =[]
    for i in range(len(df2['Idle Time'])):
        c= df2['Idle Time'][i]/df2['Average_Cleaning_Duration'][i]
        cleaning_time.append(int(c))
    df2['Cleaning_Possible_Time'] = cleaning_time 
    total_sum = df2.groupby(['MC Group'])['Cleaning_Possible_Time'].agg('sum')
    df3 = pd.DataFrame(total_sum, columns=['Cleaning_Possible_Time'] ).reset_index()
    metrics_cip_schedule['Cleaning Possible Time'] = df3['Cleaning_Possible_Time']
else:
    metrics_cip_schedule['CIP used Duration']=0

non_missed_actuals = actuals[actuals['MC Group'].isin(['MC1','MC2','MC3','MC4'])]
df_actuals = non_missed_actuals[['MC Group','Clean_Start_Time','Clean_End_Time','Cleaning Duration', 'clean_flag']]
df_actuals =df_actuals.sort_values(by=['MC Group', 'Clean_Start_Time'])
df_actuals.drop_duplicates(subset=['Clean_Start_Time','Clean_End_Time','MC Group'],keep = 'first', inplace = True)
temp_df_actuals=pd.DataFrame()
for mc in df_actuals['MC Group'].unique():
    data_temp_actuals=df_actuals[df_actuals['MC Group']==mc]
    data_temp_actuals=data_temp_actuals.sort_values('Clean_Start_Time')
    data_temp_actuals['Idle Time']=data_temp_actuals['Clean_Start_Time']-data_temp_actuals['Clean_End_Time'].shift(1)
    data_temp_actuals['Idle Time']=data_temp_actuals['Idle Time']/(np.timedelta64(1,'s')*3600)
    data_temp_actuals['Average_Cleaning_Duration'] =data_temp_actuals['Cleaning Duration'].mean()
    temp_df_actuals=pd.concat([temp_df_actuals,data_temp_actuals])
df_actuals=temp_df_actuals.reset_index(drop=True)
if len(df_actuals)!=0:
    df_actuals = df_actuals.sort_values(by=['MC Group', 'Clean_Start_Time'])
    df_actuals= df_actuals.fillna(0)
    cleaning_time =[]
    for i in range(len(df_actuals['Idle Time'])):
        c= df_actuals['Idle Time'][i]/df_actuals['Average_Cleaning_Duration'][i]
        cleaning_time.append(int(c))
    df_actuals['Cleaning_Possible_Time'] = cleaning_time 
    total_sum_actual = df_actuals.groupby(['MC Group'])['Cleaning_Possible_Time'].agg('sum')
    data_actual = pd.DataFrame(total_sum_actual, columns=['Cleaning_Possible_Time'] ).reset_index()
    metrics_cip_actuals['Cleaning Possible Time'] = data_actual['Cleaning_Possible_Time']
else:
    metrics_cip_schedule['CIP used Duration']=0

#average CIP utilization 
cip_util_schedule = round((metrics_cip_schedule['CIP used Duration'].sum() /  metrics_cip_schedule['Total Available Hours'].sum())*100,0)
cip_util_actuals = round((metrics_cip_actuals['CIP used Duration'].sum() /  metrics_cip_actuals['Total Available Hours'].sum())*100,0)
labels = ['MC1','MC2','MC3','MC4']
fig=go.Figure()
# Create subplots: use 'domain' type for Pie subplot
if len(actuals)!=0 and len(schedule)!=0:
    fig = make_subplots(rows=1, cols=2, specs=[[{'type':'domain'}, {'type':'domain'}]])
    fig.add_trace(go.Pie(labels=metrics_cip_schedule['MC Group'].tolist(), values=metrics_cip_schedule['No. of cleanings'].tolist(), name="Planned",sort=False),
                1, 1)
    fig.add_trace(go.Pie(labels=metrics_cip_actuals['MC Group'].tolist(), values=metrics_cip_actuals['No. of cleanings'], name="Actual",sort=False),
                1, 2)
    # Use `hole` to create a donut-like pie chart
    fig.update_traces(hole=.4, hoverinfo="label+percent+name",textfont=dict(color="white"),texttemplate="%{percent:0%f}")
    fig.update_layout(
        title_text="Planned V/S Actual ",
        # Add annotations in the center of the donut pies.
        annotations=[dict(text='PLANNED', x=0.165, y=0.5, font_size=26, showarrow=False),
                    dict(text='ACTUAL', x=0.837, y=0.5, font_size=26, showarrow=False)])
elif len(actuals)==0:
    fig = make_subplots(rows=1, cols=2, specs=[[{'type':'domain'}, {'type':'domain'}]])
    fig.add_trace(go.Pie(labels=metrics_cip_schedule['MC Group'].tolist(), values=metrics_cip_schedule['No. of cleanings'].tolist(), name="Planned",sort=False),
                1, 1)
    # Use `hole` to create a donut-like pie chart
    fig.update_traces(hole=.4, hoverinfo="label+percent+name",textfont=dict(color="white"),texttemplate="%{percent:0%f}")
    fig.update_layout(
        title_text="Planned V/S Actual ",
        # Add annotations in the center of the donut pies.
        annotations=[dict(text='PLANNED', x=0.165, y=0.5, font_size=26, showarrow=False)])
elif len(schedule)==0:
    fig = make_subplots(rows=1, cols=2, specs=[[{'type':'domain'}, {'type':'domain'}]])
    fig.add_trace(go.Pie(labels=metrics_cip_actuals['MC Group'].tolist(), values=metrics_cip_actuals['No. of cleanings'], name="Actual",sort=False),
                1, 1)
    # Use `hole` to create a donut-like pie chart
    fig.update_traces(hole=.4, hoverinfo="label+percent+name",textfont=dict(color="white"),texttemplate="%{percent:0%f}")
    fig.update_layout(
        title_text="Planned V/S Actual ",
        # Add annotations in the center of the donut pies.
        annotations=[dict(text='ACTUAL', x=0.165, y=0.5, font_size=26, showarrow=False)])
    

if len(actuals_full)!=0:
    sd= actuals_full[actuals_full['Cleaning_Type']=='Standard'].shape[0]
    nsd = actuals_full[actuals_full['Cleaning_Type']!='Standard'].shape[0]
    labels = ['Non-Standard','Standard']
    fig_sd = go.Figure(data = go.Pie(labels=labels, values=[nsd,sd],sort=False ))
    fig_sd.update_traces(hole=.4, hoverinfo="label+percent",textfont=dict(color="white"),texttemplate="%{percent:0%f}")
    fig.show()
else:
    fig_sd={}


fig_month=json.loads(plotly.io.to_json(fig))
fig_week=json.loads(plotly.io.to_json(fig_sd))

util_flip= {
  "front":{
    "data":{
      "value":fig_month
      }
  },
  "back":{
    "data":{
      "value":fig_week
    }
  },
  "is_flip":True
}

dynamic_outputs=json.dumps(util_flip)

"""


# In[26]:


visual_arrival_time_cip="""
import pandas as pd
from pathlib import Path
from azure.storage.blob import BlockBlobService
from io import StringIO
import json
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly

data_source = 'azure_blob_storage'
connection_string = 'DefaultEndpointsProtocol=https;AccountName=mathcotakedastorage;AccountKey=s94v9Q0+SXPPRagBKJUfB2WOA2Ts5mFPUb4sUg+C0tRzagJel6eYeeyrA7Dlz7s2mmPorbtOJQohLgjefo++Vw==;EndpointSuffix=core.windows.net'

def get_ingested_data(file_path, datasource_type, connection_uri, container_name):
    path = Path(file_path)
    block_blob_service = BlockBlobService(connection_string=connection_uri)
    blob_data = block_blob_service.get_blob_to_text(container_name=container_name, 
                                                    blob_name=file_path)
    ingested_df = pd.read_csv(StringIO(blob_data.content))

    return ingested_df

 
data_source = 'azure_blob_storage'
connection_string = 'DefaultEndpointsProtocol=https;AccountName=mathcotakedastorage;AccountKey=s94v9Q0+SXPPRagBKJUfB2WOA2Ts5mFPUb4sUg+C0tRzagJel6eYeeyrA7Dlz7s2mmPorbtOJQohLgjefo++Vw==;EndpointSuffix=core.windows.net' 
def get_ingested_data(file_path, datasource_type, connection_uri, container_name):
    path = Path(file_path)
    block_blob_service = BlockBlobService(connection_string=connection_uri)
    blob_data = block_blob_service.get_blob_to_text(container_name=container_name, 
                                                    blob_name=file_path)
    ingested_df = pd.read_csv(StringIO(blob_data.content))
 
    return ingested_df

schedule=get_ingested_data(file_path='merged_output_repository/merged_outputs.csv',datasource_type=data_source,connection_uri=connection_string,container_name='input')
actuals=get_ingested_data(file_path='merged_output_repository/merged_actual_outputs.csv',datasource_type=data_source,connection_uri=connection_string,container_name='input')
def filter_data(selected_filter, df):
    cols_in_df = list(df.columns)
    # Filtering data
    if len(selected_filters) > 0:
        for key in selected_filters.keys():
            if type(selected_filters[key])==dict and selected_filters[key].get('start_date',False) and selected_filters[key].get('end_date',False):
                selected_filters[key]['start_date']=pd.to_datetime(selected_filters[key]['start_date'],infer_datetime_format=True,utc=True)
                selected_filters[key]['end_date']=pd.to_datetime(selected_filters[key]['end_date'],infer_datetime_format=True,utc=True)
                print(selected_filters)
                df=df[(df['Clean_Start_Time']>selected_filters[key]['start_date']) & (df['Clean_End_Time']<selected_filters[key]['end_date'])]
            elif key in cols_in_df:
                if type(selected_filter[key]) == str or type(selected_filter[key]) == int:
                    if selected_filter[key] == 'All':
                        continue
                    else:
                        df = df[df[key] == selected_filter[key]]
                else:
                    if isinstance(selected_filter[key],list) and 'All' in selected_filter[key]:
                        continue
                    else:
                        df = df[df[key].isin(selected_filter[key])]
            else:
                continue
    return df
schedule.Usage_Start= pd.to_datetime(schedule.Usage_Start, format = '%Y/%m/%d %H:%M:%S',utc=True)
schedule.Usage_End= pd.to_datetime(schedule.Usage_End, format = '%Y/%m/%d %H:%M:%S')
schedule['Next Usage']= pd.to_datetime(schedule['Next Usage'], format = '%Y/%m/%d %H:%M:%S')
schedule['End DHT Time']= pd.to_datetime(schedule['End DHT Time'], format = '%Y/%m/%d %H:%M:%S')
schedule['End CHT Time']= pd.to_datetime(schedule['End CHT Time'], format = '%Y/%m/%d %H:%M:%S')
schedule.Clean_Start_Time= pd.to_datetime(schedule.Clean_Start_Time, format = '%Y/%m/%d %H:%M:%S',utc=True)
schedule.Clean_End_Time= pd.to_datetime(schedule.Clean_End_Time, format = '%Y/%m/%d %H:%M:%S',utc=True)
actuals['Cleaning Duration'] = actuals['Cleaning Duration'] / 60
actuals.Clean_Start_Time= pd.to_datetime(actuals.Clean_Start_Time, format = '%Y/%m/%d %H:%M:%S',utc=True)
actuals.Clean_End_Time= pd.to_datetime(actuals.Clean_End_Time, format = '%Y/%m/%d %H:%M:%S',utc=True)
schedule=filter_data(selected_filters,schedule)
actuals=filter_data(selected_filters,actuals)
actuals = actuals[actuals['Cleaning_Type']=='Standard']
schedule['Cleaning Duration'] = schedule['Cleaning Duration'] / 60

if schedule['Clean_Start_Time'].min() > schedule['Usage_Start'].min():
        working_hours_schedule = (schedule['Clean_End_Time'].max()-schedule['Usage_Start'].min())/ (np.timedelta64(1, 'h'))
else:
        working_hours_schedule = (schedule['Clean_End_Time'].max()-schedule['Clean_Start_Time'].min())/ (np.timedelta64(1, 'h'))
actuals = actuals[['MC Group','Clean_Start_Time','Clean_End_Time','clean_flag','Cleaning Duration']]
working_hours_actuals = (actuals['Clean_End_Time'].max()-actuals['Clean_Start_Time'].min())/ (np.timedelta64(1, 'h'))

def average_arrival_mc(item):
    df=pd.DataFrame()
    for i in item['MC Group'].unique():
        temp_data=item[item['MC Group']==i]
        temp_data.sort_values('Clean_Start_Time',inplace=True)
        temp_data['Idle_Time']=temp_data['Clean_Start_Time']-temp_data['Clean_End_Time'].shift(1)
        temp_data['Idle_Time']=temp_data['Idle_Time']/(np.timedelta64(1, 's')*3600)
        temp_data=temp_data[temp_data['Idle_Time']>0]
        inter_arrival_time=temp_data['Idle_Time'].mean()
        df=df.append({'MC Group':i,'Inter_Arrival_Time_Actual':inter_arrival_time},ignore_index=True)
        df['Inter_Arrival_Time_Actual']=df['Inter_Arrival_Time_Actual'].round(1)
    return df

df_actuals=average_arrival_mc(actuals)
df_schedule= average_arrival_mc(schedule)
import plotly.graph_objects as go

if len(df_actuals)!=0 and len(df_schedule)!=0:
    fig_arrival = go.Figure(data=[go.Bar(name= 'Planned',x=df_schedule['Inter_Arrival_Time_Actual'],y=df_schedule['MC Group'],text=df_schedule['Inter_Arrival_Time_Actual'], orientation='h'),
                                go.Bar(name='Actual',x=df_actuals['Inter_Arrival_Time_Actual'],y=df_actuals['MC Group'],text=df_actuals['Inter_Arrival_Time_Actual'], orientation='h')])
    fig_arrival.update_xaxes(categoryorder="total descending")
    fig_arrival.update_layout(title_text='CIP Usage Frequency')
    fig_arrival.update_traces(texttemplate='%{text:.2f}',textposition='outside',width=0.3,cliponaxis = False)
    fig_arrival.update_layout(xaxis_title='Average Time Gap (Hours)',
                    yaxis_title='Machine')
    fig_arrival.show()
elif len(df_actuals)==0 and len(df_schedule)!=0:
    fig_arrival = go.Figure(data=[go.Bar(name= 'Planned',x=df_schedule['Inter_Arrival_Time_Actual'],y=df_schedule['MC Group'],text=df_schedule['Inter_Arrival_Time_Actual'], orientation='h')])
    fig_arrival.update_xaxes(categoryorder="total descending")
    fig_arrival.update_layout(title_text='CIP Usage Frequency')
    fig_arrival.update_traces(texttemplate='%{text:.2f}',textposition='outside',width=0.3,cliponaxis = False)
    fig_arrival.update_layout(xaxis_title='Average Time Gap (Hours)',
                    yaxis_title='Machine')
    fig_arrival.show()
elif len(df_actuals)!=0 and len(df_schedule)==0:
    fig_arrival = go.Figure(data=[go.Bar(name='Actual',x=df_actuals['Inter_Arrival_Time_Actual'],y=df_actuals['MC Group'],text=df_actuals['Inter_Arrival_Time_Actual'], orientation='h')])
    fig_arrival.update_xaxes(categoryorder="total descending")
    fig_arrival.update_layout(title_text='CIP Usage Frequency')
    fig_arrival.update_traces(texttemplate='%{text:.2f}',textposition='outside',width=0.3,cliponaxis = False)
    fig_arrival.update_layout(xaxis_title='Average Time Gap (Hours)',
                    yaxis_title='Machine')
    fig_arrival.show()

arrival_dist=json.loads(plotly.io.to_json(fig_arrival))
dynamic_outputs=json.dumps(arrival_dist)
print(dynamic_outputs)
"""


# In[27]:


visual_periodic_util="""
import pandas as pd
from pathlib import Path
from azure.storage.blob import BlockBlobService
from io import StringIO
import json
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly
import datetime


data_source = 'azure_blob_storage'
connection_string = 'DefaultEndpointsProtocol=https;AccountName=mathcotakedastorage;AccountKey=s94v9Q0+SXPPRagBKJUfB2WOA2Ts5mFPUb4sUg+C0tRzagJel6eYeeyrA7Dlz7s2mmPorbtOJQohLgjefo++Vw==;EndpointSuffix=core.windows.net'

def get_ingested_data(file_path, datasource_type, connection_uri, container_name):
    path = Path(file_path)
    block_blob_service = BlockBlobService(connection_string=connection_uri)
    blob_data = block_blob_service.get_blob_to_text(container_name=container_name,
                                                    blob_name=file_path)
    ingested_df = pd.read_csv(StringIO(blob_data.content))

    return ingested_df
dataset=get_ingested_data(file_path='merged_output_repository/merged_outputs.csv',datasource_type=data_source,connection_uri=connection_string,container_name='input')
dataset['Clean_Start_Time']=pd.to_datetime(dataset['Clean_Start_Time'],format='%Y/%m/%d %H:%M:%S')
dataset['Clean_End_Time']=pd.to_datetime(dataset['Clean_End_Time'],format='%Y/%m/%d %H:%M:%S')
dataset['Clean_Start_Time']= pd.to_datetime(dataset['Clean_Start_Time'],infer_datetime_format=True,utc=True)
dataset['Clean_End_Time']= pd.to_datetime(dataset['Clean_End_Time'],infer_datetime_format=True,utc=True)
actual_dataset=get_ingested_data(file_path='merged_output_repository/merged_actual_outputs.csv',datasource_type=data_source,connection_uri=connection_string,container_name='input')
actual_dataset['Clean_Start_Time']=pd.to_datetime(actual_dataset['Clean_Start_Time'],format='%Y/%m/%d %H:%M:%S')
actual_dataset['Clean_End_Time']=pd.to_datetime(actual_dataset['Clean_End_Time'],format='%Y/%m/%d %H:%M:%S')
actual_dataset['Clean_Start_Time']= pd.to_datetime(actual_dataset['Clean_Start_Time'],infer_datetime_format=True,utc=True)
actual_dataset['Clean_End_Time']= pd.to_datetime(actual_dataset['Clean_End_Time'],infer_datetime_format=True,utc=True)
def filter_data(selected_filter, df):
    cols_in_df = list(df.columns)
    # Filtering data
    if len(selected_filters) > 0:
        for key in selected_filters.keys():
            if type(selected_filters[key])==dict and selected_filters[key].get('start_date',False) and selected_filters[key].get('end_date',False):
                selected_filters[key]['start_date']=pd.to_datetime(selected_filters[key]['start_date'],infer_datetime_format=True,utc=True)
                selected_filters[key]['end_date']=pd.to_datetime(selected_filters[key]['end_date'],infer_datetime_format=True,utc=True)
                print(selected_filters)
                df=df[(df['Clean_Start_Time']>selected_filters[key]['start_date']) & (df['Clean_End_Time']<selected_filters[key]['end_date'])]
            elif key in cols_in_df:
                if type(selected_filter[key]) == str or type(selected_filter[key]) == int:
                    if selected_filter[key] == 'All':
                        continue
                    else:
                        df = df[df[key] == selected_filter[key]]
                else:
                    if isinstance(selected_filter[key],list) and 'All' in selected_filter[key]:
                        continue
                    else:
                        df = df[df[key].isin(selected_filter[key])]
            else:
                continue
    return df

dataset=filter_data(selected_filters,dataset)
actual_dataset=filter_data(selected_filters,actual_dataset)
dataset = dataset.sort_values(by = "Clean_Start_Time")
dataset = dataset.reset_index(drop = True)
if len(dataset)!=0:
    df_week = pd.date_range(start=dataset['Clean_Start_Time'].min(),end=dataset['Clean_Start_Time'].max(), freq = 'W').to_pydatetime().tolist()
    df_week = pd.DataFrame(df_week, columns =['week'])
    if len(df_week)==0:
        x=dataset['Clean_End_Time'].max()
        df_week=df_week.append({'week':x},ignore_index=True)
    a=df_week['week'].max()+datetime.timedelta(days=6)
    df_week=df_week.append({'week':a},ignore_index=True)
    rows = []
    dates=[]
    for i in range(0, len(df_week['week'])):
        df_week_4 = df_week.copy()
        df_week_4['week'] = pd.to_datetime(df_week_4['week'],utc=True).dt.date    
        date_str = str(df_week_4['week'][i])    
        date_obj = datetime.datetime.strptime(date_str,'%Y-%m-%d')
        start_week = date_obj.date()
        start_week_new = datetime.datetime.strftime(start_week, '%d-%m-%Y')
        end_week= start_week + datetime.timedelta(days=6)
        end_week_new = datetime.datetime.strftime(end_week, '%d-%m-%Y')
        rows.append([end_week_new])
        for date in (dataset['Clean_Start_Time']):
            if ((date<=end_week) and (date>=start_week)):
                dates.append(rows[i])
    date_data = pd.DataFrame(dates, columns=['week_number'])
    dataset['week_classifier'] = date_data['week_number']
    dataset['week_classifier']=pd.to_datetime(dataset['week_classifier'],format='%d-%m-%Y',utc=True).dt.date
    dataset['Month']=pd.DatetimeIndex(dataset['Clean_Start_Time']).strftime("%b-%Y")
    dataset=dataset.dropna(subset=['MC Group'])
    data=pd.DataFrame()
    a=dataset['week_classifier'].unique().tolist()
    for i in range(0,len(a)):
        df=dataset.copy()
        df=df[df['week_classifier']==a[i]]
        working_hours = (df['Clean_End_Time'].max()-df['Clean_Start_Time'].min())/ (np.timedelta64(1, 'h'))
        df['working_hours']=working_hours    
        data=pd.concat([data,df])
    dataset=data.copy()
    dataset['Cleaning Duration']=dataset['Cleaning Duration']/60
    dataset.drop_duplicates(subset =["Parallel Clean Flag",'MC Group','Constraint','Clean_Start_Time','Clean_End_Time'],keep='last',inplace=True)
    metrics_cip = dataset.groupby(['MC Group','Constraint','week_classifier'], as_index = False).agg({
                                                    'clean_flag': 'sum',
                                                'Cleaning Duration':'sum',
                                                'working_hours':'mean'                                                })
    metrics_cip = metrics_cip.rename(columns = {'clean_flag':'No. of cleanings','Cleaning Duration':'CIP used Duration','working_hours':'Available Hours'})
    metrics_cip['CIP idle time'] = (metrics_cip['Available Hours'] - metrics_cip['CIP used Duration']).round(1)
    metrics_cip['CIP idle time%'] = ((metrics_cip['CIP idle time']/(metrics_cip['CIP used Duration']+metrics_cip['CIP idle time']))*100).round(1)
    metrics_cip['CIP Usage%'] = (((metrics_cip['CIP used Duration'])/(metrics_cip['CIP used Duration']+metrics_cip['CIP idle time']))*100).round(1)
    metrics_cip_total = metrics_cip.groupby(['MC Group','week_classifier'], as_index = False).agg({
                                                    'No. of cleanings': 'sum', #No of cleanings                                            
                                                    'CIP used Duration':'sum',#CIP used duration                                                
                                                    'Available Hours':'max'#CIP setup duration                                            
                                                    })
    metrics_cip_total['CIP idle time'] = (metrics_cip_total['Available Hours'] -metrics_cip_total['CIP used Duration']).round(1)
    metrics_cip_total['No. of possible cleanings'] = (metrics_cip_total['CIP idle time']/(metrics_cip_total['CIP used Duration']/metrics_cip_total['No. of cleanings'])).apply(np.floor)
    metrics_cip_total['CIP idle time%'] = ((metrics_cip_total['CIP idle time']/(metrics_cip_total['CIP used Duration']+metrics_cip_total['CIP idle time']))*100).round(1)
    metrics_cip_total['CIP Usage%'] = (((metrics_cip_total['CIP used Duration'])/(metrics_cip_total['CIP used Duration']+metrics_cip_total['CIP idle time']))*100).round(1)
    x1=metrics_cip_total
    x1['week_classifier']=pd.to_datetime(x1['week_classifier'],dayfirst=True,infer_datetime_format=True)
    x1=x1.sort_values('week_classifier')
    x1['week_classifier']=x1['week_classifier'].dt.strftime('%d-%b-%Y')
    px1 = x1
    # # # #FOR MONTH
    m=dataset['Month'].unique().tolist()
    data_month=pd.DataFrame()
    for i in range(0,len(m)):
        df=dataset.copy()
        df=df[df['Month']==m[i]]
        working_hours = (df['Clean_End_Time'].max()-df['Clean_Start_Time'].min())/ (np.timedelta64(1, 'h'))
        df['working_hours_month']=working_hours    
        data_month=pd.concat([data_month,df])
    dataset=data_month.copy()
    metrics_cip = dataset.groupby(['MC Group','Constraint','Month'], as_index = False).agg({
                                                    'clean_flag': 'sum',
                                                'Cleaning Duration':'sum',
                                                'working_hours_month':'mean'                                               
                                                })
    metrics_cip = metrics_cip.rename(columns = {
                            'clean_flag':'No. of cleanings',
                            'Cleaning Duration':'CIP used Duration',
                            'working_hours_month':'Available Hours' })
    metrics_cip['CIP idle time'] = (metrics_cip['Available Hours'] - metrics_cip['CIP used Duration']).round(1)
    metrics_cip['CIP idle time%'] = ((metrics_cip['CIP idle time']/(metrics_cip['CIP used Duration']+metrics_cip['CIP idle time']))*100).round(1)
    metrics_cip['CIP Usage%'] = (((metrics_cip['CIP used Duration'])/(metrics_cip['CIP used Duration']+metrics_cip['CIP idle time']))*100).round(1)
    metrics_cip_total = metrics_cip.groupby(['MC Group','Month'], as_index = False).agg({
                                                    'No. of cleanings': 'sum', #No of cleanings                                           
                                                    'CIP used Duration':'sum',#CIP used duration                                                
                                                    'Available Hours':'max'#CIP setup duration                                            
                                                    })
    metrics_cip_total['CIP idle time'] = (metrics_cip_total['Available Hours'] -metrics_cip_total['CIP used Duration']).round(1)
    metrics_cip_total['No. of possible cleanings'] = (metrics_cip_total['CIP idle time']/(metrics_cip_total['CIP used Duration']/metrics_cip_total['No. of cleanings'])).apply(np.floor)
    metrics_cip_total['CIP idle time%'] = ((metrics_cip_total['CIP idle time']/(metrics_cip_total['CIP used Duration']+metrics_cip_total['CIP idle time']))*100).round(1)
    metrics_cip_total['CIP Usage%'] = (((metrics_cip_total['CIP used Duration'])/(metrics_cip_total['CIP used Duration']+metrics_cip_total['CIP idle time']))*100).round(1)
    x2=metrics_cip_total
    x2['Month']=x2['Month'].map(lambda date_string: datetime.datetime.strptime(date_string,"%b-%Y"))
    x2=x2.sort_values('Month')
    x2['Month']=x2['Month'].dt.strftime('%b-%Y')
    px2=x2
    px1['Schedule_type'] = 'Planned'
    px2['Schedule_type'] = 'Planned'
else:
    px1=pd.DataFrame()
    px2=pd.DataFrame()
#Actual
actual_dataset = actual_dataset.sort_values(by = "Clean_Start_Time")
actual_dataset = actual_dataset.reset_index(drop = True)
if len(actual_dataset)!=0:
    df_week = pd.date_range(start=actual_dataset['Clean_Start_Time'].min(),end=actual_dataset['Clean_Start_Time'].max(), freq = 'W').to_pydatetime().tolist()
    df_week = pd.DataFrame(df_week, columns =['week'])
    if len(df_week)==0:
        x=actual_dataset['Clean_End_Time'].max()
        df_week=df_week.append({'week':x},ignore_index=True)
    a=df_week['week'].max()+datetime.timedelta(days=6)
    df_week=df_week.append({'week':a},ignore_index=True)
    rows = []
    dates=[]
    for i in range(0, len(df_week['week'])):
        df_week_4 = df_week.copy()
        df_week_4['week'] = pd.to_datetime(df_week_4['week'],utc=True).dt.date    
        date_str = str(df_week_4['week'][i])
        #date_str_new = date_str[9:]+"-"+date_str[6:7]+"-"+date_str[0:4]    
        date_obj = datetime.datetime.strptime(date_str,'%Y-%m-%d')
        start_week = date_obj.date()
        start_week_new = datetime.datetime.strftime(start_week, '%d-%m-%Y')
        end_week= start_week + datetime.timedelta(days=6)
        end_week_new = datetime.datetime.strftime(end_week, '%d-%m-%Y')
        rows.append([end_week_new])
        for date in (actual_dataset['Clean_Start_Time']):
            if ((date<=end_week) and (date>=start_week)):
                dates.append(rows[i])
    date_data = pd.DataFrame(dates, columns=['week_number'])
    actual_dataset['week_classifier'] = date_data['week_number']
    actual_dataset['week_classifier']=pd.to_datetime(actual_dataset['week_classifier'],format='%d-%m-%Y',utc=True).dt.date
    actual_dataset['Month']=pd.DatetimeIndex(actual_dataset['Clean_Start_Time']).strftime("%b-%Y")
    actual_dataset=actual_dataset.dropna(subset=['MC Group'])
    data=pd.DataFrame()
    a=actual_dataset['week_classifier'].unique().tolist()
    for i in range(0,len(a)):
        df=actual_dataset.copy()
        df=df[df['week_classifier']==a[i]]
        working_hours = (df['Clean_End_Time'].max()-df['Clean_Start_Time'].min())/ (np.timedelta64(1, 'h'))
        df['working_hours']=working_hours    
        data=pd.concat([data,df])
    actual_dataset=data.copy()
    actual_dataset['Cleaning Duration']=actual_dataset['Cleaning Duration']/60
    metrics_cip = actual_dataset.groupby(['MC Group','Constraint','week_classifier'], as_index = False).agg({
                                                    'clean_flag': 'sum',
                                                'Cleaning Duration':'sum',
                                                'working_hours':'mean'                                                })
    metrics_cip = metrics_cip.rename(columns = {'clean_flag':'No. of cleanings','Cleaning Duration':'CIP used Duration','working_hours':'Available Hours'})
    metrics_cip['CIP idle time'] = (metrics_cip['Available Hours'] - metrics_cip['CIP used Duration']).round(1)
    metrics_cip['CIP idle time%'] = ((metrics_cip['CIP idle time']/(metrics_cip['CIP used Duration']+metrics_cip['CIP idle time']))*100).round(1)
    metrics_cip['CIP Usage%'] = (((metrics_cip['CIP used Duration'])/(metrics_cip['CIP used Duration']+metrics_cip['CIP idle time']))*100).round(1)
    metrics_cip_total = metrics_cip.groupby(['MC Group','week_classifier'], as_index = False).agg({
                                                    'No. of cleanings': 'sum', #No of cleanings                                            
                                                    'CIP used Duration':'sum',#CIP used duration                                               
                                                    'Available Hours':'max'#CIP setup duration                                            
                                                    })
    metrics_cip_total['CIP idle time'] = (metrics_cip_total['Available Hours'] -metrics_cip_total['CIP used Duration']).round(1)
    metrics_cip_total['No. of possible cleanings'] = (metrics_cip_total['CIP idle time']/(metrics_cip_total['CIP used Duration']/metrics_cip_total['No. of cleanings'])).apply(np.floor)
    metrics_cip_total['CIP idle time%'] = ((metrics_cip_total['CIP idle time']/(metrics_cip_total['CIP used Duration']+metrics_cip_total['CIP idle time']))*100).round(1)
    metrics_cip_total['CIP Usage%'] = (((metrics_cip_total['CIP used Duration'])/(metrics_cip_total['CIP used Duration']+metrics_cip_total['CIP idle time']))*100).round(1)
    x1=metrics_cip_total

    x1['week_classifier']=pd.to_datetime(x1['week_classifier'],dayfirst=True,infer_datetime_format=True)
    x1=x1.sort_values('week_classifier')
    x1['week_classifier']=x1['week_classifier'].dt.strftime('%d-%b-%Y')
    ax1 = x1
    m=actual_dataset['Month'].unique().tolist()
    data_month=pd.DataFrame()
    for i in range(0,len(m)):
        df=actual_dataset.copy()
        df=df[df['Month']==m[i]]
        working_hours = (df['Clean_End_Time'].max()-df['Clean_Start_Time'].min())/ (np.timedelta64(1, 'h'))
        df['working_hours_month']=working_hours    
        data_month=pd.concat([data_month,df])
    actual_dataset=data_month.copy()
    metrics_cip = actual_dataset.groupby(['MC Group','Constraint','Month'], as_index = False).agg({
                                                    'clean_flag': 'sum',
                                                'Cleaning Duration':'sum',
                                                'working_hours_month':'mean'                                                })
    metrics_cip = metrics_cip.rename(columns = {
                            'clean_flag':'No. of cleanings','Cleaning Duration':'CIP used Duration','working_hours_month':'Available Hours'                        })
    metrics_cip['CIP idle time'] = (metrics_cip['Available Hours'] - metrics_cip['CIP used Duration']).round(1)
    metrics_cip['CIP idle time%'] = ((metrics_cip['CIP idle time']/(metrics_cip['CIP used Duration']+metrics_cip['CIP idle time']))*100).round(1)
    metrics_cip['CIP Usage%'] = (((metrics_cip['CIP used Duration'])/(metrics_cip['CIP used Duration']+metrics_cip['CIP idle time']))*100).round(1)
    metrics_cip_total = metrics_cip.groupby(['MC Group','Month'], as_index = False).agg({
                                                    'No. of cleanings': 'sum', #No of cleanings                                            
                                                    'CIP used Duration':'sum',#CIP used duration                                                
                                                    'Available Hours':'max'#CIP setup duration                                           
                                                    })
    metrics_cip_total['CIP idle time'] = (metrics_cip_total['Available Hours'] -metrics_cip_total['CIP used Duration']).round(1)
    metrics_cip_total['No. of possible cleanings'] = (metrics_cip_total['CIP idle time']/(metrics_cip_total['CIP used Duration']/metrics_cip_total['No. of cleanings'])).apply(np.floor)
    metrics_cip_total['CIP idle time%'] = ((metrics_cip_total['CIP idle time']/(metrics_cip_total['CIP used Duration']+metrics_cip_total['CIP idle time']))*100).round(1)
    metrics_cip_total['CIP Usage%'] = (((metrics_cip_total['CIP used Duration'])/(metrics_cip_total['CIP used Duration']+metrics_cip_total['CIP idle time']))*100).round(1)
    x2=metrics_cip_total
    x2['Month']=x2['Month'].map(lambda date_string: datetime.datetime.strptime(date_string,"%b-%Y"))
    x2=x2.sort_values('Month')
    x2['Month']=x2['Month'].dt.strftime('%b-%Y')
    ax2 = x2
    ax2['Schedule_type']='Actual'
    ax1['Schedule_type']='Actual'
else:
    ax1=pd.DataFrame()
    ax2=pd.DataFrame()
# # # #FOR MONTH
sample=pd.concat([px2,ax2])
sample2=pd.concat([px1,ax1])
sample2['week_classifier'] = pd.to_datetime(sample2['week_classifier'],format='%d-%b-%Y')
sample2=sample2.rename(columns={'week_classifier':'Week Ending'})
sample['CIP used Duration']=sample['CIP used Duration'].round(1)
sample2['CIP used Duration']=sample2['CIP used Duration'].round(1)
sample2['No. of cleanings'] = sample2['No. of cleanings'].astype(int)
#df = px.data.tips()# fig_month = px.bar(sample, x="Month", y="No. of cleanings", color="Schedule_type", barmode="group", facet_col="MC Group", text = 'No. of cleanings',title='CIP Monthly Usage %',#               category_orders={"Month": ["Jan", "Feb", "Mar", "Apr",'May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']},color_discrete_map={'Planned': 'rgb(104,131,246)','Actual': 'rgb(66,227,187)'})# fig_month.update_layout(legend_traceorder="reversed")# fig_month.update_traces(texttemplate='%{text:0f}',textposition='inside',textfont=dict(color="white"),cliponaxis = False)# df = px.data.tips()# fig_week = px.bar(sample2, x="Week Ending", y="No. of cleanings", color="MC Group", barmode="group", facet_col="Schedule_type",text = 'No. of cleanings',#               category_orders={"Month": ["Jan", "Feb", "Mar", "Apr",'May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']},color_discrete_map={'Planned': 'rgb(104,131,246)','Actual': 'rgb(66,227,187)'})# fig_week.update_layout(legend_traceorder="reversed")# fig_week.update_traces(texttemplate='%{text:0f}',textposition='inside',textfont=dict(color="white"),cliponaxis = False)# fig_week.update_xaxes(tickformat="%d-%b-%y")# fig_week.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))# fig_month.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))# fig_week.update_layout(legend=dict(#     orientation="h",#     yanchor="bottom",#     y=1.42,#     xanchor="right",#     x=1# ))# fig_month.update_layout(legend=dict(#     orientation="h",#     yanchor="bottom",#     y=1.42,#     xanchor="right",#     x=1# ))# fig_month.show()# fig_week.show()# fig_month=json.loads(plotly.io.to_json(fig_month))# fig_week=json.loads(plotly.io.to_json(fig_week))# util_flip= {#   "front":{#     "data":{#       "value":fig_month#       }#   },#   "back":{#     "data":{#       "value":fig_week#     }#   },#   "is_flip":True# }# dynamic_outputs=json.dumps(util_flip)
all_mc = sample.groupby(['Month','Schedule_type'],as_index=False).agg({
                                                    'No. of cleanings':'sum',
                                                    'CIP used Duration':'sum',
                                                    'Available Hours':'sum'})
all_mc['CIP Usage%'] = ((all_mc['CIP used Duration']/all_mc['Available Hours'])*100).round(1)
all_mc.insert(0,'MC Group','All')
cols=all_mc.columns.tolist()
sample=sample[cols]
final=pd.concat([sample,all_mc])
final['Month']=final['Month'].map(lambda date_string: datetime.datetime.strptime(date_string,"%b-%Y"))
final=final.sort_values('Month')
final['Month']=final['Month'].dt.strftime('%b-%Y')
final['CIP Usage%'] = final['CIP Usage%'].round()
final['CIP Usage%'] = final['CIP Usage%'].astype(int)
import plotly
import plotly.graph_objs as go
trace_all_Planned=go.Scatter(x=list(final[(final['MC Group']=='All') & (final['Schedule_type']=='Planned')]['Month']),
                         y=list(final[(final['MC Group']=='All') & (final['Schedule_type']=='Planned')]['CIP Usage%']),text = list(final[(final['MC Group']=='All') & (final['Schedule_type']=='Planned')]['CIP Usage%']), 
                         name='Planned',mode="lines+markers+text")
trace_all_Actual=go.Scatter(x=list(final[(final['MC Group']=='All') & (final['Schedule_type']=='Actual')]['Month']),
                         y=list(final[(final['MC Group']=='All') & (final['Schedule_type']=='Actual')]['CIP Usage%']), text = list(final[(final['MC Group']=='All') & (final['Schedule_type']=='Actual')]['CIP Usage%']),
                         name='Actual',mode="lines+markers+text")

data_loop=[trace_all_Actual,trace_all_Planned]
type=['Actual','Planned']
for j in sample['MC Group'].unique():
     for i in type:
          trace=go.Scatter(x=list(final[(final['MC Group']==j) & (final['Schedule_type']==i)]['Month']),
                         y=list(final[(final['MC Group']==j) & (final['Schedule_type']==i)]['CIP Usage%']), text = list(final[(final['MC Group']==j) & (final['Schedule_type']==i)]['CIP Usage%']),
                         name=i,visible=False,mode="lines+markers+text")
          data_loop.append(trace)
updatemenus = list([
    dict(type="buttons",
         active=-1,
         buttons=list([
            dict(label = 'All',
                 method = 'update',
                 args = [{'visible': [True, True, False,False,False,False,False,False,False,False]},
                         {'title': 'All Machine Groups'}]),
            dict(label = 'MC1',
                 method = 'update',
                 args = [{'visible': [False, False, True,True,False,False,False,False,False,False]},
                         {'title': 'MC1'}]),
            dict(label = 'MC2',
                 method = 'update',
                 args = [{'visible': [False, False, False,False,True,True,False,False,False,False]},
                         {'title': 'MC2'}]),
            dict(label = 'MC3',
                 method = 'update',
                 args = [{'visible': [False, False,False, False,False, False, True,True,False,False]},
                         {'title': 'MC3'}]),
            dict(label = 'MC4',
                 method = 'update',
                 args = [{'visible': [False, False,False, False,False, False, False,False,True,True]},
                         {'title': 'MC4'}])
        ]),
pad={"r": 10, "t": 10},
showactive=True,
name='Select Machine',
x=-0.3,
xanchor="left",
y=1,
yanchor="top"
    )
])
layout = dict(showlegend=True,
               updatemenus=updatemenus)
fig = dict(data=data_loop,layout=layout)
man=go.Figure(data=fig)
man.update_traces(textposition='top right')
man.update_traces(cliponaxis = False)
man.update_xaxes(tickformat='%Y-%b')
man.show()
all_mc_week = sample2.groupby(['Week Ending','Schedule_type'],as_index=False).agg({
                                                    'No. of cleanings':'sum',
                                                    'CIP used Duration':'sum',
                                                    'Available Hours':'sum'})
all_mc_week['CIP Usage%'] = ((all_mc_week['CIP used Duration']/all_mc_week['Available Hours'])*100).round(1)
all_mc_week = all_mc_week[all_mc_week['CIP Usage%']<100]
all_mc_week.insert(0,'MC Group','All')
cols=all_mc_week.columns.tolist()
sample2=sample2[cols]
final2=pd.concat([sample2,all_mc_week])
# final['Week Ending']=final['Week Ending'].map(lambda date_string: datetime.datetime.strptime(date_string,"%Y-%m-%d"))
final2=final2.sort_values('Week Ending')
final2['Week Ending']=final2['Week Ending'].dt.strftime('%Y-%m-%d')
final2['CIP Usage%'] = final2['CIP Usage%'].round()
final2['CIP Usage%'] = final2['CIP Usage%'].astype(int)
import plotly
import plotly.graph_objs as go
trace_all_Planned2=go.Scatter(x=list(final2[(final2['MC Group']=='All') & (final2['Schedule_type']=='Planned')]['Week Ending']),
                         y=list(final2[(final2['MC Group']=='All') & (final2['Schedule_type']=='Planned')]['CIP Usage%']),text = list(final2[(final2['MC Group']=='All') & (final2['Schedule_type']=='Planned')]['CIP Usage%']), 
                         name='Planned',mode="lines+markers+text")
trace_all_Actual2=go.Scatter(x=list(final2[(final2['MC Group']=='All') & (final2['Schedule_type']=='Actual')]['Week Ending']),
                         y=list(final2[(final2['MC Group']=='All') & (final2['Schedule_type']=='Actual')]['CIP Usage%']),text = list(final2[(final2['MC Group']=='All') & (final2['Schedule_type']=='Actual')]['CIP Usage%']),
                         name='Actual',mode="lines+markers+text")
data_loop2=[trace_all_Actual2,trace_all_Planned2]
type=['Actual','Planned']
for j in sample2['MC Group'].unique():
     for i in type:
          trace2=go.Scatter(x=list(final2[(final2['MC Group']==j) & (final2['Schedule_type']==i)]['Week Ending']),
                         y=list(final2[(final2['MC Group']==j) & (final2['Schedule_type']==i)]['CIP Usage%']),text = list(final2[(final2['MC Group']==j) & (final2['Schedule_type']==i)]['CIP Usage%']),
                         name=i,visible=False,mode="lines+markers+text")
          data_loop2.append(trace2)
updatemenus2 = list([
    dict(type="buttons",
         active=-1,
         buttons=list([
            dict(label = 'All',
                 method = 'update',
                 args = [{'visible': [True, True, False,False,False,False,False,False,False,False]},
                         {'title': 'All Machine Groups'}]),
            dict(label = 'MC1',
                 method = 'update',
                 args = [{'visible': [False, False, True,True,False,False,False,False,False,False]},
                         {'title': 'MC1'}]),
            dict(label = 'MC2',
                 method = 'update',
                 args = [{'visible': [False, False, False,False,True,True,False,False,False,False]},
                         {'title': 'MC2'}]),
            dict(label = 'MC3',
                 method = 'update',
                 args = [{'visible': [False, False,False, False,False, False, True,True,False,False]},
                         {'title': 'MC3'}]),
            dict(label = 'MC4',
                 method = 'update',
                 args = [{'visible': [False, False,False, False,False, False, False,False,True,True]},
                         {'title': 'MC4'}])
        ]),
pad={"r": 10, "t": 10},
showactive=True,
name='Select Machine',
x=-0.3,
xanchor="left",
y=1,
yanchor="top"
    )
])
layout2 = dict(showlegend=True,
               updatemenus=updatemenus2)
fig = dict(data=data_loop2,layout=layout2)
man2=go.Figure(data=fig)
man2.update_traces(textposition='top right')
man2.update_traces(cliponaxis = False)
man2.update_xaxes(tickformat='%d-%m-%Y')
man2.show()


fig_month=json.loads(plotly.io.to_json(man))
fig_week=json.loads(plotly.io.to_json(man2))

util_flip= {
  "front":{
    "data":{
      "value":fig_month
      }
  },
  "back":{
    "data":{
      "value":fig_week
    }
  },
  "is_flip":True
}

dynamic_outputs=json.dumps(util_flip)

"""


# In[28]:


visual_periodic_duration="""
import pandas as pd
from pathlib import Path
from azure.storage.blob import BlockBlobService
from io import StringIO
import json
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly
import datetime
data_source = 'azure_blob_storage'
connection_string = 'DefaultEndpointsProtocol=https;AccountName=mathcotakedastorage;AccountKey=s94v9Q0+SXPPRagBKJUfB2WOA2Ts5mFPUb4sUg+C0tRzagJel6eYeeyrA7Dlz7s2mmPorbtOJQohLgjefo++Vw==;EndpointSuffix=core.windows.net'
def get_ingested_data(file_path, datasource_type, connection_uri, container_name):
    path = Path(file_path)
    block_blob_service = BlockBlobService(connection_string=connection_uri)
    blob_data = block_blob_service.get_blob_to_text(container_name=container_name,
                                                    blob_name=file_path)
    ingested_df = pd.read_csv(StringIO(blob_data.content))
    return ingested_df
#Planned
dataset=get_ingested_data(file_path='merged_output_repository/merged_outputs.csv',datasource_type=data_source,connection_uri=connection_string,container_name='input')
dataset['Clean_Start_Time']=pd.to_datetime(dataset['Clean_Start_Time'],format='%Y/%m/%d %H:%M:%S')
dataset['Clean_End_Time']=pd.to_datetime(dataset['Clean_End_Time'],format='%Y/%m/%d %H:%M:%S')
dataset['Clean_Start_Time']= pd.to_datetime(dataset['Clean_Start_Time'],infer_datetime_format=True,utc=True)
dataset['Clean_End_Time']= pd.to_datetime(dataset['Clean_End_Time'],infer_datetime_format=True,utc=True)
actual_dataset=get_ingested_data(file_path='merged_output_repository/merged_actual_outputs.csv',datasource_type=data_source,connection_uri=connection_string,container_name='input')
actual_dataset['Clean_Start_Time']=pd.to_datetime(actual_dataset['Clean_Start_Time'],format='%Y/%m/%d %H:%M:%S')
actual_dataset['Clean_End_Time']=pd.to_datetime(actual_dataset['Clean_End_Time'],format='%Y/%m/%d %H:%M:%S')
actual_dataset['Clean_Start_Time']= pd.to_datetime(actual_dataset['Clean_Start_Time'],infer_datetime_format=True,utc=True)
actual_dataset['Clean_End_Time']= pd.to_datetime(actual_dataset['Clean_End_Time'],infer_datetime_format=True,utc=True)
def filter_data(selected_filter, df):
    cols_in_df = list(df.columns)
    # Filtering data    
    if len(selected_filters) > 0:
        for key in selected_filters.keys():
            if type(selected_filters[key])==dict and selected_filters[key].get('start_date',False) and selected_filters[key].get('end_date',False):
                selected_filters[key]['start_date']=pd.to_datetime(selected_filters[key]['start_date'],infer_datetime_format=True,utc=True)
                selected_filters[key]['end_date']=pd.to_datetime(selected_filters[key]['end_date'],infer_datetime_format=True,utc=True)
                print(selected_filters)
                df=df[(df['Clean_Start_Time']>selected_filters[key]['start_date']) & (df['Clean_End_Time']<selected_filters[key]['end_date'])]
            elif key in cols_in_df:
                if type(selected_filter[key]) == str or type(selected_filter[key]) == int:
                    if selected_filter[key] == 'All':
                        continue                    
                    else:
                        df = df[df[key] == selected_filter[key]]
                else:
                    if isinstance(selected_filter[key],list) and 'All' in selected_filter[key]:
                        continue                    
                    else:
                        df = df[df[key].isin(selected_filter[key])]
            else:
                continue    
    return df
dataset=filter_data(selected_filters,dataset)
actual_dataset=filter_data(selected_filters,actual_dataset)
actual_dataset = actual_dataset[actual_dataset['Cleaning_Type']=='Standard']
dataset = dataset.sort_values(by = "Clean_Start_Time")
dataset = dataset.reset_index(drop = True)
if len(dataset)!=0:
    df_week = pd.date_range(start=dataset['Clean_Start_Time'].min(),end=dataset['Clean_End_Time'].max(), freq = 'W').to_pydatetime().tolist()
    df_week = pd.DataFrame(df_week, columns =['week'])
    if len(df_week)==0:
        x=dataset['Clean_End_Time'].max()
        df_week=df_week.append({'week':x},ignore_index=True)
    a=df_week['week'].max()+datetime.timedelta(days=6)
    df_week=df_week.append({'week':a},ignore_index=True)
    rows = []
    dates=[]
    for i in range(0, len(df_week['week'])):
        df_week_4 = df_week.copy()
        df_week_4['week'] = pd.to_datetime(df_week_4['week'],utc=True).dt.date    
        date_str = str(df_week_4['week'][i])
        #date_str_new = date_str[9:]+"-"+date_str[6:7]+"-"+date_str[0:4]    
        date_obj = datetime.datetime.strptime(date_str,'%Y-%m-%d')
        start_week = date_obj.date()
        start_week_new = datetime.datetime.strftime(start_week, '%d-%m-%Y')
        end_week= start_week + datetime.timedelta(days=6)
        end_week_new = datetime.datetime.strftime(end_week, '%d-%m-%Y')
        rows.append([end_week_new])
        for date in (dataset['Clean_Start_Time']):
            if ((date<=end_week) and (date>=start_week)):
                dates.append(rows[i])
    date_data = pd.DataFrame(dates, columns=['week_number'])
    dataset['week_classifier'] = date_data['week_number']
    dataset['week_classifier']=pd.to_datetime(dataset['week_classifier'],format='%d-%m-%Y',utc=True).dt.date
    dataset['Month']=pd.DatetimeIndex(dataset['Clean_Start_Time']).strftime("%b-%Y")
    dataset=dataset.dropna(subset=['MC Group'])
    data=pd.DataFrame()
    a=dataset['week_classifier'].unique().tolist()
    for i in range(0,len(a)):
        df=dataset.copy()
        df=df[df['week_classifier']==a[i]]
        working_hours = (df['Clean_End_Time'].max()-df['Clean_Start_Time'].min())/ (np.timedelta64(1, 'h'))
        df['working_hours']=working_hours    
        data=pd.concat([data,df])
    dataset=data.copy()
    dataset['Cleaning Duration']=dataset['Cleaning Duration']/60
    dataset.drop_duplicates(subset =["Parallel Clean Flag",'MC Group','Constraint','Clean_Start_Time','Clean_End_Time'],keep='last',inplace=True)
    metrics_cip = dataset.groupby(['MC Group','Constraint','week_classifier'], as_index = False).agg({
                                                    'clean_flag': 'sum',
                                                'Cleaning Duration':'sum',
                                                'working_hours':'mean'                                                })
    metrics_cip = metrics_cip.rename(columns = {'clean_flag':'No. of cleanings','Cleaning Duration':'CIP used Duration','working_hours':'Available Hours'})
    metrics_cip['CIP idle time'] = (metrics_cip['Available Hours'] - metrics_cip['CIP used Duration']).round(1)
    metrics_cip['CIP idle time%'] = ((metrics_cip['CIP idle time']/(metrics_cip['CIP used Duration']+metrics_cip['CIP idle time']))*100).round(1)
    metrics_cip['CIP Usage%'] = (((metrics_cip['CIP used Duration'])/(metrics_cip['CIP used Duration']+metrics_cip['CIP idle time']))*100).round(1)
    metrics_cip_total = metrics_cip.groupby(['MC Group','week_classifier'], as_index = False).agg({
                                                    'No. of cleanings': 'sum', #No of cleanings                                            
                                                    'CIP used Duration':'sum',#CIP used duration                                                
                                                    'Available Hours':'max'#CIP setup duration                                            
                                                    })
    metrics_cip_total['CIP idle time'] = (metrics_cip_total['Available Hours'] -metrics_cip_total['CIP used Duration']).round(1)
    metrics_cip_total['No. of possible cleanings'] = (metrics_cip_total['CIP idle time']/(metrics_cip_total['CIP used Duration']/metrics_cip_total['No. of cleanings'])).apply(np.floor)
    metrics_cip_total['CIP idle time%'] = ((metrics_cip_total['CIP idle time']/(metrics_cip_total['CIP used Duration']+metrics_cip_total['CIP idle time']))*100).round(1)
    metrics_cip_total['CIP Usage%'] = (((metrics_cip_total['CIP used Duration'])/(metrics_cip_total['CIP used Duration']+metrics_cip_total['CIP idle time']))*100).round(1)
    x1=metrics_cip_total
    x1['week_classifier']=pd.to_datetime(x1['week_classifier'],dayfirst=True,infer_datetime_format=True)
    x1=x1.sort_values('week_classifier')
    x1['week_classifier']=x1['week_classifier'].dt.strftime('%d-%b-%Y')
    px1 = x1
    # # # #FOR MONTH
    m=dataset['Month'].unique().tolist()
    data_month=pd.DataFrame()
    for i in range(0,len(m)):
        df=dataset.copy()
        df=df[df['Month']==m[i]]
        working_hours = (df['Clean_End_Time'].max()-df['Clean_Start_Time'].min())/ (np.timedelta64(1, 'h'))
        df['working_hours_month']=working_hours    
        data_month=pd.concat([data_month,df])
    dataset=data_month.copy()
    metrics_cip = dataset.groupby(['MC Group','Constraint','Month'], as_index = False).agg({
                                                    'clean_flag': 'sum',
                                                'Cleaning Duration':'sum',
                                                'working_hours_month':'mean'                                               
                                                })
    metrics_cip = metrics_cip.rename(columns = {
                            'clean_flag':'No. of cleanings',
                            'Cleaning Duration':'CIP used Duration',
                            'working_hours_month':'Available Hours' })
    metrics_cip['CIP idle time'] = (metrics_cip['Available Hours'] - metrics_cip['CIP used Duration']).round(1)
    metrics_cip['CIP idle time%'] = ((metrics_cip['CIP idle time']/(metrics_cip['CIP used Duration']+metrics_cip['CIP idle time']))*100).round(1)
    metrics_cip['CIP Usage%'] = (((metrics_cip['CIP used Duration'])/(metrics_cip['CIP used Duration']+metrics_cip['CIP idle time']))*100).round(1)
    metrics_cip_total = metrics_cip.groupby(['MC Group','Month'], as_index = False).agg({
                                                    'No. of cleanings': 'sum', #No of cleanings                                           
                                                    'CIP used Duration':'sum',#CIP used duration                                                
                                                    'Available Hours':'max'#CIP setup duration                                            
                                                    })
    metrics_cip_total['CIP idle time'] = (metrics_cip_total['Available Hours'] -metrics_cip_total['CIP used Duration']).round(1)
    metrics_cip_total['No. of possible cleanings'] = (metrics_cip_total['CIP idle time']/(metrics_cip_total['CIP used Duration']/metrics_cip_total['No. of cleanings'])).apply(np.floor)
    metrics_cip_total['CIP idle time%'] = ((metrics_cip_total['CIP idle time']/(metrics_cip_total['CIP used Duration']+metrics_cip_total['CIP idle time']))*100).round(1)
    metrics_cip_total['CIP Usage%'] = (((metrics_cip_total['CIP used Duration'])/(metrics_cip_total['CIP used Duration']+metrics_cip_total['CIP idle time']))*100).round(1)
    x2=metrics_cip_total
    x2['Month']=x2['Month'].map(lambda date_string: datetime.datetime.strptime(date_string,"%b-%Y"))
    x2=x2.sort_values('Month')
    x2['Month']=x2['Month'].dt.strftime('%b-%Y')
    px2=x2
    px1['Schedule_type'] = 'Planned'
    px2['Schedule_type'] = 'Planned'
else:
    px1=pd.DataFrame()
    px2=pd.DataFrame()
#Actual
actual_dataset = actual_dataset.sort_values(by = "Clean_Start_Time")
actual_dataset = actual_dataset.reset_index(drop = True)
if len(actual_dataset)!=0:
    df_week = pd.date_range(start=actual_dataset['Clean_Start_Time'].min(),end=actual_dataset['Clean_End_Time'].max(), freq = 'W').to_pydatetime().tolist()
    df_week = pd.DataFrame(df_week, columns =['week'])
    if len(df_week)==0:
        x=actual_dataset['Clean_End_Time'].max()
        df_week=df_week.append({'week':x},ignore_index=True)
    a=df_week['week'].max()+datetime.timedelta(days=6)
    df_week=df_week.append({'week':a},ignore_index=True)
    rows = []
    dates=[]
    for i in range(0, len(df_week['week'])):
        df_week_4 = df_week.copy()
        df_week_4['week'] = pd.to_datetime(df_week_4['week'],utc=True).dt.date    
        date_str = str(df_week_4['week'][i])
        #date_str_new = date_str[9:]+"-"+date_str[6:7]+"-"+date_str[0:4]    
        date_obj = datetime.datetime.strptime(date_str,'%Y-%m-%d')
        start_week = date_obj.date()
        start_week_new = datetime.datetime.strftime(start_week, '%d-%m-%Y')
        end_week= start_week + datetime.timedelta(days=6)
        end_week_new = datetime.datetime.strftime(end_week, '%d-%m-%Y')
        rows.append([end_week_new])
        for date in (actual_dataset['Clean_Start_Time']):
            if ((date<=end_week) and (date>=start_week)):
                dates.append(rows[i])
    date_data = pd.DataFrame(dates, columns=['week_number'])
    actual_dataset['week_classifier'] = date_data['week_number']
    actual_dataset['week_classifier']=pd.to_datetime(actual_dataset['week_classifier'],format='%d-%m-%Y',utc=True).dt.date
    actual_dataset['Month']=pd.DatetimeIndex(actual_dataset['Clean_Start_Time']).strftime("%b-%Y")
    actual_dataset=actual_dataset.dropna(subset=['MC Group'])
    data=pd.DataFrame()
    a=actual_dataset['week_classifier'].unique().tolist()
    for i in range(0,len(a)):
        df=actual_dataset.copy()
        df=df[df['week_classifier']==a[i]]
        working_hours = (df['Clean_End_Time'].max()-df['Clean_Start_Time'].min())/ (np.timedelta64(1, 'h'))
        df['working_hours']=working_hours    
        data=pd.concat([data,df])
    actual_dataset=data.copy()
    actual_dataset['Cleaning Duration']=actual_dataset['Cleaning Duration']/60
    metrics_cip = actual_dataset.groupby(['MC Group','Constraint','week_classifier'], as_index = False).agg({
                                                    'clean_flag': 'sum',
                                                'Cleaning Duration':'sum',
                                                'working_hours':'mean'                                                })
    metrics_cip = metrics_cip.rename(columns = {'clean_flag':'No. of cleanings','Cleaning Duration':'CIP used Duration','working_hours':'Available Hours'})
    metrics_cip['CIP idle time'] = (metrics_cip['Available Hours'] - metrics_cip['CIP used Duration']).round(1)
    metrics_cip['CIP idle time%'] = ((metrics_cip['CIP idle time']/(metrics_cip['CIP used Duration']+metrics_cip['CIP idle time']))*100).round(1)
    metrics_cip['CIP Usage%'] = (((metrics_cip['CIP used Duration'])/(metrics_cip['CIP used Duration']+metrics_cip['CIP idle time']))*100).round(1)
    metrics_cip_total = metrics_cip.groupby(['MC Group','week_classifier'], as_index = False).agg({
                                                    'No. of cleanings': 'sum', #No of cleanings                                            
                                                    'CIP used Duration':'sum',#CIP used duration                                               
                                                    'Available Hours':'max'#CIP setup duration                                            
                                                    })
    metrics_cip_total['CIP idle time'] = (metrics_cip_total['Available Hours'] -metrics_cip_total['CIP used Duration']).round(1)
    metrics_cip_total['No. of possible cleanings'] = (metrics_cip_total['CIP idle time']/(metrics_cip_total['CIP used Duration']/metrics_cip_total['No. of cleanings'])).apply(np.floor)
    metrics_cip_total['CIP idle time%'] = ((metrics_cip_total['CIP idle time']/(metrics_cip_total['CIP used Duration']+metrics_cip_total['CIP idle time']))*100).round(1)
    metrics_cip_total['CIP Usage%'] = (((metrics_cip_total['CIP used Duration'])/(metrics_cip_total['CIP used Duration']+metrics_cip_total['CIP idle time']))*100).round(1)
    x1=metrics_cip_total

    x1['week_classifier']=pd.to_datetime(x1['week_classifier'],dayfirst=True,infer_datetime_format=True)
    x1=x1.sort_values('week_classifier')
    x1['week_classifier']=x1['week_classifier'].dt.strftime('%d-%b-%Y')
    ax1 = x1
    # # # #FOR MONTH
    m=actual_dataset['Month'].unique().tolist()
    data_month=pd.DataFrame()
    for i in range(0,len(m)):
        df=actual_dataset.copy()
        df=df[df['Month']==m[i]]
        working_hours = (df['Clean_End_Time'].max()-df['Clean_Start_Time'].min())/ (np.timedelta64(1, 'h'))
        df['working_hours_month']=working_hours    
        data_month=pd.concat([data_month,df])
    actual_dataset=data_month.copy()
    metrics_cip = actual_dataset.groupby(['MC Group','Constraint','Month'], as_index = False).agg({
                                                    'clean_flag': 'sum',
                                                'Cleaning Duration':'sum',
                                                'working_hours_month':'mean'                                                })
    metrics_cip = metrics_cip.rename(columns = {
                            'clean_flag':'No. of cleanings','Cleaning Duration':'CIP used Duration','working_hours_month':'Available Hours'                        })
    metrics_cip['CIP idle time'] = (metrics_cip['Available Hours'] - metrics_cip['CIP used Duration']).round(1)
    metrics_cip['CIP idle time%'] = ((metrics_cip['CIP idle time']/(metrics_cip['CIP used Duration']+metrics_cip['CIP idle time']))*100).round(1)
    metrics_cip['CIP Usage%'] = (((metrics_cip['CIP used Duration'])/(metrics_cip['CIP used Duration']+metrics_cip['CIP idle time']))*100).round(1)
    metrics_cip_total = metrics_cip.groupby(['MC Group','Month'], as_index = False).agg({
                                                    'No. of cleanings': 'sum', #No of cleanings                                            
                                                    'CIP used Duration':'sum',#CIP used duration                                                
                                                    'Available Hours':'max'#CIP setup duration                                           
                                                    })
    metrics_cip_total['CIP idle time'] = (metrics_cip_total['Available Hours'] -metrics_cip_total['CIP used Duration']).round(1)
    metrics_cip_total['No. of possible cleanings'] = (metrics_cip_total['CIP idle time']/(metrics_cip_total['CIP used Duration']/metrics_cip_total['No. of cleanings'])).apply(np.floor)
    metrics_cip_total['CIP idle time%'] = ((metrics_cip_total['CIP idle time']/(metrics_cip_total['CIP used Duration']+metrics_cip_total['CIP idle time']))*100).round(1)
    metrics_cip_total['CIP Usage%'] = (((metrics_cip_total['CIP used Duration'])/(metrics_cip_total['CIP used Duration']+metrics_cip_total['CIP idle time']))*100).round(1)
    x2=metrics_cip_total
    x2['Month']=x2['Month'].map(lambda date_string: datetime.datetime.strptime(date_string,"%b-%Y"))
    x2=x2.sort_values('Month')
    x2['Month']=x2['Month'].dt.strftime('%b-%Y')
    ax2 = x2
    ax1['Schedule_type']='Actual'
    ax2['Schedule_type']='Actual'
else:
    ax1=pd.DataFrame()
    ax2=pd.DataFrame()
sample=pd.concat([px2,ax2])
sample2=pd.concat([px1,ax1])
sample2['week_classifier'] = pd.to_datetime(sample2['week_classifier'],format='%d-%b-%Y')
sample2=sample2.rename(columns={'week_classifier':'Week Ending'})
sample['CIP used Duration']=sample['CIP used Duration'].round(1)
sample2['CIP used Duration']=sample2['CIP used Duration'].round(1)
sample2['No. of cleanings'] = sample2['No. of cleanings'].astype(int)
#df = px.data.tips()# fig_month = px.bar(sample, x="Month", y="No. of cleanings", color="Schedule_type", barmode="group", facet_col="MC Group", text = 'No. of cleanings',title='CIP Monthly Usage %',#               category_orders={"Month": ["Jan", "Feb", "Mar", "Apr",'May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']},color_discrete_map={'Planned': 'rgb(104,131,246)','Actual': 'rgb(66,227,187)'})# fig_month.update_layout(legend_traceorder="reversed")# fig_month.update_traces(texttemplate='%{text:0f}',textposition='inside',textfont=dict(color="white"),cliponaxis = False)# df = px.data.tips()# fig_week = px.bar(sample2, x="Week Ending", y="No. of cleanings", color="MC Group", barmode="group", facet_col="Schedule_type",text = 'No. of cleanings',#               category_orders={"Month": ["Jan", "Feb", "Mar", "Apr",'May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']},color_discrete_map={'Planned': 'rgb(104,131,246)','Actual': 'rgb(66,227,187)'})# fig_week.update_layout(legend_traceorder="reversed")# fig_week.update_traces(texttemplate='%{text:0f}',textposition='inside',textfont=dict(color="white"),cliponaxis = False)# fig_week.update_xaxes(tickformat="%d-%b-%y")# fig_week.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))# fig_month.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))# fig_week.update_layout(legend=dict(#     orientation="h",#     yanchor="bottom",#     y=1.42,#     xanchor="right",#     x=1# ))# fig_month.update_layout(legend=dict(#     orientation="h",#     yanchor="bottom",#     y=1.42,#     xanchor="right",#     x=1# ))# fig_month.show()# fig_week.show()# fig_month=json.loads(plotly.io.to_json(fig_month))# fig_week=json.loads(plotly.io.to_json(fig_week))# util_flip= {#   "front":{#     "data":{#       "value":fig_month#       }#   },#   "back":{#     "data":{#       "value":fig_week#     }#   },#   "is_flip":True# }# dynamic_outputs=json.dumps(util_flip)
all_mc = sample.groupby(['Month','Schedule_type'],as_index=False).agg({
                                                    'No. of cleanings':'sum',
                                                    'CIP used Duration':'sum',
                                                    'Available Hours':'sum'})
all_mc['CIP Usage%'] = ((all_mc['CIP used Duration']/all_mc['Available Hours'])*100).round(1)
all_mc.insert(0,'MC Group','All')
cols=all_mc.columns.tolist()
sample=sample[cols]
final=pd.concat([sample,all_mc])
final['Month']=final['Month'].map(lambda date_string: datetime.datetime.strptime(date_string,"%b-%Y"))
final=final.sort_values('Month')
final['Month']=final['Month'].dt.strftime('%b-%Y')
import plotly
import plotly.graph_objs as go
trace_all_Planned=go.Scatter(x=list(final[(final['MC Group']=='All') & (final['Schedule_type']=='Planned')]['Month']),
                         y=list(final[(final['MC Group']=='All') & (final['Schedule_type']=='Planned')]['No. of cleanings']),text = list(final[(final['MC Group']=='All') & (final['Schedule_type']=='Planned')]['No. of cleanings']),
                         name='Planned',mode="lines+markers+text")
trace_all_Actual=go.Scatter(x=list(final[(final['MC Group']=='All') & (final['Schedule_type']=='Actual')]['Month']),
                         y=list(final[(final['MC Group']=='All') & (final['Schedule_type']=='Actual')]['No. of cleanings']),text = list(final[(final['MC Group']=='All') & (final['Schedule_type']=='Actual')]['No. of cleanings']),
                         name='Actual',mode="lines+markers+text")
data_loop=[trace_all_Actual,trace_all_Planned]
type=['Actual','Planned']
for j in sample['MC Group'].unique():
     for i in type:
          trace=go.Scatter(x=list(final[(final['MC Group']==j) & (final['Schedule_type']==i)]['Month']),
                         y=list(final[(final['MC Group']==j) & (final['Schedule_type']==i)]['No. of cleanings']),text = list(final[(final['MC Group']==j) & (final['Schedule_type']==i)]['No. of cleanings']),
                         name=i,visible=False,mode="lines+markers+text")
          data_loop.append(trace)
updatemenus = list([
    dict(type="buttons",
         active=-1,
         buttons=list([
            dict(label = 'All',
                 method = 'update',
                 args = [{'visible': [True, True, False,False,False,False,False,False,False,False]},
                         {'title': 'All Machine Groups'}]),
            dict(label = 'MC1',
                 method = 'update',
                 args = [{'visible': [False, False, True,True,False,False,False,False,False,False]},
                         {'title': 'MC1'}]),
            dict(label = 'MC2',
                 method = 'update',
                 args = [{'visible': [False, False, False,False,True,True,False,False,False,False]},
                         {'title': 'MC2'}]),
            dict(label = 'MC3',
                 method = 'update',
                 args = [{'visible': [False, False,False, False,False, False, True,True,False,False]},
                         {'title': 'MC3'}]),
            dict(label = 'MC4',
                 method = 'update',
                 args = [{'visible': [False, False,False, False,False, False, False,False,True,True]},
                         {'title': 'MC4'}])
        ]),
pad={"r": 10, "t": 10},
showactive=True,
name='Select Machine',
x=-0.3,
xanchor="left",
y=1,
yanchor="top"
    )
])
layout = dict(showlegend=True,
               updatemenus=updatemenus)
fig = dict(data=data_loop,layout=layout)
man=go.Figure(data=fig)
man.update_traces(texttemplate='%{text:1s}',textposition='top right')
man.update_traces(cliponaxis = False)
man.update_xaxes(tickformat='%Y-%b')
man.show()
all_mc_week = sample2.groupby(['Week Ending','Schedule_type'],as_index=False).agg({
                                                    'No. of cleanings':'sum',
                                                    'CIP used Duration':'sum',
                                                    'Available Hours':'sum'})
all_mc_week['CIP Usage%'] = ((all_mc_week['CIP used Duration']/all_mc_week['Available Hours'])*100).round(1)
all_mc_week = all_mc_week[all_mc_week['CIP Usage%']<100]
all_mc_week.insert(0,'MC Group','All')
cols=all_mc_week.columns.tolist()
sample2=sample2[cols]
final2=pd.concat([sample2,all_mc_week])
# final['Week Ending']=final['Week Ending'].map(lambda date_string: datetime.datetime.strptime(date_string,"%Y-%m-%d"))
final2=final2.sort_values('Week Ending')
final2['Week Ending']=final2['Week Ending'].dt.strftime('%Y-%m-%d')
import plotly
import plotly.graph_objs as go
trace_all_Planned2=go.Scatter(x=list(final2[(final2['MC Group']=='All') & (final2['Schedule_type']=='Planned')]['Week Ending']),
                         y=list(final2[(final2['MC Group']=='All') & (final2['Schedule_type']=='Planned')]['No. of cleanings']),text = list(final2[(final2['MC Group']=='All') & (final2['Schedule_type']=='Planned')]['No. of cleanings']),
                         name='Planned',mode="lines+markers+text")
trace_all_Actual2=go.Scatter(x=list(final2[(final2['MC Group']=='All') & (final2['Schedule_type']=='Actual')]['Week Ending']),
                         y=list(final2[(final2['MC Group']=='All') & (final2['Schedule_type']=='Actual')]['No. of cleanings']),text = list(final2[(final2['MC Group']=='All') & (final2['Schedule_type']=='Actual')]['No. of cleanings']),
                         name='Actual',mode="lines+markers+text")
data_loop2=[trace_all_Actual2,trace_all_Planned2]
type=['Actual','Planned']
for j in sample2['MC Group'].unique():
     for i in type:
          trace2=go.Scatter(x=list(final2[(final2['MC Group']==j) & (final2['Schedule_type']==i)]['Week Ending']),
                         y=list(final2[(final2['MC Group']==j) & (final2['Schedule_type']==i)]['No. of cleanings']),text = list(final2[(final2['MC Group']==j) & (final2['Schedule_type']==i)]['No. of cleanings']),
                         name=i,visible=False,mode="lines+markers+text")
          data_loop2.append(trace2)
updatemenus2 = list([
    dict(type="buttons",
         active=-1,
         buttons=list([
            dict(label = 'All',
                 method = 'update',
                 args = [{'visible': [True, True, False,False,False,False,False,False,False,False]},
                         {'title': 'All Machine Groups'}]),
            dict(label = 'MC1',
                 method = 'update',
                 args = [{'visible': [False, False, True,True,False,False,False,False,False,False]},
                         {'title': 'MC1'}]),
            dict(label = 'MC2',
                 method = 'update',
                 args = [{'visible': [False, False, False,False,True,True,False,False,False,False]},
                         {'title': 'MC2'}]),
            dict(label = 'MC3',
                 method = 'update',
                 args = [{'visible': [False, False,False, False,False, False, True,True,False,False]},
                         {'title': 'MC3'}]),
            dict(label = 'MC4',
                 method = 'update',
                 args = [{'visible': [False, False,False, False,False, False, False,False,True,True]},
                         {'title': 'MC4'}])
        ]),
pad={"r": 10, "t": 10},
showactive=True,
name='Select Machine',
x=-0.3,
xanchor="left",
y=1,
yanchor="top"
    )
])
layout2 = dict(showlegend=True,
               updatemenus=updatemenus2)
fig = dict(data=data_loop2,layout=layout2)
man2=go.Figure(data=fig)
man2.update_traces(texttemplate='%{text:1s}',textposition='top right')
man2.update_traces(cliponaxis = False)
man2.update_xaxes(tickformat='%d-%m-%Y')
man2.show()


fig_month=json.loads(plotly.io.to_json(man))
fig_week=json.loads(plotly.io.to_json(man2))

util_flip= {
  "front":{
    "data":{
      "value":fig_month
      }
  },
  "back":{
    "data":{
      "value":fig_week
    }
  },
  "is_flip":True
}

dynamic_outputs=json.dumps(util_flip)
"""


# # Resource Utilization Metrics Code String #

# In[29]:


filter_code_string_Resource_Metrics='''
import pandas as pd
from pathlib import Path
from azure.storage.blob import BlockBlobService
from io import StringIO
import json

data_source = 'azure_blob_storage'
connection_string = 'DefaultEndpointsProtocol=https;AccountName=mathcotakedastorage;AccountKey=s94v9Q0+SXPPRagBKJUfB2WOA2Ts5mFPUb4sUg+C0tRzagJel6eYeeyrA7Dlz7s2mmPorbtOJQohLgjefo++Vw==;EndpointSuffix=core.windows.net'

def get_ingested_data(file_path, datasource_type, connection_uri, container_name):
    path = Path(file_path)
    block_blob_service = BlockBlobService(connection_string=connection_uri)
    blob_data = block_blob_service.get_blob_to_text(container_name=container_name, 
                                                    blob_name=file_path)
    ingested_df = pd.read_csv(StringIO(blob_data.content))

    return ingested_df


def get_filter_options(df= pd.DataFrame(), on_col="", using_cols={}, default_val='All'):
    """gets the filter options from a column based on a set of dependent columns
    and the values provided against these columns

    Args:
        df ([type]): DataFrame to be searched in.
        on_col (str): Column to be searched in.
        using_cols (dict, optional): the unique values to be searched in column 'on' will be based on the options provided
        in this dict, Its keys will be columns and values will be list of values. Defaults to {}.
        default_val (str, optional): a default value to be passed when no filter options are present. Defaults to 'All'
        if default_val == ''/None, Nothing will be added to dict

    Returns:
        [list]: list of options for the filter based on the data passed in using_cols param
    """
    cols_in_df = list(df.columns)

    filter_options = []
    if on_col in cols_in_df:
        if len(using_cols) == 0:
            filter_options = list(df[on_col].unique())
        else:
            for key in using_cols.keys():
                if key in cols_in_df:
                    if type(using_cols[key]) == str or type(using_cols[key]) == int :
                        if using_cols[key] == 'All':
                            continue
                        else:
                            df = df[df[key] == using_cols[key]]
                    else:
                        df = df[df[key].isin(using_cols[key])]
                else:
                    continue
        filter_options = list(df[on_col].unique())
    if default_val:
        filter_options.insert(0, default_val)
    return filter_options

# Reference
list_of_cols = ['Resource','Date']
using_cols_for_test = {}

schedule=get_ingested_data(file_path='merged_output_repository/merged_outputs.csv',datasource_type=data_source,connection_uri=connection_string,container_name='input')
def_options_for_filters= {}
for x in list_of_cols:
       #\print('------------------'+x+'---------------------------')
    if x == 'Resource':
        def_options_for_filters[x]=get_filter_options(schedule, on_col=x, using_cols=using_cols_for_test)
    elif x== 'Date':
        def_options_for_filters[x]={'start_date':str(pd.to_datetime(schedule['Clean_Start_Time'],infer_datetime_format=True,utc=True).min()), 'end_date': str(pd.to_datetime(schedule['Clean_End_Time'],infer_datetime_format=True,utc=True).max())}
    else:
        def_options_for_filters[x]={}


fil = {
    'Resource': {
        'index': 0,
        'label': 'Resource',
        'type': 'multiple',
        'options': def_options_for_filters['Resource']
    },
    'Date':{
        'index':1,
        'label':'Date Range',
        'type':'date_range',
        'options':[]
    }
}



def generate_filter_json(current_filter_params,filter_options=fil, default_values=def_options_for_filters):
    basic_filter_dict = {
        "widget_filter_index": 0,
        "widget_filter_function": False,
        "widget_filter_function_parameter": False,
        "widget_filter_hierarchy_key": False,
        "widget_filter_isall": False,
        "widget_filter_multiselect": False,
        "widget_tag_input_type": "select",
        "widget_tag_key": "",
        "widget_tag_label": "",
        "widget_tag_value": [],
        "widget_filter_type": "",
        "widget_filter_params":None
    }
    dataValues = []
    defaultValues = {}

    for filter in filter_options.keys():
        instance_dict = dict(basic_filter_dict)
        instance_dict['widget_tag_key'] = filter
        instance_dict['widget_filter_index'] = filter_options[filter]['index']
        instance_dict['widget_tag_label'] = filter_options[filter]['label']
        instance_dict['widget_tag_value'] = filter_options[filter]['options']
        instance_dict['widget_filter_multiselect'] = True if filter_options[filter]['type'] == 'multiple' else False
        dataValues.append(instance_dict)
        if filter_options[filter]['type']=='date_range':
            instance_dict['widget_filter_type']='date_range'
            instance_dict['widget_filter_params']={'start_date':{'format':"DD/MM/yyyy"},'end_date':{'format':"DD/MM/yyyy"}}
        if current_filter_params=={}:
          defaultValues[filter] = default_values.get(
            filter, ['All'] if instance_dict['widget_filter_multiselect'] else 'All')
        else:
          defaultValues[filter] = current_filter_params['selected'].get(
            filter, ['All'] if instance_dict['widget_filter_multiselect'] else 'All')
    final_json = {'dataValues': dataValues, 'defaultValues': defaultValues}
    return final_json

dynamic_outputs=json.dumps(generate_filter_json(current_filter_params,filter_options=fil, default_values=def_options_for_filters))
print(dynamic_outputs)
'''


# In[30]:


kpi_res_reclean_overall="""
import pandas as pd
from pathlib import Path
from azure.storage.blob import BlockBlobService
from io import StringIO
import json
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly

data_source = 'azure_blob_storage'
connection_string = 'DefaultEndpointsProtocol=https;AccountName=mathcotakedastorage;AccountKey=s94v9Q0+SXPPRagBKJUfB2WOA2Ts5mFPUb4sUg+C0tRzagJel6eYeeyrA7Dlz7s2mmPorbtOJQohLgjefo++Vw==;EndpointSuffix=core.windows.net'

def get_ingested_data(file_path, datasource_type, connection_uri, container_name):
    path = Path(file_path)
    block_blob_service = BlockBlobService(connection_string=connection_uri)
    blob_data = block_blob_service.get_blob_to_text(container_name=container_name, 
                                                    blob_name=file_path)
    ingested_df = pd.read_csv(StringIO(blob_data.content))

    return ingested_df

dataset=get_ingested_data(file_path='merged_output_repository/merged_outputs.csv',datasource_type=data_source,connection_uri=connection_string,container_name='input')
dataset['Clean_Start_Time']=pd.to_datetime(dataset['Clean_Start_Time'],format='%Y/%m/%d %H:%M:%S')
dataset['Usage_End']=pd.to_datetime(dataset['Usage_End'],format='%Y/%m/%d %H:%M:%S')
dataset['Clean_End_Time']=pd.to_datetime(dataset['Clean_End_Time'],format='%Y/%m/%d %H:%M:%S')
dataset['Usage_Start']=pd.to_datetime(dataset['Usage_Start'],format='%Y/%m/%d %H:%M:%S')
dataset['Usage_Start']= pd.to_datetime(dataset['Usage_Start'],infer_datetime_format=True,utc=True)
dataset['Usage_End']= pd.to_datetime(dataset['Usage_End'],infer_datetime_format=True,utc=True)
dataset['Clean_Start_Time']= pd.to_datetime(dataset['Clean_Start_Time'],infer_datetime_format=True,utc=True)
dataset['Clean_End_Time']= pd.to_datetime(dataset['Clean_End_Time'],infer_datetime_format=True,utc=True)
dataset['MC Group'] = dataset['MC Group'].fillna(1)
dataset['Missed Cleanings'] = np.where(dataset['MC Group']==1,1,0)
def filter_data(selected_filter, df):
    cols_in_df = list(df.columns)
    # Filtering data
    if len(selected_filters) > 0:
        for key in selected_filters.keys():
            if type(selected_filters[key])==dict and selected_filters[key].get('start_date',False) and selected_filters[key].get('end_date',False):
                selected_filters[key]['start_date']=pd.to_datetime(selected_filters[key]['start_date'],infer_datetime_format=True,utc=True)
                selected_filters[key]['end_date']=pd.to_datetime(selected_filters[key]['end_date'],infer_datetime_format=True,utc=True)
                print(selected_filters)
                df=df[(df['Clean_Start_Time']>selected_filters[key]['start_date']) & (df['Clean_End_Time']<selected_filters[key]['end_date'])]
            elif key in cols_in_df:
                if type(selected_filter[key]) == str or type(selected_filter[key]) == int:
                    if selected_filter[key] == 'All':
                        continue
                    else:
                        df = df[df[key] == selected_filter[key]]
                else:
                    if isinstance(selected_filter[key],list) and 'All' in selected_filter[key]:
                        continue
                    else:
                        df = df[df[key].isin(selected_filter[key])]
            else:
                continue
    return df
dataset=filter_data(selected_filters,dataset)

def business_metrics_resource(dataset):
    if dataset['Clean_Start_Time'].min() > dataset['Usage_Start'].min():
        working_hours = (dataset['Clean_End_Time'].max()-dataset['Usage_Start'].min())/ (np.timedelta64(1, 'h'))
    else:
        working_hours = (dataset['Clean_End_Time'].max()-dataset['Clean_Start_Time'].min())/ (np.timedelta64(1, 'h'))
    dataset['Number of Pre-cleanings']= np.where(((dataset['Reclean Status']==1) & 
                                                (dataset['preclean']==1) & 
                                                (dataset['Cleaning Type']=='Pre-Cleaning')),1,0)
    bm = dataset.groupby('Resource Alt', as_index = False).agg({'Usage_Start': 'min',
                                                    'Usage_End': 'max',
                                                    'Utilized': 'sum',
                                                   'run_flag': 'sum',  
                                                   'clean_flag': 'sum',
                                                   'Unutilized_gap':'sum',
                                                   'Cleaning Duration':'max',
                                                   'DHT Violation Flag': 'sum',
                                                   'CHT Violation Flag':'sum',
                                                   'Number of Pre-cleanings':'sum',
                                                    'Parallel Clean Flag':pd.Series.nunique,
                                                            'Missed Cleanings':'sum'
                                                  })
    
    bm['Unutilized_gap'] = bm['Unutilized_gap'].round(1)
    bm['Utilized'] = bm['Utilized'].round(1)
    bm.Schedule_Start= pd.to_datetime(bm.Usage_Start, format = '%d/%m/%Y %H:%M:%S')
    bm.Schedule_End= pd.to_datetime(bm.Usage_End, format = '%d/%m/%Y %H:%M:%S')
    bm = bm.rename(columns = {'Resource Alt':'Resource','Usage_Start': 'Schedule Start', 'Usage_End': 'Schedule End', 'run_flag':'No. of Utilizations',
                              'clean_flag': 'Total no. of Cleanings','DHT Violation Flag': 'DHT Violations','Unutilized_gap':'Unutilized time',
                             'Utilized':'Utilized Duration','CHT Violation Flag':'CHT Violations','CHT Violation Flag Pre Clean':'Number of Pre-cleanings',
                              'Parallel Clean Flag':'Parallel Cleanings'})
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
def business_metrics_CIP(dataset):
    if dataset['Clean_Start_Time'].min() > dataset['Usage_Start'].min():
        working_hours = (dataset['Clean_End_Time'].max()-dataset['Usage_Start'].min())/ (np.timedelta64(1, 'h'))
    else:
        working_hours = (dataset['Clean_End_Time'].max()-dataset['Clean_Start_Time'].min())/ (np.timedelta64(1, 'h'))
    dataset['timebw'] = dataset['Clean_Start_Time'] - dataset['Usage_End']
    dataset['timebw'] = dataset['timebw'] / np.timedelta64(1, 'h')
    dataset['setup_flag'] = np.where(dataset['timebw'] > 0.166 , 1, 0)
    dataset['setup_flag'] = dataset['setup_flag']*10
    dataset = dataset.rename(columns = {'setup_flag': 'Setup time'})
    dataset['Setup time'] = (dataset['Setup time']/60).round(1)
    metrics_cip = dataset.groupby(['MC Group','Constraint'], as_index = False).agg({
                                                   'Resource': pd.Series.nunique, 
                                                    'clean_flag': 'sum', 
                                                   'Cleaning Duration':'sum',
                                                    'Setup time':'sum',
                                                    'Parallel Clean Flag':pd.Series.nunique
                                                    })
    metrics_cip = metrics_cip.rename(columns = {'Resource': 'Resources cleaned',
                               'clean_flag':'No. of cleanings','Cleaning Duration':'CIP used Duration',
                               })
    metrics_cip['No. of cleanings'] = metrics_cip['No. of cleanings'] - metrics_cip['Parallel Clean Flag']
    metrics_cip['Available Hours']= working_hours
    metrics_cip['CIP idle time'] = (metrics_cip['Available Hours'] - metrics_cip['CIP used Duration']).round(1)
    metrics_cip['CIP idle time%'] = ((metrics_cip['CIP idle time']/(metrics_cip['CIP used Duration']+metrics_cip['Setup time']+metrics_cip['CIP idle time']))*100).round(1)
    metrics_cip['CIP Usage%'] = (((metrics_cip['CIP used Duration']+metrics_cip['Setup time'])/(metrics_cip['CIP used Duration']+metrics_cip['Setup time']+metrics_cip['CIP idle time']))*100).round(1)

    metrics_cip_total = metrics_cip.groupby(['MC Group'], as_index = False).agg({
                                                   'Resources cleaned': 'sum',
                                                    'No. of cleanings': 'sum', #No of cleanings
                                                   'CIP used Duration':'sum',#CIP used duration
                                                    'Setup time':'sum',
                                                    'Available Hours':'max',
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
        df1=df1.append({'MC Group':i,'Min Downtime between 2 CIP cleanings':inter_arrival_time},ignore_index=True)
        df1['Min Downtime between 2 CIP cleanings']=df1['Min Downtime between 2 CIP cleanings'].round(2)
        df1 = df1.dropna()
    metrics_cip_total = pd.merge(metrics_cip_total,df1,on=['MC Group'])
    return metrics_cip,metrics_cip_total
df_res = business_metrics_resource(dataset)
df_pipe,df_cip = business_metrics_CIP(dataset)
non_missed = dataset[dataset['MC Group'].isin(['MC1','MC2','MC3','MC4'])]
def pre_re_cleaning(dataframe):   
    df1 = dataframe[(dataframe['preclean']==1)]
    df1['Cleaning Duration'] = df1['Cleaning Duration']/60
    data_temp = pd.DataFrame()
    for res in dataframe.Resource.unique():
        temp_df = dataframe[dataframe['Resource']==res]
        y=temp_df['Usage_Start'].min()
        data_temp = data_temp.append({"Resource":res,'Min_Usage_Start':y},ignore_index=True)
    new = pd.merge(df1,data_temp,on=['Resource'])
    new['CHT 1st U']=np.where((new['Usage_Start']== new['Min_Usage_Start']),'PREclean','REclean')
    x = new[new['CHT 1st U']=='PREclean'].shape[0]
    y =new[new['CHT 1st U']=='REclean'].shape[0]
    pre_re = pd.DataFrame([['Total no. of Pre-cleanings ',x],['Total no. of Re-cleanings',y]],columns=['Insights','Count'])
    return pre_re,new 
m,new1 = pre_re_cleaning(non_missed)
recleanings=df_res['Number of Pre-cleanings'].sum().round(0)
def convert(o):
    if isinstance(o, np.generic): return o.item()  
    raise TypeError
recleanings=convert(recleanings)

res_recleaning_kpi= {
  "value":str(recleanings)
}

dynamic_outputs=json.dumps(res_recleaning_kpi)
print(dynamic_outputs)
"""


# In[31]:


kpi_res_dht_overall="""
import pandas as pd
from pathlib import Path
from azure.storage.blob import BlockBlobService
from io import StringIO
import json
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly

data_source = 'azure_blob_storage'
connection_string = 'DefaultEndpointsProtocol=https;AccountName=mathcotakedastorage;AccountKey=s94v9Q0+SXPPRagBKJUfB2WOA2Ts5mFPUb4sUg+C0tRzagJel6eYeeyrA7Dlz7s2mmPorbtOJQohLgjefo++Vw==;EndpointSuffix=core.windows.net'

def get_ingested_data(file_path, datasource_type, connection_uri, container_name):
    path = Path(file_path)
    block_blob_service = BlockBlobService(connection_string=connection_uri)
    blob_data = block_blob_service.get_blob_to_text(container_name=container_name, 
                                                    blob_name=file_path)
    ingested_df = pd.read_csv(StringIO(blob_data.content))

    return ingested_df

dataset=get_ingested_data(file_path='merged_output_repository/merged_outputs.csv',datasource_type=data_source,connection_uri=connection_string,container_name='input')
dataset['Clean_Start_Time']=pd.to_datetime(dataset['Clean_Start_Time'],format='%Y/%m/%d %H:%M:%S')
dataset['Usage_End']=pd.to_datetime(dataset['Usage_End'],format='%Y/%m/%d %H:%M:%S')
dataset['Clean_End_Time']=pd.to_datetime(dataset['Clean_End_Time'],format='%Y/%m/%d %H:%M:%S')
dataset['Usage_Start']=pd.to_datetime(dataset['Usage_Start'],format='%Y/%m/%d %H:%M:%S')
dataset['Usage_Start']= pd.to_datetime(dataset['Usage_Start'],infer_datetime_format=True,utc=True)
dataset['Usage_End']= pd.to_datetime(dataset['Usage_End'],infer_datetime_format=True,utc=True)
dataset['Clean_Start_Time']= pd.to_datetime(dataset['Clean_Start_Time'],infer_datetime_format=True,utc=True)
dataset['Clean_End_Time']= pd.to_datetime(dataset['Clean_End_Time'],infer_datetime_format=True,utc=True)
dataset['MC Group'] = dataset['MC Group'].fillna(1)
dataset['Missed Cleanings'] = np.where(dataset['MC Group']==1,1,0)
def filter_data(selected_filter, df):
    cols_in_df = list(df.columns)
    # Filtering data
    if len(selected_filters) > 0:
        for key in selected_filters.keys():
            if type(selected_filters[key])==dict and selected_filters[key].get('start_date',False) and selected_filters[key].get('end_date',False):
                selected_filters[key]['start_date']=pd.to_datetime(selected_filters[key]['start_date'],infer_datetime_format=True,utc=True)
                selected_filters[key]['end_date']=pd.to_datetime(selected_filters[key]['end_date'],infer_datetime_format=True,utc=True)
                print(selected_filters)
                df=df[(df['Usage_Start']>selected_filters[key]['start_date']) & (df['Usage_End']<selected_filters[key]['end_date'])]
            elif key in cols_in_df:
                if type(selected_filter[key]) == str or type(selected_filter[key]) == int:
                    if selected_filter[key] == 'All':
                        continue
                    else:
                        df = df[df[key] == selected_filter[key]]
                else:
                    if isinstance(selected_filter[key],list) and 'All' in selected_filter[key]:
                        continue
                    else:
                        df = df[df[key].isin(selected_filter[key])]
            else:
                continue
    return df
dataset=filter_data(selected_filters,dataset)
def business_metrics_resource(dataset):
    if dataset['Clean_Start_Time'].min() > dataset['Usage_Start'].min():
        working_hours = (dataset['Clean_End_Time'].max()-dataset['Usage_Start'].min())/ (np.timedelta64(1, 'h'))
    else:
        working_hours = (dataset['Clean_End_Time'].max()-dataset['Clean_Start_Time'].min())/ (np.timedelta64(1, 'h'))
    dataset['Number of Pre-cleanings']= np.where(((dataset['Reclean Status']==1) & 
                                                (dataset['preclean']==1) & 
                                                (dataset['Cleaning Type']=='Pre-Cleaning')),1,0)
    bm = dataset.groupby('Resource Alt', as_index = False).agg({'Usage_Start': 'min',
                                                    'Usage_End': 'max',
                                                    'Utilized': 'sum',
                                                   'run_flag': 'sum',  
                                                   'clean_flag': 'sum',
                                                   'Unutilized_gap':'sum',
                                                   'Cleaning Duration':'max',
                                                   'DHT Violation Flag': 'sum',
                                                   'CHT Violation Flag':'sum',
                                                   'Number of Pre-cleanings':'sum',
                                                    'Parallel Clean Flag':pd.Series.nunique,
                                                            'Missed Cleanings':'sum'
                                                  })
    
    bm['Unutilized_gap'] = bm['Unutilized_gap'].round(1)
    bm['Utilized'] = bm['Utilized'].round(1)
    bm.Schedule_Start= pd.to_datetime(bm.Usage_Start, format = '%d/%m/%Y %H:%M:%S')
    bm.Schedule_End= pd.to_datetime(bm.Usage_End, format = '%d/%m/%Y %H:%M:%S')
    bm = bm.rename(columns = {'Resource Alt':'Resource','Usage_Start': 'Schedule Start', 'Usage_End': 'Schedule End', 'run_flag':'No. of Utilizations',
                              'clean_flag': 'Total no. of Cleanings','DHT Violation Flag': 'DHT Violations','Unutilized_gap':'Unutilized time',
                             'Utilized':'Utilized Duration','CHT Violation Flag':'CHT Violations','CHT Violation Flag Pre Clean':'Number of Pre-cleanings',
                              'Parallel Clean Flag':'Parallel Cleanings'})
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
def business_metrics_CIP(dataset):
    if dataset['Clean_Start_Time'].min() > dataset['Usage_Start'].min():
        working_hours = (dataset['Clean_End_Time'].max()-dataset['Usage_Start'].min())/ (np.timedelta64(1, 'h'))
    else:
        working_hours = (dataset['Clean_End_Time'].max()-dataset['Clean_Start_Time'].min())/ (np.timedelta64(1, 'h'))
    dataset['timebw'] = dataset['Clean_Start_Time'] - dataset['Usage_End']
    dataset['timebw'] = dataset['timebw'] / np.timedelta64(1, 'h')
    dataset['setup_flag'] = np.where(dataset['timebw'] > 0.166 , 1, 0)
    dataset['setup_flag'] = dataset['setup_flag']*10
    dataset = dataset.rename(columns = {'setup_flag': 'Setup time'})
    dataset['Setup time'] = (dataset['Setup time']/60).round(1)
    metrics_cip = dataset.groupby(['MC Group','Constraint'], as_index = False).agg({
                                                   'Resource': pd.Series.nunique, 
                                                    'clean_flag': 'sum', 
                                                   'Cleaning Duration':'sum',
                                                    'Setup time':'sum',
                                                    'Parallel Clean Flag':pd.Series.nunique
                                                    })
    metrics_cip = metrics_cip.rename(columns = {'Resource': 'Resources cleaned',
                               'clean_flag':'No. of cleanings','Cleaning Duration':'CIP used Duration',
                               })
    metrics_cip['No. of cleanings'] = metrics_cip['No. of cleanings'] - metrics_cip['Parallel Clean Flag']
    metrics_cip['Available Hours']= working_hours
    metrics_cip['CIP idle time'] = (metrics_cip['Available Hours'] - metrics_cip['CIP used Duration']).round(1)
    metrics_cip['CIP idle time%'] = ((metrics_cip['CIP idle time']/(metrics_cip['CIP used Duration']+metrics_cip['Setup time']+metrics_cip['CIP idle time']))*100).round(1)
    metrics_cip['CIP Usage%'] = (((metrics_cip['CIP used Duration']+metrics_cip['Setup time'])/(metrics_cip['CIP used Duration']+metrics_cip['Setup time']+metrics_cip['CIP idle time']))*100).round(1)

    metrics_cip_total = metrics_cip.groupby(['MC Group'], as_index = False).agg({
                                                   'Resources cleaned': 'sum',
                                                    'No. of cleanings': 'sum', #No of cleanings
                                                   'CIP used Duration':'sum',#CIP used duration
                                                    'Setup time':'sum',
                                                    'Available Hours':'max',
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
        df1=df1.append({'MC Group':i,'Min Downtime between 2 CIP cleanings':inter_arrival_time},ignore_index=True)
        df1['Min Downtime between 2 CIP cleanings']=df1['Min Downtime between 2 CIP cleanings'].round(2)
        df1 = df1.dropna()
    metrics_cip_total = pd.merge(metrics_cip_total,df1,on=['MC Group'])
    return metrics_cip,metrics_cip_total
df_res = business_metrics_resource(dataset)
kpi_dht=df_res['DHT Violations'].sum().round(0)
def convert(o):
    if isinstance(o, np.generic): return o.item()  
    raise TypeError
dht=convert(kpi_dht)

res_dht_kpi= {
  "value":str(dht)
}

dynamic_outputs=json.dumps(res_dht_kpi)
print(dynamic_outputs)
"""


# In[32]:


kpi_res_cht_overall="""
import pandas as pd
from pathlib import Path
from azure.storage.blob import BlockBlobService
from io import StringIO
import json
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly

data_source = 'azure_blob_storage'
connection_string = 'DefaultEndpointsProtocol=https;AccountName=mathcotakedastorage;AccountKey=s94v9Q0+SXPPRagBKJUfB2WOA2Ts5mFPUb4sUg+C0tRzagJel6eYeeyrA7Dlz7s2mmPorbtOJQohLgjefo++Vw==;EndpointSuffix=core.windows.net'

def get_ingested_data(file_path, datasource_type, connection_uri, container_name):
    path = Path(file_path)
    block_blob_service = BlockBlobService(connection_string=connection_uri)
    blob_data = block_blob_service.get_blob_to_text(container_name=container_name, 
                                                    blob_name=file_path)
    ingested_df = pd.read_csv(StringIO(blob_data.content))

    return ingested_df

dataset=get_ingested_data(file_path='merged_output_repository/merged_outputs.csv',datasource_type=data_source,connection_uri=connection_string,container_name='input')
dataset['Clean_Start_Time']=pd.to_datetime(dataset['Clean_Start_Time'],format='%Y/%m/%d %H:%M:%S')
dataset['Usage_End']=pd.to_datetime(dataset['Usage_End'],format='%Y/%m/%d %H:%M:%S')
dataset['Clean_End_Time']=pd.to_datetime(dataset['Clean_End_Time'],format='%Y/%m/%d %H:%M:%S')
dataset['Usage_Start']=pd.to_datetime(dataset['Usage_Start'],format='%Y/%m/%d %H:%M:%S')
dataset['Usage_Start']= pd.to_datetime(dataset['Usage_Start'],infer_datetime_format=True,utc=True)
dataset['Usage_End']= pd.to_datetime(dataset['Usage_End'],infer_datetime_format=True,utc=True)
dataset['Clean_Start_Time']= pd.to_datetime(dataset['Clean_Start_Time'],infer_datetime_format=True,utc=True)
dataset['Clean_End_Time']= pd.to_datetime(dataset['Clean_End_Time'],infer_datetime_format=True,utc=True)
dataset['MC Group'] = dataset['MC Group'].fillna(1)
dataset['Missed Cleanings'] = np.where(dataset['MC Group']==1,1,0)
def filter_data(selected_filter, df):
    cols_in_df = list(df.columns)
    # Filtering data
    if len(selected_filters) > 0:
        for key in selected_filters.keys():
            if type(selected_filters[key])==dict and selected_filters[key].get('start_date',False) and selected_filters[key].get('end_date',False):
                selected_filters[key]['start_date']=pd.to_datetime(selected_filters[key]['start_date'],infer_datetime_format=True,utc=True)
                selected_filters[key]['end_date']=pd.to_datetime(selected_filters[key]['end_date'],infer_datetime_format=True,utc=True)
                print(selected_filters)
                df=df[(df['Usage_Start']>selected_filters[key]['start_date']) & (df['Usage_End']<selected_filters[key]['end_date'])]
            elif key in cols_in_df:
                if type(selected_filter[key]) == str or type(selected_filter[key]) == int:
                    if selected_filter[key] == 'All':
                        continue
                    else:
                        df = df[df[key] == selected_filter[key]]
                else:
                    if isinstance(selected_filter[key],list) and 'All' in selected_filter[key]:
                        continue
                    else:
                        df = df[df[key].isin(selected_filter[key])]
            else:
                continue
    return df
dataset=filter_data(selected_filters,dataset)
kpi_cht=dataset['CHT Violation Flag'].sum().round(0)
def convert(o):
    if isinstance(o, np.generic): return o.item()  
    raise TypeError
cht=convert(kpi_cht)

res_cht_kpi= {
  "value":str(cht)
}

dynamic_outputs=json.dumps(res_cht_kpi)
print(dynamic_outputs)
"""


# In[33]:


kpi_res_usage_overall="""
import pandas as pd
from pathlib import Path
from azure.storage.blob import BlockBlobService
from io import StringIO
import json
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly

data_source = 'azure_blob_storage'
connection_string = 'DefaultEndpointsProtocol=https;AccountName=mathcotakedastorage;AccountKey=s94v9Q0+SXPPRagBKJUfB2WOA2Ts5mFPUb4sUg+C0tRzagJel6eYeeyrA7Dlz7s2mmPorbtOJQohLgjefo++Vw==;EndpointSuffix=core.windows.net'

def get_ingested_data(file_path, datasource_type, connection_uri, container_name):
    path = Path(file_path)
    block_blob_service = BlockBlobService(connection_string=connection_uri)
    blob_data = block_blob_service.get_blob_to_text(container_name=container_name, 
                                                    blob_name=file_path)
    ingested_df = pd.read_csv(StringIO(blob_data.content))

    return ingested_df

dataset=get_ingested_data(file_path='merged_output_repository/merged_outputs.csv',datasource_type=data_source,connection_uri=connection_string,container_name='input')
dataset['Clean_Start_Time']=pd.to_datetime(dataset['Clean_Start_Time'],format='%Y/%m/%d %H:%M:%S')
dataset['Usage_End']=pd.to_datetime(dataset['Usage_End'],format='%Y/%m/%d %H:%M:%S')
dataset['Clean_End_Time']=pd.to_datetime(dataset['Clean_End_Time'],format='%Y/%m/%d %H:%M:%S')
dataset['Usage_Start']=pd.to_datetime(dataset['Usage_Start'],format='%Y/%m/%d %H:%M:%S')
dataset['Usage_Start']= pd.to_datetime(dataset['Usage_Start'],infer_datetime_format=True,utc=True)
dataset['Usage_End']= pd.to_datetime(dataset['Usage_End'],infer_datetime_format=True,utc=True)
dataset['Clean_Start_Time']= pd.to_datetime(dataset['Clean_Start_Time'],infer_datetime_format=True,utc=True)
dataset['Clean_End_Time']= pd.to_datetime(dataset['Clean_End_Time'],infer_datetime_format=True,utc=True)
dataset['MC Group'] = dataset['MC Group'].fillna(1)
dataset['Missed Cleanings'] = np.where(dataset['MC Group']==1,1,0)
def filter_data(selected_filter, df):
    cols_in_df = list(df.columns)
    # Filtering data
    if len(selected_filters) > 0:
        for key in selected_filters.keys():
            if type(selected_filters[key])==dict and selected_filters[key].get('start_date',False) and selected_filters[key].get('end_date',False):
                selected_filters[key]['start_date']=pd.to_datetime(selected_filters[key]['start_date'],infer_datetime_format=True,utc=True)
                selected_filters[key]['end_date']=pd.to_datetime(selected_filters[key]['end_date'],infer_datetime_format=True,utc=True)
                print(selected_filters)
                df=df[(df['Usage_Start']>selected_filters[key]['start_date']) & (df['Usage_End']<selected_filters[key]['end_date'])]
            elif key in cols_in_df:
                if type(selected_filter[key]) == str or type(selected_filter[key]) == int:
                    if selected_filter[key] == 'All':
                        continue
                    else:
                        df = df[df[key] == selected_filter[key]]
                else:
                    if isinstance(selected_filter[key],list) and 'All' in selected_filter[key]:
                        continue
                    else:
                        df = df[df[key].isin(selected_filter[key])]
            else:
                continue
    return df
dataset=filter_data(selected_filters,dataset)
def business_metrics_resource(dataset):
    if dataset['Clean_Start_Time'].min() > dataset['Usage_Start'].min():
        working_hours = (dataset['Clean_End_Time'].max()-dataset['Usage_Start'].min())/ (np.timedelta64(1, 'h'))
    else:
        working_hours = (dataset['Clean_End_Time'].max()-dataset['Clean_Start_Time'].min())/ (np.timedelta64(1, 'h'))
    dataset['Number of Pre-cleanings']= np.where(((dataset['Reclean Status']==1) & 
                                                (dataset['preclean']==1) & 
                                                (dataset['Cleaning Type']=='Pre-Cleaning')),1,0)
    bm = dataset.groupby('Resource Alt', as_index = False).agg({'Usage_Start': 'min',
                                                    'Usage_End': 'max',
                                                    'Utilized': 'sum',
                                                   'run_flag': 'sum',  
                                                   'clean_flag': 'sum',
                                                   'Unutilized_gap':'sum',
                                                   'Cleaning Duration':'max',
                                                   'DHT Violation Flag': 'sum',
                                                   'CHT Violation Flag':'sum',
                                                   'Number of Pre-cleanings':'sum',
                                                    'Parallel Clean Flag':pd.Series.nunique,
                                                            'Missed Cleanings':'sum'
                                                  })
    
    bm['Unutilized_gap'] = bm['Unutilized_gap'].round(1)
    bm['Utilized'] = bm['Utilized'].round(1)
    bm.Schedule_Start= pd.to_datetime(bm.Usage_Start, format = '%d/%m/%Y %H:%M:%S')
    bm.Schedule_End= pd.to_datetime(bm.Usage_End, format = '%d/%m/%Y %H:%M:%S')
    bm = bm.rename(columns = {'Resource Alt':'Resource','Usage_Start': 'Schedule Start', 'Usage_End': 'Schedule End', 'run_flag':'No. of Utilizations',
                              'clean_flag': 'Total no. of Cleanings','DHT Violation Flag': 'DHT Violations','Unutilized_gap':'Unutilized time',
                             'Utilized':'Utilized Duration','CHT Violation Flag':'CHT Violations','CHT Violation Flag Pre Clean':'Number of Pre-cleanings',
                              'Parallel Clean Flag':'Parallel Cleanings'})
    bm['Total no. of Cleanings'] = bm['Total no. of Cleanings'] - bm['Missed Cleanings'] - bm['Parallel Cleanings']
    bm['Available working Hours'] = round(working_hours,1)
    bm['Cleaning time Available'] = (bm['Available working Hours'] - bm['Utilized Duration']).round(1)
    bm['Total actual Cleaning time'] = (bm['Total no. of Cleanings']*(bm['Cleaning Duration']))
    bm['Resource idle duration'] = (bm['Cleaning time Available'] - bm['Total actual Cleaning time']).round(1) 
    bm['Avg Idle Time %'] = ((bm['Resource idle duration']/bm['Available working Hours'])*100).round(1)
    bm['Resource Usage %'] = ((bm['Utilized Duration']/bm['Available working Hours'])*100).round(1)
    bm.loc[(bm['Resource']=='Flexible'),['Utilized Duration','Unutilized time','Available working Hours','Cleaning time Available','Total actual Cleaning time','Resource idle duration','Avg Idle Time %','Resource Usage %']] = ''
    return bm

df_res=business_metrics_resource(dataset)
df_res['Resource Usage %'] = df_res['Resource Usage %'].apply(pd.to_numeric)
df_res = df_res.sort_values(by = ['Resource Usage %'],ascending=False)
highest_usage_value = df_res['Resource Usage %'].iloc[0]
highest_usage_value = highest_usage_value.round(0)
highest_usage_res = df_res['Resource'].iloc[0]
kpi_usage_overall=pd.to_numeric(df_res['Resource Usage %']).mean().round(0)

def convert(o):
    if isinstance(o, np.generic): return o.item()  
    raise TypeError
kpi_usage_overall=convert(kpi_usage_overall)

res_usage_kpi= {
  "value": str(highest_usage_res) + ": " + str(int(highest_usage_value)) + " %"
}

dynamic_outputs=json.dumps(res_usage_kpi)
print(dynamic_outputs)
"""


# # Resource Utilization Visualization Code String #

# In[34]:


vis_reclean_overall="""
import pandas as pd
from pathlib import Path
from azure.storage.blob import BlockBlobService
from io import StringIO
import json
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly
import datetime as datetime

data_source = 'azure_blob_storage'
connection_string = 'DefaultEndpointsProtocol=https;AccountName=mathcotakedastorage;AccountKey=s94v9Q0+SXPPRagBKJUfB2WOA2Ts5mFPUb4sUg+C0tRzagJel6eYeeyrA7Dlz7s2mmPorbtOJQohLgjefo++Vw==;EndpointSuffix=core.windows.net'

def get_ingested_data(file_path, datasource_type, connection_uri, container_name):
    path = Path(file_path)
    block_blob_service = BlockBlobService(connection_string=connection_uri)
    blob_data = block_blob_service.get_blob_to_text(container_name=container_name,
                                                    blob_name=file_path)
    ingested_df = pd.read_csv(StringIO(blob_data.content))

    return ingested_df

dataset=get_ingested_data(file_path='merged_output_repository/merged_outputs.csv',datasource_type=data_source,connection_uri=connection_string,container_name='input')
dataset['Clean_Start_Time']=pd.to_datetime(dataset['Clean_Start_Time'],format='%Y/%m/%d %H:%M:%S')
dataset['Usage_End']=pd.to_datetime(dataset['Usage_End'],format='%Y/%m/%d %H:%M:%S')
dataset['Clean_End_Time']=pd.to_datetime(dataset['Clean_End_Time'],format='%Y/%m/%d %H:%M:%S')
dataset['Usage_Start']=pd.to_datetime(dataset['Usage_Start'],format='%Y/%m/%d %H:%M:%S')
dataset['Usage_Start']= pd.to_datetime(dataset['Usage_Start'],infer_datetime_format=True,utc=True)
dataset['Usage_End']= pd.to_datetime(dataset['Usage_End'],infer_datetime_format=True,utc=True)
dataset['Clean_Start_Time']= pd.to_datetime(dataset['Clean_Start_Time'],infer_datetime_format=True,utc=True)
dataset['Clean_End_Time']= pd.to_datetime(dataset['Clean_End_Time'],infer_datetime_format=True,utc=True)
dataset['MC Group'] = dataset['MC Group'].fillna(1)
dataset['Missed Cleanings'] = np.where(dataset['MC Group']==1,1,0)
def filter_data(selected_filter, df):
    cols_in_df = list(df.columns)
    # Filtering data
    if len(selected_filters) > 0:
        for key in selected_filters.keys():
            if type(selected_filters[key])==dict and selected_filters[key].get('start_date',False) and selected_filters[key].get('end_date',False):
                selected_filters[key]['start_date']=pd.to_datetime(selected_filters[key]['start_date'],infer_datetime_format=True,utc=True)
                selected_filters[key]['end_date']=pd.to_datetime(selected_filters[key]['end_date'],infer_datetime_format=True,utc=True)
                print(selected_filters)
                df=df[(df['Clean_Start_Time']>selected_filters[key]['start_date']) & (df['Clean_End_Time']<selected_filters[key]['end_date'])]
            elif key in cols_in_df:
                if type(selected_filter[key]) == str or type(selected_filter[key]) == int:
                    if selected_filter[key] == 'All':
                        continue
                    else:
                        df = df[df[key] == selected_filter[key]]
                else:
                    if isinstance(selected_filter[key],list) and 'All' in selected_filter[key]:
                        continue
                    else:
                        df = df[df[key].isin(selected_filter[key])]
            else:
                continue
    return df
dataset=filter_data(selected_filters,dataset)
dataset = dataset.sort_values(by = "Clean_Start_Time")
dataset = dataset.reset_index(drop = True)
def pre_re_cleaning(dataframe):
    df1 = dataframe[(dataframe['preclean']==1) & ~(dataframe['MC Group']==' ')]
    df1['MC Group'].replace('', np.nan, inplace=True)
    df1.dropna(subset=['MC Group'], inplace=True)
    data_temp = pd.DataFrame()
    for res in dataframe.Resource.unique():
        temp_df = dataframe[dataframe['Resource']==res]
        y=temp_df['Usage_Start'].min()
        data_temp = data_temp.append({"Resource":res,'Min_Usage_Start':y},ignore_index=True)
    new = pd.merge(df1,data_temp,on=['Resource'])
    new['CHT 1st U']=np.where((new['Usage_Start']== new['Min_Usage_Start']),'PREclean','REclean')
    new1 = new[new['MC Group'] !=1]
    pre_re = pd.DataFrame()
    for res in new1.Resource.unique():
        temp_df1 = new1[new1['Resource']==res]
        pre1 = temp_df1[temp_df1['CHT 1st U']=='PREclean'].shape[0]
        re1 = temp_df1[temp_df1['CHT 1st U']=='REclean'].shape[0]
        pre_re = pre_re.append({'Resource':res,'PREcleans':pre1,'REcleans':re1},ignore_index=True)
    pre_re1 = pre_re.set_index('Resource')
    pre_re1=pre_re1.reset_index(drop=False)
    return pre_re1
x1=pre_re_cleaning(dataset)
x1['Total']=x1['PREcleans']+x1['REcleans']
x1=x1.sort_values('Total',ascending=False).head(10).reset_index(drop=True)
fig_cht = go.Figure(data=[
    go.Bar(name='Pre-Cleaning', x=x1['Resource'], y=x1['PREcleans'],text=x1['PREcleans']),
    go.Bar(name='Re-Cleaning', x=x1['Resource'], y=x1['REcleans'],text=x1['REcleans'])
])
# Change the bar mode
fig_cht.update_traces(texttemplate='%{text:0f}',textposition='outside',width=0.3,cliponaxis = False)
fig_cht.update_layout(barmode='stack')
fig_cht.update_layout(xaxis_title='Resource',
                  yaxis_title='# Recleanings & Precleanings')
fig_cht.show()
util_dist_period=json.loads(plotly.io.to_json(fig_cht))
dynamic_outputs=json.dumps(util_dist_period)
"""


# In[35]:


cht_violation_trend="""
import pandas as pd
from pathlib import Path
from azure.storage.blob import BlockBlobService
from io import StringIO
import json
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly


data_source = 'azure_blob_storage'
connection_string = 'DefaultEndpointsProtocol=https;AccountName=mathcotakedastorage;AccountKey=s94v9Q0+SXPPRagBKJUfB2WOA2Ts5mFPUb4sUg+C0tRzagJel6eYeeyrA7Dlz7s2mmPorbtOJQohLgjefo++Vw==;EndpointSuffix=core.windows.net'

def get_ingested_data(file_path, datasource_type, connection_uri, container_name):
    path = Path(file_path)
    block_blob_service = BlockBlobService(connection_string=connection_uri)
    blob_data = block_blob_service.get_blob_to_text(container_name=container_name,
                                                    blob_name=file_path)
    ingested_df = pd.read_csv(StringIO(blob_data.content))

    return ingested_df

dataset=get_ingested_data(file_path='merged_output_repository/merged_outputs.csv',datasource_type=data_source,connection_uri=connection_string,container_name='input')
dataset['Clean_Start_Time']=pd.to_datetime(dataset['Clean_Start_Time'],format='%Y/%m/%d %H:%M:%S')
dataset['Usage_End']=pd.to_datetime(dataset['Usage_End'],format='%Y/%m/%d %H:%M:%S')
dataset['Clean_End_Time']=pd.to_datetime(dataset['Clean_End_Time'],format='%Y/%m/%d %H:%M:%S')
dataset['Usage_Start']=pd.to_datetime(dataset['Usage_Start'],format='%Y/%m/%d %H:%M:%S')
dataset['Usage_Start']= pd.to_datetime(dataset['Usage_Start'],infer_datetime_format=True,utc=True)
dataset['Usage_End']= pd.to_datetime(dataset['Usage_End'],infer_datetime_format=True,utc=True)
dataset['Clean_Start_Time']= pd.to_datetime(dataset['Clean_Start_Time'],infer_datetime_format=True,utc=True)
dataset['Clean_End_Time']= pd.to_datetime(dataset['Clean_End_Time'],infer_datetime_format=True,utc=True)
dataset['MC Group'] = dataset['MC Group'].fillna(1)
dataset['Missed Cleanings'] = np.where(dataset['MC Group']==1,1,0)
def filter_data(selected_filter, df):
    cols_in_df = list(df.columns)
    # Filtering data
    if len(selected_filters) > 0:
        for key in selected_filters.keys():
            if type(selected_filters[key])==dict and selected_filters[key].get('start_date',False) and selected_filters[key].get('end_date',False):
                selected_filters[key]['start_date']=pd.to_datetime(selected_filters[key]['start_date'],infer_datetime_format=True,utc=True)
                selected_filters[key]['end_date']=pd.to_datetime(selected_filters[key]['end_date'],infer_datetime_format=True,utc=True)
                print(selected_filters)
                df=df[(df['Clean_Start_Time']>selected_filters[key]['start_date']) & (df['Clean_End_Time']<selected_filters[key]['end_date'])]
            elif key in cols_in_df:
                if type(selected_filter[key]) == str or type(selected_filter[key]) == int:
                    if selected_filter[key] == 'All':
                        continue
                    else:
                        df = df[df[key] == selected_filter[key]]
                else:
                    if isinstance(selected_filter[key],list) and 'All' in selected_filter[key]:
                        continue
                    else:
                        df = df[df[key].isin(selected_filter[key])]
            else:
                continue
    return df


dataset=filter_data(selected_filters,dataset)
cht= dataset.groupby('Resource Alt',as_index = False).agg({'CHT Violation Flag':'sum'})
cht=cht[cht['CHT Violation Flag']>0]
cht=cht.sort_values('CHT Violation Flag',ascending=False).head(10).reset_index(drop=True)
cht

fig = px.bar(cht,
             y=cht['CHT Violation Flag'],
             x=cht['Resource Alt'], text=cht['CHT Violation Flag'],title='Top 10 Resources CHT Violations').update_xaxes(categoryorder="total descending")
fig.update_traces(texttemplate='%{text:0f}',textposition='outside',width=0.3,cliponaxis = False)
fig.update_layout(xaxis_title='Resource',
                  yaxis_title='# CHT Violations')
fig.show()
util_dist_period=json.loads(plotly.io.to_json(fig))
dynamic_outputs=json.dumps(util_dist_period)
"""


# In[36]:


dht_violation_trend="""
import pandas as pd
from pathlib import Path
from azure.storage.blob import BlockBlobService
from io import StringIO
import json
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly


data_source = 'azure_blob_storage'
connection_string = 'DefaultEndpointsProtocol=https;AccountName=mathcotakedastorage;AccountKey=s94v9Q0+SXPPRagBKJUfB2WOA2Ts5mFPUb4sUg+C0tRzagJel6eYeeyrA7Dlz7s2mmPorbtOJQohLgjefo++Vw==;EndpointSuffix=core.windows.net'

def get_ingested_data(file_path, datasource_type, connection_uri, container_name):
    path = Path(file_path)
    block_blob_service = BlockBlobService(connection_string=connection_uri)
    blob_data = block_blob_service.get_blob_to_text(container_name=container_name,
                                                    blob_name=file_path)
    ingested_df = pd.read_csv(StringIO(blob_data.content))

    return ingested_df

dataset=get_ingested_data(file_path='merged_output_repository/merged_outputs.csv',datasource_type=data_source,connection_uri=connection_string,container_name='input')
dataset['Clean_Start_Time']=pd.to_datetime(dataset['Clean_Start_Time'],format='%Y/%m/%d %H:%M:%S')
dataset['Usage_End']=pd.to_datetime(dataset['Usage_End'],format='%Y/%m/%d %H:%M:%S')
dataset['Clean_End_Time']=pd.to_datetime(dataset['Clean_End_Time'],format='%Y/%m/%d %H:%M:%S')
dataset['Usage_Start']=pd.to_datetime(dataset['Usage_Start'],format='%Y/%m/%d %H:%M:%S')
dataset['Usage_Start']= pd.to_datetime(dataset['Usage_Start'],infer_datetime_format=True,utc=True)
dataset['Usage_End']= pd.to_datetime(dataset['Usage_End'],infer_datetime_format=True,utc=True)
dataset['Clean_Start_Time']= pd.to_datetime(dataset['Clean_Start_Time'],infer_datetime_format=True,utc=True)
dataset['Clean_End_Time']= pd.to_datetime(dataset['Clean_End_Time'],infer_datetime_format=True,utc=True)
dataset['MC Group'] = dataset['MC Group'].fillna(1)
dataset['Missed Cleanings'] = np.where(dataset['MC Group']==1,1,0)
def filter_data(selected_filter, df):
    cols_in_df = list(df.columns)
    # Filtering data
    if len(selected_filters) > 0:
        for key in selected_filters.keys():
            if type(selected_filters[key])==dict and selected_filters[key].get('start_date',False) and selected_filters[key].get('end_date',False):
                selected_filters[key]['start_date']=pd.to_datetime(selected_filters[key]['start_date'],infer_datetime_format=True,utc=True)
                selected_filters[key]['end_date']=pd.to_datetime(selected_filters[key]['end_date'],infer_datetime_format=True,utc=True)
                print(selected_filters)
                df=df[(df['Usage_Start']>selected_filters[key]['start_date']) & (df['Usage_End']<selected_filters[key]['end_date'])]
            elif key in cols_in_df:
                if type(selected_filter[key]) == str or type(selected_filter[key]) == int:
                    if selected_filter[key] == 'All':
                        continue
                    else:
                        df = df[df[key] == selected_filter[key]]
                else:
                    if isinstance(selected_filter[key],list) and 'All' in selected_filter[key]:
                        continue
                    else:
                        df = df[df[key].isin(selected_filter[key])]
            else:
                continue
    return df


dataset=filter_data(selected_filters,dataset)
dht= dataset.groupby('Resource',as_index = False).agg({'DHT Violation Flag':'sum'})
dht=dht[dht['DHT Violation Flag']>0]
dht=dht.sort_values('DHT Violation Flag',ascending=False).head(10).reset_index(drop=True)
fig = px.bar(dht,
             y=dht['DHT Violation Flag'],
             x=dht['Resource'], text=dht['DHT Violation Flag'],title='Top 10 Resources DHT Violations').update_xaxes(categoryorder="total descending")
fig.update_traces(texttemplate='%{text:0f}',textposition='outside',width=0.3,cliponaxis = False)
fig.update_layout(xaxis_title='Resource',
                  yaxis_title='# DHT Violations')
fig.show()
util_dist_period=json.loads(plotly.io.to_json(fig))
dynamic_outputs=json.dumps(util_dist_period)
"""


# In[37]:


visual_usage_family_trend="""
import pandas as pd
from pathlib import Path
from azure.storage.blob import BlockBlobService
from io import StringIO
import json
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly


data_source = 'azure_blob_storage'
connection_string = 'DefaultEndpointsProtocol=https;AccountName=mathcotakedastorage;AccountKey=s94v9Q0+SXPPRagBKJUfB2WOA2Ts5mFPUb4sUg+C0tRzagJel6eYeeyrA7Dlz7s2mmPorbtOJQohLgjefo++Vw==;EndpointSuffix=core.windows.net'

def get_ingested_data(file_path, datasource_type, connection_uri, container_name):
    path = Path(file_path)
    block_blob_service = BlockBlobService(connection_string=connection_uri)
    blob_data = block_blob_service.get_blob_to_text(container_name=container_name,
                                                    blob_name=file_path)
    ingested_df = pd.read_csv(StringIO(blob_data.content))

    return ingested_df

dataset=get_ingested_data(file_path='merged_output_repository/merged_outputs.csv',datasource_type=data_source,connection_uri=connection_string,container_name='input')
dataset['Clean_Start_Time']=pd.to_datetime(dataset['Clean_Start_Time'],format='%Y/%m/%d %H:%M:%S')
dataset['Usage_End']=pd.to_datetime(dataset['Usage_End'],format='%Y/%m/%d %H:%M:%S')
dataset['Clean_End_Time']=pd.to_datetime(dataset['Clean_End_Time'],format='%Y/%m/%d %H:%M:%S')
dataset['Usage_Start']=pd.to_datetime(dataset['Usage_Start'],format='%Y/%m/%d %H:%M:%S')
dataset['Usage_Start']= pd.to_datetime(dataset['Usage_Start'],infer_datetime_format=True,utc=True)
dataset['Usage_End']= pd.to_datetime(dataset['Usage_End'],infer_datetime_format=True,utc=True)
dataset['Clean_Start_Time']= pd.to_datetime(dataset['Clean_Start_Time'],infer_datetime_format=True,utc=True)
dataset['Clean_End_Time']= pd.to_datetime(dataset['Clean_End_Time'],infer_datetime_format=True,utc=True)
dataset['MC Group'] = dataset['MC Group'].fillna(1)
dataset['Missed Cleanings'] = np.where(dataset['MC Group']==1,1,0)
def filter_data(selected_filter, df):
    cols_in_df = list(df.columns)
    # Filtering data
    if len(selected_filters) > 0:
        for key in selected_filters.keys():
            if type(selected_filters[key])==dict and selected_filters[key].get('start_date',False) and selected_filters[key].get('end_date',False):
                selected_filters[key]['start_date']=pd.to_datetime(selected_filters[key]['start_date'],infer_datetime_format=True,utc=True)
                selected_filters[key]['end_date']=pd.to_datetime(selected_filters[key]['end_date'],infer_datetime_format=True,utc=True)
                print(selected_filters)
                df=df[(df['Usage_Start']>selected_filters[key]['start_date']) & (df['Usage_End']<selected_filters[key]['end_date'])]
            elif key in cols_in_df:
                if type(selected_filter[key]) == str or type(selected_filter[key]) == int:
                    if selected_filter[key] == 'All':
                        continue
                    else:
                        df = df[df[key] == selected_filter[key]]
                else:
                    if isinstance(selected_filter[key],list) and 'All' in selected_filter[key]:
                        continue
                    else:
                        df = df[df[key].isin(selected_filter[key])]
            else:
                continue
    return df
dataset=filter_data(selected_filters,dataset)
if dataset['Clean_Start_Time'].min() > dataset['Usage_Start'].min():
        working_hours = (dataset['Clean_End_Time'].max()-dataset['Usage_Start'].min())/ (np.timedelta64(1, 'h'))
else:
        working_hours = (dataset['Clean_End_Time'].max()-dataset['Clean_Start_Time'].min())/ (np.timedelta64(1, 'h'))
schedule1 = dataset[dataset['run_flag']==1]
bm = schedule1.groupby('Resource', as_index = False).agg({'Utilized': 'sum'})
bm['working_hours'] = working_hours
bm = bm.rename(columns = {'Resource': 'Resource Alt'})
am = bm[bm["Resource Alt"].str.contains('Fl')==False]
am = am[am["Resource Alt"].str.contains('P')==False]
am = am[am["Resource Alt"].str.contains('D')==False]
am = am[am["Resource Alt"].str.contains('SA')==False]
am = am[am["Resource Alt"].str.contains('SB')==False]
am = am[am["Resource Alt"].str.contains('SC')==False]
am = am[am["Resource Alt"].str.contains('SD')==False]
usage= pd.DataFrame(  [['DOME',bm[bm['Resource Alt'].str[:2].isin(['Do'])].Utilized.sum(),bm[bm['Resource Alt'].str[:2].isin(['Do'])].working_hours.sum()],
                      ['THAWING TANK',bm[bm['Resource Alt'].str[:1].isin(['P'])].Utilized.sum(),bm[bm['Resource Alt'].str[:1].isin(['P'])].working_hours.sum()],
                      ['MASS CAPTURE TANK',bm[bm['Resource Alt'].str[:1].isin(['S'])].Utilized.sum(),bm[bm['Resource Alt'].str[:1].isin(['S'])].working_hours.sum()],
                      ['Flexibles',bm[bm['Resource Alt'].str[:1].isin(['F'])].Utilized.sum(),bm[bm['Resource Alt'].str[:1].isin(['F'])].working_hours.sum()]],
                      columns = ['Resource Alt','Utilized','working_hours']) 
usage = pd.concat([usage,am],axis = 0)
usage = usage.rename(columns = {'Resource Alt': 'Resource Family','Utilized':'Usage','working_hours':'Available Hours'})
usage['Usage %'] = round(((usage['Usage']/usage['Available Hours'])*100),1)
usage.sort_values(by='Usage %', ascending=False)
fig_pipe2 = px.bar(usage,
             y=usage['Usage %'],
             x=usage['Resource Family'], title='CIP-CONSTRAINT USAGE DURATION',text=usage['Usage %'],
             labels= {'Usage %':'Resource Family'}
             ).update_xaxes(categoryorder="total descending")
fig_pipe2.update_traces(texttemplate="%{text:0f}",textfont=dict(color="white"),textposition = "outside",cliponaxis = False)
fig_pipe2.update_layout(xaxis_title='Resource Family',
                  yaxis_title='Resource Usage %')
fig_pipe2.show()
dynamic_outputs = plotly.io.to_json(fig_pipe2)
"""


# # Historical Schedules Table Code String #

# In[38]:


schedule_repository_table="""
from azure.cosmosdb.table.tableservice import TableService
from azure.storage.blob.sharedaccesssignature import BlobSharedAccessSignature
from datetime import datetime, timedelta
import os
import pandas as pd
import json
 
AZURE_ACC_NAME='mathcotakedastorage'
AZURE_PRIMARY_KEY='s94v9Q0+SXPPRagBKJUfB2WOA2Ts5mFPUb4sUg+C0tRzagJel6eYeeyrA7Dlz7s2mmPorbtOJQohLgjefo++Vw=='
blobSharedAccessSignature= BlobSharedAccessSignature(AZURE_ACC_NAME, AZURE_PRIMARY_KEY)
def get_sas_link(path_name):
    AZURE_CONTAINER, AZURE_BLOB=os.path.split(path_name)
    expiry=datetime.utcnow() +timedelta(hours=1)
    sasToken=blobSharedAccessSignature.generate_blob(AZURE_CONTAINER, AZURE_BLOB, expiry=expiry, permission="r")
    url='https://'+AZURE_ACC_NAME+'.blob.core.windows.net/'+AZURE_CONTAINER+'/'+AZURE_BLOB+'?'+sasToken
    return(url)
 
def format_date(s):
    d = pd.to_datetime(s,infer_datetime_format=True)
 
def get_historical_uploads():
    table_service= TableService(account_name='mathcotakedastorage', account_key='s94v9Q0+SXPPRagBKJUfB2WOA2Ts5mFPUb4sUg+C0tRzagJel6eYeeyrA7Dlz7s2mmPorbtOJQohLgjefo++Vw==')
    
    upto = datetime.utcnow()-timedelta(days=365)
    upto_str = upto.strftime('%Y-%m-%dT%H:%M:%S.%fZ')
    
    tasks=table_service.query_entities('schedule', f"PartitionKey eq 'schedule' and Timestamp gt datetime'{upto_str}'")
    rows=list(tasks)
    rows.sort(key=lambda x: x.Timestamp, reverse=True)
    table_data = []
    for index, el in enumerate(rows):
        row= [
            index+1,
            pd.to_datetime(el.pipeline_start).strftime('%d/%m/%Y %H:%M'),
            el.type,
            pd.to_datetime(el.start_at).strftime('%d/%m/%Y'),
            pd.to_datetime(el.end_at).strftime('%d/%m/%Y'),
            {"url": get_sas_link(el.result_file)}]
        table_data.append(row)
    table_headers= ["SI no", "Scheduler Run Date", "Schedule Type", "Schedule Start Date", "Schedule End Date", ""]
    result= {"assumptions": False, "table_data": table_data, "table_headers": table_headers,'suppress_download':True}
    return(result)


schedules_repository=get_historical_uploads()

dynamic_outputs = json.dumps(schedules_repository)
"""


# # Action Settings Code String #

# In[39]:


#BEGIN CUSTOM CODE BELOW...

#put your output in this response param for connecting to downstream widgets

#END CUSTOM CODE

action_generator1 = '''
import json
from datetime import datetime, timedelta
default_action1 = {
    "actions": [
      {
        "action_type": "download_latest_schedule",
        "component_type": "download_link",
        "params": {
          "is_icon": True,
          "text": "Download Schedule",
          "fetch_on_click": True
        },
        "position": {
          "portal": "tab_nav_bar"
        }
      },
      {
        "action_type": "generate_schedule",
        "component_type": "popup_form",
        "params": {
          "trigger_button": {
            "text": "Generate Schedule"
          },
          "dialog": {
            "title": "Generate Schedule"
          },
          "dialog_actions": [
            {
              "is_cancel": True,
              "text": "Cancel"
            },
            {
              "name": "generate",
              "text": "Generate",
              "variant": "contained"
            }
          ],
          "form_config": {
            "title": "",
            "fields": [
              {
                "id": 1,
                "name": "Active MC",
                "label": "Active Machine",
                "type": "select",
                "value": [
                  "MC1",
                  "MC2",
                  "MC3",
                  "MC4"
                ],
                "variant": "outlined",
                "multiple": True,
                "options": [
                  "MC1",
                  "MC2",
                  "MC3",
                  "MC4",
                  "MC5",
                  "MC6"
                ],
                "margin": "none",
                "fullWidth": True,
                "inputprops": {
                  "type": "select"
                },
                "placeholder": "Enter your Input",
                "grid": 6
              },
              {
                "id": 2,
                "type": "blank",
                "grid": 6
              },
              {
                "id": 3,
                "name": "Start Date",
                "suppressUTC":True,
                "label": "Schedule Start Date",
                "type": "datepicker",
                "variant": "outlined",
                "margin": "none",
                "value": datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0).isoformat(),
                "inputprops": {
                  "format": "DD/MM/yyyy",
                  "variant": "inline"
                },
                "placeholder": "Enter StartDate",
                "fullWidth": True,
                "grid": 6
              },
              {
                "id": 4,
                "name": "End Date",
                "suppressUTC":True,                
                "label": "Schedule End Date",
                "type": "datepicker",
                "variant": "outlined",
                "margin": "none",
                "value": (datetime.utcnow() + timedelta(days=6)).replace(hour=23, minute=59, second=59, microsecond=0).isoformat(),
                "inputprops": {
                  "format": "DD/MM/yyyy",
                  "variant": "inline"
                },
                "placeholder": "Enter EndDate",
                "fullWidth": True,
                "grid": 6
              },
              {
                "id": 5,
                "grid": 12,
                "name": "activeMachine",
                "type": "tabularForm",
                "value": [],
                "tableprops": {
                  "coldef": [
                    {
                      "headerName": "Active Machine",
                      "field": "activeMachine",
                      "cellEditor": "select",
                      "cellEditorParams": {
                        "variant": "outlined",
                        "options": [
                          "MC1",
                          "MC2",
                          "MC3",
                          "MC4",
                          "MC5",
                          "MC6"
                        ],
                        "fullWidth": True
                      },
                      "editable": True
                    },
                    {
                      "headerName": "Start Date",
                      "field": "startDate",
                      "cellEditor": "dateTime",
                      "editable": True,
                      "value": datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0).isoformat(),
                      "cellEditorParams": {
                        "suppressUTC":True,
                        "variant": "outlined",
                        "inputprops": {
                          "format": "DD/MM/YYYY LT"
                        }
                      }
                    },
                    {
                      "headerName": "End Date",
                      "field": "endDate",
                      "cellEditor": "dateTime",
                      "editable": True,
                      "value": (datetime.utcnow() + timedelta(days=6)).replace(hour=23, minute=59, second=59, microsecond=0).isoformat(),
                      "cellEditorParams": {
                        "suppressUTC":True,
                        "variant": "outlined",
                        "inputprops": {
                          "format": "DD/MM/YYYY LT"
                        }
                      }
                    }
                  ],
                  "gridOptions": {
                    "enableAddRow": True,
                    "enableInRowDelete": True,
                    "tableSize": "small",
                    "tableTitle": "CIP Downtime Details in between the Schedule Start & End Date",
                    "editorMode": True,
                    "addRowToTop": True,
                    "tableMaxHeight": "300px"
                  }
                }
              },
              {
                "id": 6,
                "name": "use_latest_schedule",
                "label": "Use Historical Schedule",
                "type": "checkbox",
                "value": False,
                "grid": 12,
                "noGutterBottom": True
              },
              {
                "id": 7,
                "name": "",
                "type": "label2",
                "fullWidth": True,
                "value": {
                  "fetch_on_load": True,
                },
                "actionType": "popup_schedule_period",
                "grid": 12,
                "noGutterTop": True
              },
              {
                "id": 9,
                "name": "",
                "type": "label",
                "value": "Input Data",
                "fullWidth": True,
                "InputLabelProps": {
                  "variant": "h4"
                },
                "underline": True,
                "grid": 12
              },
              {
                "id": 10,
                "name": "Schedule",
                "label": "Upload Schedules",
                "type": "upload",
                "value": "",
                "variant": "outlined",
                "margin": "none",
                "inputprops": {
                  "type": "file",
                  "multiple": True,
                  "accept": ""
                },
                "InputLabelProps": {
                  "disableAnimation": True,
                  "shrink": True
                },
                "placeholder": "Enter your Input",
                "grid": 12
              }
            ]
          }
        },
        "position": {
          "portal": "tab_nav_bar"
        }
      },
      {
        "action_type": "latest_schedule_params",
        "component_type": "text_list",
        "params": {
          "fetch_on_load": True
        },
        "position": {
          "portal": "screen_top_left"
        }
      }
    ]
  }
dynamic_outputs = json.dumps(default_action1)
'''


# reqired to handle the user actions on UI
# action_type would be set to global which is same as the default > actions > ... > action_type
# action_param would be set to global which will be passed from UI.
action_handler1 = '''
from __future__ import print_function
import json
from azure.storage.blob import BlockBlobService
from requests.utils import requote_uri
from azure.storage.blob.sharedaccesssignature import BlobSharedAccessSignature
from datetime import datetime, timedelta

# overview download starts -------------------------------------------------------------------

def get_overview_download_link():
  from azure.cosmosdb.table.tableservice import TableService
  from pathlib import Path
  from azure.storage.blob import BlockBlobService
  from io import StringIO
  from datetime import datetime, timedelta
  import os
  import pandas as pd
  import json
  from dateutil import parser
  from azure.storage.blob.sharedaccesssignature import BlobSharedAccessSignature
  from azure.cosmosdb.table.tableservice import TableService
  from pathlib import Path
  from azure.storage.blob import BlockBlobService
  from io import StringIO
  from datetime import datetime, timedelta
  import os
  import pandas as pd
  import json
  from dateutil import parser
  import zipfile
  from azure.storage.blob.sharedaccesssignature import BlobSharedAccessSignature
  # from azure.identity import DefaultAzureCredential
  key_value = 's94v9Q0+SXPPRagBKJUfB2WOA2Ts5mFPUb4sUg+C0tRzagJel6eYeeyrA7Dlz7s2mmPorbtOJQohLgjefo++Vw=='
  connection_string = 'DefaultEndpointsProtocol=https;AccountName=mathcotakedastorage;AccountKey=s94v9Q0+SXPPRagBKJUfB2WOA2Ts5mFPUb4sUg+C0tRzagJel6eYeeyrA7Dlz7s2mmPorbtOJQohLgjefo++Vw==;EndpointSuffix=core.windows.net'
  accountname = 'mathcotakedastorage'
  data_source = 'azure_blob_storage'
  AZURE_ACC_NAME = 'mathcotakedastorage'
  AZURE_PRIMARY_KEY = 's94v9Q0+SXPPRagBKJUfB2WOA2Ts5mFPUb4sUg+C0tRzagJel6eYeeyrA7Dlz7s2mmPorbtOJQohLgjefo++Vw=='
  AZURE_CONTAINER = 'input'
  #AZURE_BLOB='output_zip.zip'
  expiry= datetime.utcnow() + timedelta(hours=1)
  def get_ingested_data(file_path, datasource_type, connection_uri, container_name):
      path = Path(file_path)
      block_blob_service = BlockBlobService(connection_string=connection_uri)
      blob_data = block_blob_service.get_blob_to_text(container_name=container_name, 
                                                      blob_name=file_path)
      ingested_df = pd.read_csv(StringIO(blob_data.content))
      return ingested_df
  block_blob_service = BlockBlobService(connection_string=connection_string)
  blob_data = block_blob_service.get_blob_to_text(container_name='input1', blob_name='Scheduler_Input/schedule_input.json')
  f = pd.read_json(StringIO(blob_data.content))
  #Start Date UTC and round handling
  Start_Date = f['data']['Start Date'][0:19]
  Start_Date=parser.parse(datetime.strptime(Start_Date ,'%Y-%m-%dT%H:%M:%S').strftime('%Y-%m-%dT%H:%M:%S'))
  Start_Date=datetime.strftime(Start_Date ,'%Y-%m-%d')
  Start_Date=parser.parse(datetime.strptime(Start_Date ,'%Y-%m-%d').strftime('%Y-%m-%d'))
  #End Date UTC and round handling
  End_Date = f['data']['End Date'][0:19]
  End_Date=parser.parse(datetime.strptime(End_Date ,'%Y-%m-%dT%H:%M:%S').strftime('%Y-%m-%dT%H:%M:%S'))
  End_Date=datetime.strftime(End_Date ,'%Y-%m-%d')
  End_Date=parser.parse(datetime.strptime(End_Date ,'%Y-%m-%d').strftime('%Y-%m-%d'))+timedelta(hours=23.99)
  # All the metadata is stored into table
  table_service= TableService(account_name='mathcotakedastorage', account_key='s94v9Q0+SXPPRagBKJUfB2WOA2Ts5mFPUb4sUg+C0tRzagJel6eYeeyrA7Dlz7s2mmPorbtOJQohLgjefo++Vw==')
  tasks=table_service.query_entities('schedule', f"PartitionKey eq 'schedule' and type eq 'Planned'")
  table=pd.DataFrame(tasks)
  # Converting start date and end date in metadata df to datetime
  table['start_at'] = pd.to_datetime(table['start_at'],utc=True).dt.date
  table['end_at'] = pd.to_datetime(table['end_at'],utc=True).dt.date
  table['start_at']=pd.to_datetime(table['start_at'])
  table['end_at']=pd.to_datetime(table['end_at'])
  # Query df such that we get file path of data with schedule date between the given date on UI
  # path eg = 'Src1/300921_Schedule/2021-09-04T04-10/output/20210930_2021-09-04T04-10_Optimized_Schedule_Output_(R).csv'
  if f['data']['use_latest_schedule']==True:
      try:
          queried_table = table[(table['start_at'] <= datetime.strftime(Start_Date,'%Y-%m-%d')) & (table['end_at'] >= datetime.strftime(End_Date ,'%Y-%m-%d'))]
          queried_table = queried_table.tail(1)
          queried_table.reset_index(inplace = True)
          queried_path = queried_table['result_file'][0]
          # queried_path = queried_path[6:76] + '_Optimized_Schedule_Output_(R).csv'
          condition1=[(queried_table['start_at'] == datetime.strftime(Start_Date,'%Y-%m-%d')) & (queried_table['end_at'] == datetime.strftime(End_Date ,'%Y-%m-%d'))][0]
          condition2=[(queried_table['start_at'] < datetime.strftime(Start_Date,'%Y-%m-%d')) | (queried_table['end_at'] > datetime.strftime(End_Date ,'%Y-%m-%d'))][0]
          if condition1[0]==True:
              queried_path_condition1 = queried_path[6:76] + '_output_zip.zip'
              blobSharedAccessSignature = BlobSharedAccessSignature(AZURE_ACC_NAME, AZURE_PRIMARY_KEY)
              sasToken = blobSharedAccessSignature.generate_blob('input', queried_path_condition1, expiry=expiry, permission="r")
              url = 'https://'+AZURE_ACC_NAME+'.blob.core.windows.net/'+AZURE_CONTAINER+'/'+queried_path_condition1+'?'+sasToken
      #schedule = get_ingested_data(file_path=queried_path,datasource_type=data_source,connection_uri=connection_string,container_name='input')
          if condition2[0]==True:
              queried_path_condition2 = queried_path[6:76] + '_Optimized_Schedule_Output_(R).csv'
              temp_schedule = get_ingested_data(file_path=queried_path_condition2,datasource_type=data_source,connection_uri=connection_string,container_name='input')
              temp_schedule['Usage_Start'] = pd.to_datetime(temp_schedule['Usage_Start'])
              temp_schedule['Usage_End'] = pd.to_datetime(temp_schedule['Usage_End'])
              temp_schedule.sort_values(by='Usage_Start')
              schedule = temp_schedule[(temp_schedule['Usage_Start']>=Start_Date) & (temp_schedule['Usage_End']<=End_Date)]
              
              schedule.to_csv('schedule.csv')
              infor_schedule = schedule[['Resource','LOT','Usage_Start','Usage_End','Clean_Start_Time','Clean_End_Time','Cleaning Duration','End DHT Time','End CHT Time','MC Group','Constraint','Cleaning Type']]
              infor_schedule.to_csv('infor_schedule.csv')
              queried_path_maint = queried_path[6:76] + '_Maintenance-Infor(OFD).csv'
              
              temp_maint_schedule = get_ingested_data(file_path=queried_path_maint,datasource_type=data_source,connection_uri=connection_string,container_name='input')
              temp_maint_schedule['Usage_Start'] = pd.to_datetime(temp_maint_schedule['Usage_Start'])
              temp_maint_schedule['Usage_End'] = pd.to_datetime(temp_maint_schedule['Usage_End'])
              temp_maint_schedule.sort_values(by='Usage_Start')
              maint_schedule = temp_maint_schedule[(temp_maint_schedule['Usage_Start']>=Start_Date) & (temp_maint_schedule['Usage_End']<=End_Date)]
              maint_schedule.to_csv('maint_schedule.csv')
              zf = zipfile.ZipFile('selected_schedule.zip', mode='w')
              zf.write('schedule.csv',arcname ='schedule.csv')
              zf.write('infor_schedule.csv',arcname ='infor_schedule.csv')
              zf.write('maint_schedule.csv',arcname ='maint_schedule.csv')
              zf.close()
              block_blob_service = BlockBlobService(connection_string=connection_string)
              blob_data = block_blob_service.create_blob_from_path(container_name='input', blob_name='use_old_data/selected_schedule.zip', file_path="selected_schedule.zip")
              blobSharedAccessSignature = BlobSharedAccessSignature(AZURE_ACC_NAME, AZURE_PRIMARY_KEY)
              sasToken = blobSharedAccessSignature.generate_blob('input', 'use_old_data/selected_schedule.zip', expiry=expiry, permission="r")
              url = 'https://'+AZURE_ACC_NAME+'.blob.core.windows.net/'+AZURE_CONTAINER+'/'+'use_old_data/selected_schedule.zip'+'?'+sasToken
      except:
          temp_schedule = get_ingested_data(file_path='merged_output_repository/merged_outputs.csv',datasource_type=data_source,connection_uri=connection_string,container_name='input')
          temp_schedule['Usage_Start'] = pd.to_datetime(temp_schedule['Usage_Start'])
          temp_schedule['Usage_End'] = pd.to_datetime(temp_schedule['Usage_End'])
          schedule = temp_schedule[(temp_schedule['Usage_Start']>=Start_Date) & (temp_schedule['Usage_End']<=End_Date)]
          schedule = schedule[['Resource','LOT','Usage_Start','Usage_End','Clean_Start_Time','Clean_End_Time','Cleaning Duration','End DHT Time','End CHT Time','MC Group','Constraint','Cleaning Type']]
          block_blob_service = BlockBlobService(connection_string=connection_string)
          blob_data = block_blob_service.create_blob_from_text(container_name='input', blob_name='use_old_data/Historical_Schedule.csv', text = str(schedule.to_csv()))
          blobSharedAccessSignature = BlobSharedAccessSignature(AZURE_ACC_NAME, AZURE_PRIMARY_KEY)
          sasToken = blobSharedAccessSignature.generate_blob('input', 'use_old_data/Historical_Schedule.csv', expiry=expiry, permission="r")
          url = 'https://'+AZURE_ACC_NAME+'.blob.core.windows.net/'+AZURE_CONTAINER+'/'+'use_old_data/Historical_Schedule.csv'+'?'+sasToken
  else:
      blobSharedAccessSignature = BlobSharedAccessSignature(AZURE_ACC_NAME, AZURE_PRIMARY_KEY)
      sasToken = blobSharedAccessSignature.generate_blob('input', 'merged_output_repository/output_zip.zip', expiry=expiry, permission="r")
      url = 'https://'+AZURE_ACC_NAME+'.blob.core.windows.net/'+AZURE_CONTAINER+'/'+'merged_output_repository/output_zip.zip'+'?'+sasToken
  obj = {"message": "Success", "url": url}
  return json.dumps(obj)
# overview download end -------------------------------------------------------------------


# generate schedule starts --------------------------------------------------

# Python libs


CONNECTION_STRING = 'DefaultEndpointsProtocol=https;AccountName=mathcotakedastorage;AccountKey=s94v9Q0+SXPPRagBKJUfB2WOA2Ts5mFPUb4sUg+C0tRzagJel6eYeeyrA7Dlz7s2mmPorbtOJQohLgjefo++Vw==;EndpointSuffix=core.windows.net'

def folder_name(date):
    d = datetime.strptime(date[0:19] ,'%Y-%m-%dT%H:%M:%S')
    date_str = d.strftime('%d%m%y')
    folder_name_with_date = date_str+'_Schedule'
    return(folder_name_with_date)

def generate_json(action_param):
    if (action_param['data']['use_latest_schedule']==False and len(action_param['data']['Schedule'])==0):
        return json.dumps({"message": "You have not provided any inputs.Please upload latest schedule or use historical schedule","severity":"error"})
    # File path where files are uploaded on blob ('input1/Scheduler_Input/' + 'ddmmyy_Schedule')
    file_path = 'input1/Scheduler_Input'
    blob_service = BlockBlobService(connection_string=CONNECTION_STRING)
    input_schedule_json = json.dumps(action_param)
    blob_service.create_blob_from_text(file_path , "schedule_input.json", input_schedule_json)
    return input_schedule_json

def generate_schedule(action_param):
  # Gets start date from json file. This is converted to folder name: 'ddmmyy_Schedule' using folder_name() function
  if action_param['data']['use_latest_schedule']==False and len(action_param['data']['Schedule'])!=0 and action_param['data']['Start Date']<action_param['data']['End Date']:
      date = action_param['data']['Start Date']
      raw_data_folder_name = folder_name(date)
      # File path where files are uploaded on blob ('input1/Scheduler_Input/' + 'ddmmyy_Schedule')
      file_path = 'raw-data/Scheduler_Input/' + raw_data_folder_name
      blob_service = BlockBlobService(connection_string=CONNECTION_STRING)
      for ele in action_param['data']['Schedule']:
          url = ele['path']
          file_name = ele['filename']
          #The below 4 lines take urls and copy them on Takeda blob
          source_blob = requote_uri(url)
          blob_service.copy_blob(file_path , file_name, source_blob)
      input_schedule_json = json.dumps(action_param)
      blob_service.create_blob_from_text(file_path , "schedule_input.json", input_schedule_json)
      generate_json(action_param)
      return json.dumps({"message": "Your Schedule is initiated successfully. It would take 30 to 35 minutes to generate."})
  elif action_param['data']['use_latest_schedule']==True and action_param['data']['activeMachine']==[] and action_param['data']['Active MC']==['MC1', 'MC2', 'MC3', 'MC4'] and len(action_param['data']['Schedule'])==0 and action_param['data']['Start Date']<action_param['data']['End Date']:
      generate_json(action_param)
      return json.dumps({"message": "Your Schedule is filtered successfully. Please refresh the page to view the results."})
  elif action_param['data']['Start Date']>action_param['data']['End Date']:
      return json.dumps({"message": "Start date is greater than the end date"})
  else:
      return json.dumps({"message": "ERROR : INVALID INPUTS"})



# generate schedule ends --------------------------------------------------

    
# get schedule inputs starts ----------------------------------------------


def get_date(date):
    d = datetime.strptime(date[0:19],'%Y-%m-%dT%H:%M:%S')
    return d.strftime('%d/%m/%y')

def get_schedule_inputs():
    CONNECTION_STRING = 'DefaultEndpointsProtocol=https;AccountName=mathcotakedastorage;AccountKey=s94v9Q0+SXPPRagBKJUfB2WOA2Ts5mFPUb4sUg+C0tRzagJel6eYeeyrA7Dlz7s2mmPorbtOJQohLgjefo++Vw==;EndpointSuffix=core.windows.net'
    blob_service = BlockBlobService(connection_string=CONNECTION_STRING)

    copied_blob = blob_service.get_blob_to_text('input/merged_output_repository' , 'schedule_input.json')
    input_schedule = json.loads(copied_blob.content)
    obj = {
        "list": [
            {
                "text": "Schedule Inputs:",
                "style": {
                    "fontWeight": 600
                }
            },
            {
                "text": "Active Machine:"
            },
            {
                "text": ", ".join(input_schedule.get("data")["Active MC"]),
                "color": "contrast"
            },
            {
                "text": "|"
            },
            {
                "text": "Dates:"
            },
            {
                "text": f"{get_date( input_schedule.get('data')['Start Date'])} to {get_date( input_schedule.get('data')['End Date'])}",
                "color": "contrast"
            }
        ]
    }
    return json.dumps(obj)

# get schedule inputs ends ----------------------------------------------

# get_popup_schedule_period starts ---------------------------------------------

def get_popup_schedule_period():
    CONNECTION_STRING = 'DefaultEndpointsProtocol=https;AccountName=mathcotakedastorage;AccountKey=s94v9Q0+SXPPRagBKJUfB2WOA2Ts5mFPUb4sUg+C0tRzagJel6eYeeyrA7Dlz7s2mmPorbtOJQohLgjefo++Vw==;EndpointSuffix=core.windows.net'
    blob_service = BlockBlobService(connection_string=CONNECTION_STRING)

    copied_blob = blob_service.get_blob_to_text('input/merged_output_repository' , 'schedule_input.json')
    input_schedule = json.loads(copied_blob.content)
    obj = {
        "list": [
            {
                "text": "Latest Schedule Run:"
            },
            {
                "text": f"{get_date( input_schedule.get('data')['Start Date'])} to {get_date( input_schedule.get('data')['End Date'])}",
                "color": "contrast"
            }
        ]
    }
    return json.dumps(obj)

# get_popup_schedule_period ends ---------------------------------------------

dynamic_outputs = None

if action_type == "download_latest_schedule":
    dynamic_outputs = get_overview_download_link()
if action_type == "generate_schedule":
    dynamic_outputs = generate_schedule(action_param)
if action_type == "latest_schedule_params":
    dynamic_outputs = get_schedule_inputs()
if action_type == "popup_schedule_period":
    dynamic_outputs = get_popup_schedule_period()
'''

action_handler3 = '''
from __future__ import print_function
import json
from azure.storage.blob import BlockBlobService
from requests.utils import requote_uri
from azure.storage.blob.sharedaccesssignature import BlobSharedAccessSignature
from datetime import datetime, timedelta

# Historical schedule upload starts --------------------------------------------------

# Python libs


CONNECTION_STRING = 'DefaultEndpointsProtocol=https;AccountName=mathcotakedastorage;AccountKey=s94v9Q0+SXPPRagBKJUfB2WOA2Ts5mFPUb4sUg+C0tRzagJel6eYeeyrA7Dlz7s2mmPorbtOJQohLgjefo++Vw==;EndpointSuffix=core.windows.net'

def folder_name(date):
    d = datetime.strptime(date ,'%Y-%m-%dT%H:%M:%S.%fZ')
    date_str = d.strftime('%d%m%y')
    folder_name_with_date = date_str+'_Schedule'
    return(folder_name_with_date)

def generate_schedule(action_param):
    # Gets start date from json file. This is converted to folder name: 'ddmmyy_Schedule' using folder_name() function
    date = action_param['data']['Start Date']
    raw_data_folder_name = folder_name(date)
    # File path where files are uploaded on blob ('input1/Scheduler_Input/' + 'ddmmyy_Schedule')
    file_path = 'actual-output/Actual_Input/' + raw_data_folder_name

    blob_service = BlockBlobService(connection_string=CONNECTION_STRING)
    
    for ele in action_param['data']['Schedule']:
        url = ele['path']
        file_name = ele['filename']
        
        #The below 4 lines take urls and copy them on Takeda blob
        source_blob = requote_uri(url)
        blob_service.copy_blob(file_path , file_name, source_blob)

    input_schedule_json = json.dumps(action_param)
    blob_service.create_blob_from_text(file_path , "actual_input.json", input_schedule_json)
    return json.dumps({"message": "Your Schedule is initiated successfully. It would take 30 to 35 minutes to generate."})


# Historical schedule upload schedule ends --------------------------------------------------

dynamic_outputs = None

if action_type == "upload_actuals":
    dynamic_outputs = generate_schedule(action_param)

'''


action_generator3 = '''
import json
from datetime import datetime, timedelta
default_action3 = {
    "actions": [
      {
        "action_type": "upload_actuals",
        "component_type": "popup_form",
        "params": {
          "trigger_button": {
            "text": "Upload Actual Schedules"
          },
          "dialog": {
            "title": "Upload Document"
          },
          "dialog_actions": [
            {
              "is_cancel": True,
              "text": "Cancel"
            },
            {
              "name": "generate",
              "text": "Generate",
              "variant": "contained"
            }
          ],
          "form_config": {
            "title": "",
            "fields": [
              {
                "id": 3,
                "name": "Start Date",
                "suppressUTC": True,
                "label": "Start Date",
                "type": "datepicker",
                "variant": "outlined",
                "margin": "none",
                "value": (datetime.utcnow() - timedelta(days=6)).replace(hour=0, minute=0, second=0, microsecond=0).isoformat(),
                "inputprops": {
                  "format": "DD/MM/yyyy",
                  "variant": "inline"
                },
                "InputLabelProps": {
                  "disableAnimation": True,
                  "shrink": True
                },
                "placeholder": "Enter StartDate",
                "helperText": "Invalid Input",
                "fullWidth": True,
                "grid": 6
              },
              {
                "id": 4,
                "name": "End Date",
                "suppressUTC": True,
                "label": "End Date",
                "type": "datepicker",
                "variant": "outlined",
                "margin": "none",
                "value": datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0).isoformat(),
                "inputprops": {
                  "format": "DD/MM/yyyy",
                  "variant": "inline"
                },
                "InputLabelProps": {
                  "disableAnimation": True,
                  "shrink": True
                },
                "placeholder": "Enter EndDate",
                "helperText": "Invalid Input",
                "fullWidth": True,
                "grid": 6
              },
              {
                "id": 5,
                "label": "Input Data",
                "type": "label",
                "value": "Input Data",
                "variant": "outlined",
                "margin": "none",
                "inputprops": {
                  "elevation": 0,
                  "variant": "outlined"
                },
                "InputLabelProps": {
                  "variant": "h4",
                  "align": "inherit",
                  "display": "initial"
                },
                "placeholder": "Input Data",
                "helperText": "Input Data",
                "fullWidth": True,
                "grid": 12
              },
              {
                "id": 9,
                "name": "Schedule",
                "label": "Schedule",
                "type": "upload",
                "value": "",
                "variant": "outlined",
                "margin": "none",
                "inputprops": {
                  "type": "file",
                  "multiple": True,
                  "accept": ""
                },
                "InputLabelProps": {
                  "disableAnimation": True,
                  "shrink": True
                },
                "placeholder": "Enter your Input",
                "grid": 12
              }
            ]
          }
        },
        "position": {
          "portal": "screen_top_right"
        }
      }
    ]
  }
dynamic_outputs = json.dumps(default_action3)
'''


# # Deploy #

# In[40]:


dynamic_visual_results={
    "GANTT Chart CIP":code_string_CIP,
    "GANTT Chart Resource":code_string_Resource,
}

dynamic_filter_codes={
    "Resource utilization":filter_code_string_Resource,
    "CIP utilization":filter_code_string_CIP
}

dynamic_filter_codes_metrics={
    "CIP utilization":filter_code_string_CIP_Metrics,
    "Resource utilization":filter_code_string_Resource_Metrics
}


cip_summary_results={
    "CIP Utilization %":kpi_cip_utilization,
    "# Re-Cleanings & Pre-Cleanings":kpi_recleanings,
    "# DHT Violations":kpi_dht_violations,
    "# CHT Violations":kpi_cht_violations,
    "Distribution of Cleaning Cycles":kpi_cleaning_cycles,
    "# RE-CLEANINGS DUE TO PREVIOUS SCHEDULE":kpi_preclean,
    "# RE-CLEANINGS DUE TO CHT VIOLATIONS":kpi_reclean,
    "Average CIP Utilization %":kpi_cip_util,
    "Average Group Constraint Utilization %":kpi_pipe_util,
    "Frequency of Utilization":kpi_res_util,
    "Average Resource Usage":kpi_res_usage,
    "# RE-CLEANINGS DUE TO PREVIOUS SCHEDULE - Resources ":kpi_res_preclean,
    "# RE-CLEANINGS DUE TO CHT VIOLATIONS - Resources ":kpi_res_reclean,
}

cip_metric_results={
}

res_metric_results={
    "# Re-Cleanings & Pre-Cleanings":kpi_res_reclean_overall,
    "# DHT Violations":kpi_res_dht_overall,
    "# CHT Violations":kpi_res_cht_overall,
    "Average Usage %":kpi_res_usage_overall
}

res_visual_results={
    "Recleaning Trend":vis_reclean_overall,
    "CHT Violations Trend":cht_violation_trend,
    "DHT Violation Trend":dht_violation_trend,
    "Resource Usage %":visual_usage_family_trend

}


cip_visual_results={
    "Average CIP Utilization":kpi_cip_utilization_overall,
    "Total CIP Usage Duration (Hours)":kpi_cip_duration_overall,
    "Total Cleaning Cycles":kpi_clean_cycles_kpi,
    "Additional Cleanings Possible":kpi_possible_cycles_kpi,
    "CIP Usage Duration":visual_cip_dur_overall,
    "CIP Usage Distribution":visual_cip_usage_dist,
    "Avg Time Gap between consecutive CIP":visual_arrival_time_cip,
    "CIP Utilisation Weekly/Monthly Trend":visual_periodic_util,
    "CIP Usage Duration Weekly/Monthly Trend":visual_periodic_duration
}

schedules_repository={
    "Actual & Planned Schedules":schedule_repository_table
}

response_0 = {
    "action_setting_overview_cip_summary": {
        "default": None,
        "action_generator": action_generator1,
        "action_handler": action_handler1
    },
    "action_setting_overview_resource_summary": {
        "default": None,
        "action_generator": action_generator1,
        "action_handler": action_handler1
    },
    "action_setting_historical_schedule": {
        "default":None,
        "action_generator": action_generator3,
        "action_handler": action_handler3
    },
}


results_json.append({
    'type':'Compare_1',
    'name': 'Compare_2',
    'component':'Compare_3',
    'actions': response_0
})



results_json.append({
    'type':'Overview',
    'name':'Utilization Timeline',
    'component':'Utilization',
    'dynamic_visual_results':dynamic_visual_results,
    'dynamic_code_filters':dynamic_filter_codes,
    'metrics':False
})
results_json.append({
    'type':'Overview',
    'name':'Summary',
    'component':'Utilization',
    'dynamic_visual_results':cip_summary_results,
    'metrics':False
})
results_json.append({
    'type':'Overview',
    'name':'Metrics',
    'component':'CIP',
    'dynamic_metrics_results':cip_metric_results,
    'dynamic_code_filters':dynamic_filter_codes_metrics,
    'dynamic_visual_results':cip_visual_results,
    'metrics':False
})
results_json.append({
    'type':'Overview',
    'name':'Metrics',
    'component':'Resource',
    'dynamic_metrics_results':res_metric_results,
    'dynamic_visual_results':res_visual_results,
    'metrics':False
})
results_json.append({
    'type':'Overview',
    'name':'Summary',
    'component':'Utilization',
    'dynamic_visual_results':cip_summary_results,
    'metrics':False
})
results_json.append({
    'type':'Overview',
    'name':'Repository',
    'component':'Schedules',
    'dynamic_visual_results':schedules_repository,
    'metrics':False
})


# ### Please save and checkpoint notebook before submitting params

# In[41]:



currentNotebook = 'Utilization_Timeline_202108200834_Prod.ipynb'

get_ipython().system('jupyter nbconvert --to script {currentNotebook}')


# In[42]:


#Production Url
utils.submit_config_params(url='https://codex-api-takeda.azurewebsites.net/codex-api/projects/upload-config-params/C86A1EF6CE4F103C106E76B446AF1170C6C748F55C62EA7DAE4CDFF76D0C7CC0', nb_name=currentNotebook, results=results_json, codex_tags=codex_tags, args={})

