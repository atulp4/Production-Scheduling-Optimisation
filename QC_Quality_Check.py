def get_data():
    get_lib()
    global dataset
    #Importing the Output file for QC Check
    #dataset=pd.read_excel(r'C:\Users\AtulPoddar\OneDrive - TheMathCompany Private Limited\Documents\Takeda\20210930_2021-08-31T10-50_Optimized_Schedule_Output_(R).xlsx',engine='openpyxl')
    dataset = pd.read_excel(r'C:\Users\AtulPoddar\OneDrive - TheMathCompany Private Limited\Documents\Takeda\Modelling_Test_Results\20211017_Takeda_Priority Production(With CIP Freezed)_v11.xlsx',engine='openpyxl')
    dataset['Cleaning Duration'] = dataset['Cleaning Duration'] / 60
    dataset.Usage_Start= pd.to_datetime(dataset.Usage_Start, format = '%Y/%m/%d %H:%M:%S')
    dataset.Usage_End= pd.to_datetime(dataset.Usage_End, format = '%Y/%m/%d %H:%M:%S')
    dataset['Next Usage']= pd.to_datetime(dataset['Next Usage'], format = '%Y/%m/%d %H:%M:%S')
    dataset['End DHT Time']= pd.to_datetime(dataset['End DHT Time'], format = '%Y/%m/%d %H:%M:%S')
    dataset['End CHT Time']= pd.to_datetime(dataset['End CHT Time'], format = '%Y/%m/%d %H:%M:%S')
    dataset.Clean_Start_Time= pd.to_datetime(dataset.Clean_Start_Time, format = '%Y/%m/%d %H:%M:%S')
    dataset.Clean_End_Time= pd.to_datetime(dataset.Clean_End_Time, format = '%Y/%m/%d %H:%M:%S')
    return dataset
#Quality Check without Maintenance
def run_quality_check():
    dataset=get_data()
    cl = check_cleaning(dataset)
    if cl is None:
        cl = ['All cleanings are fulfilled.There is no resource where cleaning was missed']
        cl1 = pd.DataFrame(cl,columns = [''])
    else:
        cl1=cl
    #Added constraint to exclude parallel cleanings so that it doesn't show in overlap check
    df1 = dataset.groupby(['Parallel Clean Flag','MC Group','Constraint'],as_index = False).agg({
                                                                            'Cleaning Duration':'max',
                                                                            'Clean_Start_Time':'min',
                                                                            'Clean_End_Time':'max'
                                                                            })
    df1 = df1.drop(['Parallel Clean Flag'],axis=1)
    new_df = dataset[dataset['Parallel Clean Flag'].isna()]
    map_columns = ['MC Group', 'Constraint', 'Cleaning Duration','Clean_Start_Time', 'Clean_End_Time']
    df_mc_group = (new_df[map_columns]).dropna()
    df_mc_group = pd.concat([df_mc_group, df1], axis=0)
    mf = cleaning_availablity(df = df_mc_group.copy(), sort_by_col = 'Clean_Start_Time', for_col = 'Constraint', sub_col = 'Clean_End_Time')
    mcg = cleaning_availablity(df = df_mc_group.copy(), sort_by_col = 'Clean_Start_Time', for_col = 'MC Group', sub_col = 'Clean_End_Time')
    o1 = Overlap_at_Constraint_level_QC(mf)
    o2 = Overlap_at_MC_level_QC(mcg)
    #m = check_dht_1(dataset)
    m = check_dht(dataset)
    if m is None:
        m = ['DHT is adhered for all cleaning Cycles']
        dht = pd.DataFrame(m,columns = [''])
    else:
        dht=m
    n= check_cht1(dataset)
    if n is None:
        n = ['CHT is adhered for all cleaning Cycles']
        cht = pd.DataFrame(n,columns = [''])
    else:
        cht=n
    # cht= check_cht(dataset)
    pre = pre_re_cleaning(dataset)
    var = variance_Cleaning(dataset)
    flex_map = flexibles_map(dataset)
    if flex_map is None:
        flex_map = ['All flexibles have been mapped correctly']
        fm = pd.DataFrame(flex_map,columns = [''])
    else:
        fm=flex_map
    pcm1 = parallel_cleaning_map(dataset)
    if pcm1 is None:
        pcm1 = ['All Parallel Cleanings have been mapped correctly']
        pcm = pd.DataFrame(pcm1,columns = [''])
    else:
        pcm=pcm1
    #Saving all
    writer = pd.ExcelWriter('Quality_Check_20211017_Takeda_Priority Production(With CIP Freezed)_v11.xlsx', engine='xlsxwriter')
    cl1.to_excel(writer, sheet_name='Cleanings Missed')
    o1.to_excel(writer, sheet_name='Overlap Constraint')
    o2.to_excel(writer, sheet_name='Overlap CIP')
    dht.to_excel(writer, sheet_name='DHT')
    cht.to_excel(writer, sheet_name='CHT')
    pre.to_excel(writer, sheet_name='Precleans')
    var.to_excel(writer, sheet_name='Variance of Cleaning')
    fm.to_excel(writer, sheet_name='Flexibles')
    pcm.to_excel(writer, sheet_name='Parallel Cleaning')
    writer.save()
#Quality Check with Maintenance
def run_quality_check_M(dataset):
    cl = check_cleaning(dataset)
    if cl is None:
        cl = ['All cleanings are fulfilled.There is no resource where cleaning was missed']
        cl1 = pd.DataFrame(cl,columns = [''])
    else:
        cl1=cl
    map_columns = ['MC Group', 'Constraint', 'Cleaning Duration','Clean_Start_Time', 'Clean_End_Time']
    df_mc_group = dataset[map_columns]
    mf = cleaning_availablity(df = df_mc_group.copy(), sort_by_col = 'Clean_Start_Time', for_col = 'Constraint', sub_col = 'Clean_End_Time')
    mcg = cleaning_availablity(df = df_mc_group.copy(), sort_by_col = 'Clean_Start_Time', for_col = 'MC Group', sub_col = 'Clean_End_Time')
    o1 = Overlap_at_Constraint_level_QC(mf)
    o2 = Overlap_at_MC_level_QC(mcg)
    m = check_dht(dataset)
    if m is None:
        m = ['DHT is adhered for all cleaning Cycles']
        dht = pd.DataFrame(m,columns = [''])
    else:
        dht=m
    cht= check_cht1(dataset)
    maint = check_maint_recleaning(dataset)
    if maint is None:
        z = ['Cleaning conducted after all Maintenance']
        maint_c = pd.DataFrame(z,columns = [''])
    else:
        maint_c = maint
    pre = pre_re_cleaning(dataset)
    var = variance_Cleaning(dataset)
    flex_map = flexibles_map(dataset)
    if flex_map is None:
        flex_map = ['All flexibles have been mapped correctly']
        fm = pd.DataFrame(flex_map,columns = [''])
    else:
        fm=flex_map
    pcm1 = parallel_cleaning_map(dataset)
    if pcm1 is None:
        pcm1 = ['All Parallel Cleanings have been mapped correctly']
        pcm = pd.DataFrame(pcm1,columns = [''])
    else:
        pcm=pcm1
    #Saving all
    writer = pd.ExcelWriter('Quality_Check_Takeda_Priority Production planning_Outcomev18(With LOT).xlsx', engine='xlsxwriter')
    cl1.to_excel(writer, sheet_name='Cleanings Missed')
    o1.to_excel(writer, sheet_name='Overlap Constraint')
    o2.to_excel(writer, sheet_name='Overlap CIP')
    dht.to_excel(writer, sheet_name='DHT')
    cht.to_excel(writer, sheet_name='CHT')
    maint_c.to_excel(writer, sheet_name='Maintenance')
    pre.to_excel(writer, sheet_name='Pre/Re Cleanings')
    var.to_excel(writer, sheet_name='Variance of Cleaning')
    fm.to_excel(writer, sheet_name='Flexibles')
    pcm.to_excel(writer, sheet_name='Parallel Cleaning')
    writer.save()

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
###### 1. Checking if all resources have cleaned or not
def check_cleaning(item):
    if item['Clean_Start_Time'].isnull().any()==False:
        x = ['All cleanings are fulfilled.There is no resource where cleaning was missed']
        cc = pd.DataFrame(x,columns = [''])
        print(cc)
    else:
        y = ['Cleanings missed']
        cc = pd.DataFrame(y,columns = [''])
        cc.set_index("", inplace=True)
        y = item[item['Clean_Start_Time'].isna()]
        result = pd.concat([cc,y])
        return result
###### 2. Check for overlap in cleaning schedules
#Output of this function used for input in overlap qc
def cleaning_availablity(df = None, sort_by_col = '', for_col = '', sub_col = 'new_clean_end'):
    temp_df= pd.DataFrame()
    for val in df[for_col].unique():
        data_temp = df[df[for_col] == val]
        data_temp = data_temp.sort_values(sort_by_col)
        data_temp['idle_till_time'] = data_temp[sort_by_col].shift(-1)
        temp_df = pd.concat([temp_df, data_temp])
    temp_df = temp_df.sort_values([for_col, sort_by_col]).reset_index(drop = True)
    temp_df['Available_Time'] = (temp_df['idle_till_time'] - temp_df[sub_col])/(np.timedelta64(1, 's')*3600)
    temp_df['Last_pos_Clean'] = temp_df['idle_till_time'] - (pd.to_timedelta(temp_df['Cleaning Duration'], unit = 'h') +
                                                             pd.Timedelta(minutes=10))
    return temp_df
def Overlap_at_Constraint_level_QC(dataset):
    global x
    dataset['Available_Time'] = dataset['Available_Time'].fillna(0)
    dataset['Overlap'] = np.where(dataset['Available_Time'] < 0,1,0)
    x = dataset.groupby(['MC Group','Constraint'], as_index = False).Overlap.sum()
    return x
def Overlap_at_MC_level_QC(dataset):
    global mcg
    dataset['Available_Time'] = dataset['Available_Time'].fillna(0)
    dataset['Overlap'] = np.where(dataset['Available_Time'] < 0,1,0)
    y = dataset.groupby(['MC Group'], as_index = False).Overlap.sum()
    return y
###### 3. Checking DHT constraint
def check_dht_1(dataframe):
    if ((dataframe['DHT Violation Flag']==0).all() == True):
        n = ['DHT is adhered for all cleaning Cycles']
        dht = pd.DataFrame(n,columns = [''])
        print(dht)
    else:
        x1 = dataframe[dataframe['DHT Violation Flag'] > 0]
        y = ['DHT Violations']
        dht_violated = pd.DataFrame(y,columns = [''])
        dht_violated.set_index("", inplace=True)
        result = pd.concat([dht_violated,x1])
        return result
def check_dht(dataframe):
    if ((dataframe['End DHT Time'] > dataframe['Clean_Start_Time']).all() == True):
        n = ['DHT is adhered for all cleaning Cycles']
        dht = pd.DataFrame(n,columns = [''])
        print(dht)
    else:
        x = dataframe['End DHT Time'] < dataframe['Clean_Start_Time']
        y = ['DHT Violations']
        dht_violated = pd.DataFrame(y,columns = [''])
        dht_violated.set_index("", inplace=True)
        result = pd.concat([dht_violated,x])
        return result
###### 4. Checking CHT constraint
def check_cht1(item):
    if ((item['CHT Violation Flag']==0).all() == True):
        z = ['CHT is adhered for all cleaning Cycles']
        cht = pd.DataFrame(x,columns = [''])
        print(cht)
    else:
        x2 = item[item['CHT Violation Flag'] >0]
        y = ['CHT Violations']
        cht_violated = pd.DataFrame(y,columns = [''])
        cht_violated.set_index("", inplace=True)
        result = pd.concat([cht_violated,x2])
        return result
def check_cht(item):
    v=pd.DataFrame()
    y = ['CHT Violations']
    cht_violated = pd.DataFrame(y,columns = [''])
    cht_violated.set_index("", inplace=True)
    v = pd.concat([cht_violated,v])
    for i in item['Resource'].unique():
        n=item[item['Resource']==i]
        n['CHT Overlap']=((n['Next Usage']-n['Clean_End_Time'])/(np.timedelta64(1, 's')*3600))-n['Max CHT']
        n['CHT Flag']=n['CHT Overlap'].apply(lambda x:1 if x>0 else 0)
        d=n[n['CHT Flag']==1]
        v=v.append(d)
    return v
###### 5. Checking Maintenance re-cleaning
def check_maint_recleaning(dataset):
    dataset = dataset[dataset['Maint_Flag']== 1]
    clean = dataset['clean_flag']
    x = dataset[dataset['clean_flag']==0]
    if (clean.all() == True):
        z = ['Cleaning conducted after all Maintenance']
        maint = pd.DataFrame(z,columns = [''])
        print(maint)
    else:
        z1 = ['Cleaning not conducted for the following schedules']
        maint1 = pd.DataFrame(z1,columns = [''])
        print(maint1)
        maint1.set_index("", inplace=True)
        result = pd.concat([maint1,x])
        return result
###### 6. Number of Pre-cleanings
def pre_re_cleaning(dataframe):
    df1 = dataframe[(dataframe['preclean']==1) & ~(dataframe['MC Group']==' ')]
    df1['MC Group'].replace('', np.nan, inplace=True)
    df1.dropna(subset=['MC Group'], inplace=True)
    data_temp = pd.DataFrame()
    for res in dataframe.Resource.unique():
        temp_df = dataframe[dataframe['Resource']==res]
        y=temp_df.min()['Usage_Start']
        data_temp = data_temp.append({"Resource":res,'Min_Usage_Start':y},ignore_index=True)
    new = pd.merge(df1,data_temp,on=['Resource'])
    new['CHT 1st U']=np.where((new['Usage_Start']== new['Min_Usage_Start']),'PREclean','REclean')
    x = new[new['CHT 1st U']=='PREclean'].shape[0]
    y =new[new['CHT 1st U']=='REclean'].shape[0]
    pre_re = pd.DataFrame([['Total no. of Pre-cleanings ',x],['Total no. of Re-cleanings',y]])
    return pre_re
###### 6. Variance of Cleaning Duration
def variance_Cleaning(dataset):
    temp_df = pd.DataFrame()
    for res in dataset.Resource.unique():
        data_temp = dataset[dataset['Resource'] == res]
        m = data_temp.var()['Cleaning Duration'].round(0)
        temp_df=temp_df.append({'Resource':res,'Variance of Cleaning Duration':m},ignore_index=True)
    return temp_df
##### 7. Flexibles mapping
def flexibles_map(dataframe):
    dataframe['flexible_resource_flag']=np.where((dataframe['Resource'].isin(['AS26 TD','Long Transfer Line - L717L','Tank : Tank Transfer'
                                                         'AS16','Harvested Blow & Wash Additions','AS16 Line - L2103',
                                                          'AS26 VII'])),1,0)
    dataframe['flexible_mapping_flag']= np.where(((dataframe['Resource']=='AS26 TD') & dataframe['Resource Alt'].isin(['Flexibles - Cryo KIT A','Flexibles - Cryo KIT B',
                                                                                       'Flexibles - Cryo KIT Colorati 1',
                                                                                       'Flexibles - Cryo KIT Colorati 2','AS26 TD'])),1,0)
    dataframe['flexible_mapping_flag']= np.where(((dataframe['Resource']=='Long Transfer Line - L717L') & dataframe['Resource Alt'].isin(['Flexibles - Mass Capture 1/2',
                                                                                           'Flexibles - Mass Capture 3/4',
                                                                                            'Long Transfer Line - L717L'])),1,dataframe['flexible_mapping_flag'])
    dataframe['flexible_mapping_flag']= np.where(((dataframe['Resource']=='AS26 VII') & dataframe['Resource Alt'].isin(['AS26 VII',
                                                                                           'Flexibles - AS26 VII',])),1,dataframe['flexible_mapping_flag'])
    dataframe['flexible_mapping_flag']= np.where(((dataframe['Resource']=='AS16 Line - L2103') & dataframe['Resource Alt'].isin(['AS16 Line - L2103',
                                                                                           'AS16',])),1,dataframe['flexible_mapping_flag'])
    flex = dataframe[(dataframe['flexible_resource_flag']==1) & (dataframe['flexible_mapping_flag']==0)]
    if (flex.shape[0] == 0):
        z = ['All flexibles have been mapped correctly']
        maint = pd.DataFrame(z,columns = [''])
        print(maint)
    else:
        g1 = ['Uncorrectly mapped flexibles']
        flex1 = pd.DataFrame(g1,columns = [''])
        print(flex1)
        flex1.set_index("", inplace=True)
        result = pd.concat([flex1,flex])
        return flex1
##### 8. Parallel Cleaning Mapping
def parallel_cleaning_map(dataframe):
    df = dataframe.groupby(['Parallel Clean Flag','Resource Alt','MC Group','Constraint'],as_index = False).agg({
                                                                                'Cleaning Duration':'max',
                                                                                'Clean_Start_Time':'min',
                                                                                'Clean_End_Time':'max'
                                                                                })
    pc_1=['Flexibles - Mass Capture 1/2','Flexibles - Mass Capture 3/4','Flexibles - AS16','Flexibles - Factor IX/FEIBA','Flexibles - AS26 VII']
    pc_2=['Flexibles - Cryo KIT A','Flexibles - Cryo KIT Colorati 1','Flexibles - Mass Capture 3/4','Flexibles - AS16','Flexibles - Factor IX/FEIBA','Flexibles - AS26 VII']
    pc_3=['Flexibles - Cryo KIT A','Flexibles - Cryo KIT Colorati 1','Flexibles - Mass Capture 1/2','Flexibles - Mass Capture 3/4','Flexibles - Factor IX/FEIBA','Flexibles - AS26 VII']
    pc_4=['Flexibles - Cryo KIT A','Flexibles - Cryo KIT Colorati 1','Flexibles - Mass Capture 1/2','Flexibles - Mass Capture 3/4','Flexibles - AS16','Flexibles - AS26 VII']
    pc_5=['Flexibles - Cryo KIT A','Flexibles - Cryo KIT Colorati 1','Flexibles - Mass Capture 1/2','Flexibles - Mass Capture 3/4','Flexibles - Factor IX/FEIBA','Flexibles - AS16']
    pc_6=['Short Line - L717S']
    pc_7=['AS26 TD']
    pc_3_4=['Flexibles - Cryo KIT A','Flexibles - Cryo KIT Colorati 1','Flexibles - Mass Capture 1/2','Flexibles - AS16','Flexibles - Factor IX/FEIBA','Flexibles - AS26 VII']
    pc1 = df.iloc[::2]
    pc2 = df.iloc[1::2]
    pc = pd.merge(pc1,pc2,on=['Parallel Clean Flag','MC Group','Constraint','Cleaning Duration','Clean_Start_Time','Clean_End_Time'])
    pc['PC_flag']= np.where(((pc['Resource Alt_x'] == 'Flexibles - Cryo KIT A') & (pc['Resource Alt_y'].isin(pc_1))),1,0)
    pc['PC_flag']= np.where(((pc['Resource Alt_x'] == 'Flexibles - Cryo KIT Colorati 1') & (pc['Resource Alt_y'].isin(pc_1))),1,pc['PC_flag'])
    pc['PC_flag']= np.where(((pc['Resource Alt_x'] == 'Flexibles - Mass Capture 1/2') & (pc['Resource Alt_y'].isin(pc_2))),1,pc['PC_flag'])
    pc['PC_flag']= np.where(((pc['Resource Alt_x'] == 'Flexibles - Mass Capture 3/4') & (pc['Resource Alt_y'].isin(pc_3_4))),1,pc['PC_flag'])
    pc['PC_flag']= np.where(((pc['Resource Alt_x'] == 'Flexibles - AS16') & (pc['Resource Alt_y'].isin(pc_3))),1,pc['PC_flag'])
    pc['PC_flag']= np.where(((pc['Resource Alt_x'] == 'Flexibles - Factor IX/FEIBA') & (pc['Resource Alt_y'].isin(pc_4))),1,pc['PC_flag'])
    pc['PC_flag']= np.where(((pc['Resource Alt_x'] == 'Flexibles - AS26 VII') & (pc['Resource Alt_y'].isin(pc_5))),1,pc['PC_flag'])
    pc['PC_flag']= np.where(((pc['Resource Alt_x'] == 'AS26 TD') & (pc['Resource Alt_y'].isin(pc_6))),1,pc['PC_flag'])
    pc['PC_flag']= np.where(((pc['Resource Alt_x'] == 'Short Line - L717S') & (pc['Resource Alt_y'].isin(pc_7))),1,pc['PC_flag'])
    pc_incorrect = pc[pc['PC_flag']==0]
    if ((pc['PC_flag']==1).all() == True):
        p_c = ['Parallel Cleaning Mapping is correct']
        p_clean = pd.DataFrame(p_c,columns = [''])
        print(p_clean)
    else:
        p1 = ['Incorrect Parallel Mappings']
        p_clean = pd.DataFrame(p1,columns = [''])
        print(p_clean)
        p_clean.set_index("", inplace=True)
        result = pd.concat([p_clean,pc_incorrect])
        return result
