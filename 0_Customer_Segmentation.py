import streamlit as st
import pandas as pd
import plotly.express as px
from plotly.graph_objects import Figure
from utils import load_css
import datetime
import numpy as np
import plotly.graph_objects as go
from typing import Union

@st.cache_data
def load_dataset() -> Union[pd.DataFrame,pd.DataFrame]:
    '''Load example dataset and cache'''
    df = pd.read_csv('dataset/cleaned/cleaned_retail_transaction.csv', index_col=None, parse_dates=['InvoiceDate'])
    df = df.assign(year=df.InvoiceDate.dt.year)
    df = df.assign(month=df.InvoiceDate.dt.month)
    df = df.assign(day=df.InvoiceDate.dt.dayofweek)
    df = df.assign(hour=df.InvoiceDate.dt.hour)
    df = df.assign(is_member=~df.CustomerID.isna())
    desc_product = df.groupby('StockCode')['Description'].first()
    return df, desc_product

def find_cohort_index(df):
    cohort_year = int(df['cohort_month'].year)
    cohort_month = int(df['cohort_month'].month)
    sale_year = int(df.year)
    sale_month = int(df.month)

    cohort_index = 12*(sale_year-cohort_year)+(sale_month-cohort_month) +1
    return cohort_index

def add_cohort(df):
    df['cohort_month'] = df.groupby(['CustomerID'])['InvoiceDate'].transform('min')
    df['cohort_index'] = df.apply(find_cohort_index,axis=1)
    df['cohort_month'] = df['cohort_month'].apply(lambda x: x.strftime("%Y-%m"))
    return df

@st.cache_data
def heatmap_retention(df):
    df = add_cohort(df)
    cohort = df.groupby(['cohort_month','cohort_index'], as_index=False)['CustomerID'].nunique()
    cohort.rename(columns={'cohort_month':'Cohort Month', 'cohort_index':'Retention rates by month'}, inplace=True)
    cohort = cohort.pivot(index='Cohort Month',columns='Retention rates by month',values='CustomerID')
    cohort_sizes = cohort.iloc[:,0]
    retention = cohort.divide(cohort_sizes, axis=0).round(3)*100
    # retention.iloc[:,0] = cohort.iloc[:,0]
    return retention

@st.cache_data 
def heatmap_primetime(df) -> Figure:
    pre_pivot = df.groupby(['day', 'hour'], as_index=False)['InvoiceNo'].nunique()
    pre_pivot.rename(columns={'day':'Day', 'hour':'Hour'}, inplace=True)
    freq_arrive_pivot = pre_pivot.pivot(index='Hour', columns='Day', values='InvoiceNo')
    freq_arrive_pivot.fillna(0, inplace=True)
    freq_arrive_pivot.rename(columns = {0:'Monday',1:'Tuesday', 2:'Wendesday',3:'Thusday', 4:'Friday',6:'Sunday'}, inplace=True)
    
    return freq_arrive_pivot

@st.cache_data
def load_rfm_data(df_mem) -> None:
    rfm_data = df_mem.groupby(['CustomerID']).agg({
        'InvoiceNo':'nunique',
        'Amount':'sum'
    })
    rfm_data.index = rfm_data.index.astype(int)
    rfm_data.rename(columns={'InvoiceNo':'Frequency', 'Amount':'MonetaryValue'}, inplace=True)

    snapshot_date = max(df_mem['InvoiceDate'])+datetime.timedelta(days=1)
    rfm_data['Recency'] = df_mem.groupby('CustomerID')['InvoiceDate'].apply(lambda x:(snapshot_date-x.max()).days)

    def freq_cut(freq: int) -> int:
        if freq <= rfm_data.Frequency.quantile(q=0.33):
            return 3
        elif freq <= rfm_data.Frequency.quantile(q=0.33):
            return 2
        else:
            return 1
        
    r_quartiles =  pd.qcut(rfm_data['Recency'], 3, labels=range(1,4)).astype(int)
    f_quartiles =  rfm_data['Frequency'].apply(freq_cut)
    m_quartiles =  pd.qcut(rfm_data['MonetaryValue'], 3, labels=range(3,0,-1)).astype(int)
    rfm_data = rfm_data.assign(R = r_quartiles)
    rfm_data = rfm_data.assign(F = f_quartiles)
    rfm_data = rfm_data.assign(M = m_quartiles)

    def join_rfm(x): return str(int(x['R']))+str(int(x['F']))+str(int(x['M']))

    rfm_data['Segment'] = rfm_data.apply(join_rfm, axis=1)
    rfm_data['rfm_Score'] = rfm_data[['R','F','M']].sum(axis=1)
    return rfm_data

def segment_label(df):
    if df['Segment']=='111':
        return 'Best'
    elif df['Segment']=='211':
        return 'Almost_Lost'
    elif df['Segment']=='311':
        return 'Lost'
    elif df['Segment']=='333':
        return 'Lost_Cheap'
    elif df['Segment'][2]=='1': # XX1
        return 'Spenders'
    elif df['Segment'][1]=='1': # X1X
        return 'Loyal'
    else:
        return 'Regular'


        
load_css()

st.title('Customer Segmentation')

df, stockcode2desc = load_dataset()
df_mem = df[df.is_member]
df_non_mem = df[~df.is_member]



tab1, tab2, tab3 = st.tabs(["Overview", "Segmentation", "Customer Profile"])

tab1.subheader('About dataset')
tab1.write('''
[Online Retail Data](https://archive.ics.uci.edu/ml/datasets/online+retail) contains all the transactions occurring between **01/12/2010** and **09/12/2011** for a UK-based and registered online retailer.
''', unsafe_allow_html=True)
tab1.markdown('---')

tab1.header('EDA')
with tab1.expander(label='First 100 rows of The dataset', expanded=False):
    raw_df = pd.read_csv('dataset/cleaned/cleaned_retail_transaction.csv', index_col=0, parse_dates=['InvoiceDate'], nrows=100)
    st.write(raw_df)
# tab1.subheader('EDA data')


num_rows = df.shape[0]
percent_mem_rows = df_mem.shape[0]/num_rows*100

tab1.subheader('Basic information')
c1, c2, c3, c4 = tab1.columns(4)
c1.metric(label='Number of rows', value=num_rows)
c2.metric(label='Transaction form member', value=f'{percent_mem_rows:.1f}%')
# c3.metric(label='Member in UK', value=f'{(uk_mem/uniq_mem*100):.1f}%')
# c4.metric(label='Number StockCodes', value=uniq_stockcode)

#  prepare metric values
uniq_mem=df[df.is_member].CustomerID.nunique()
uk_mem=df_mem[df_mem.Country=='United Kingdom']['CustomerID'].nunique()
uniq_stockcode=df.StockCode.nunique()
uniq_countrys=df.Country.nunique()
c1.metric(label='Number of member', value=uniq_mem)
c3.metric(label='Number of Countries', value=uniq_countrys)
c2.metric(label='Member in UK', value=f'{(uk_mem/uniq_mem*100):.1f}%')
c4.metric(label='Number of StockCodes', value=uniq_stockcode)

dfs = df.groupby('is_member')[['Amount']].sum()
fig = px.bar(dfs, x=dfs.index, y='Amount',)
fig.update_layout(
    title=dict(text='Compare sum amount between member and non-member',
               font=dict(size=24)),
    yaxis_title='Amount (Dollar)'
    )
tab1.plotly_chart(fig)

# dfs = df.groupby(['is_member', 'InvoiceDate'], as_index=False)[['Amount']].sum()
# dfs = dfs.assign(sum_amount=dfs.groupby('is_member')['Amount'].cumsum())
# fig = px.line(dfs, x='InvoiceDate', y='sum_amount', color='is_member', title='Cumsum of Amount base on member and non-member')
# tab1.plotly_chart(fig)

tab1.write('According to the revenue of member customer, that more than non-member. After this would analyze only member customer')

tab1.subheader('General questions which can find from transaction dataset')
tab1.write('''
- [What are top 5 products](#top-5-products)
- [When the most frequent customers purchases?](#heatmap-prime-time)
- [What is the retention of our customers? or When will the customers come back to buy our products again](#cohort-analysis)
''', unsafe_allow_html=True)
tab1.markdown('---')

#top 10 sales products
tab1.subheader('Top 5 products')
tab1.table(df.StockCode.value_counts().to_frame('Count').reset_index().rename(columns={'index':'StockCode'})[:5])
tab1.markdown('---')

#prime time
tab1.subheader('Heatmap Prime Time')
choice = tab1.selectbox(label='From', options=['Member data', 'Non-member data'], key='radio_heatmap_time')
if choice=='Member data':
    pivot = heatmap_primetime(df_mem)
else:
    pivot = heatmap_primetime(df_non_mem)
fig = px.imshow(pivot, text_auto=True)
tab1.plotly_chart(fig)
tab1.write('''Base on result from the heatmap above, we gain 2 insights.
- For member, have the most sales at 12 PM and the most peak on Wendesday
- For non-member, have the most sales at 3 PM and the most peak on Tuesday
''', unsafe_allow_html=True)
tab1.markdown('---')

# tab1.subheader('Cohort Analysis')
retention = heatmap_retention(df_mem)
fig = px.imshow(retention, text_auto=True)
fig.update_layout(title=dict(text='Cohort Analysis', font=dict(size=24)))
tab1.plotly_chart(fig, theme='streamlit')
tab1.write('''Base on result from the Cohort chart, we gain an insights.
- Customers make repeat purchases with in about 6 month
- If we have longer period data, we may see the lifetime of customers
- If we did a marketing campaigns, we can observe the effectiveness of campaigns by tracking customer acquisition rates and customer lifetime value in another cohort.
''', unsafe_allow_html=True)
with tab1.expander(label='Approximation of 6 months', expanded=False):
    st.image("images/cohort_example.png")
    st.write('But the reason which the number is high on that month, maybe had a promotion or campaign')
tab1.markdown('---')



# ------------------------tab2--------------------------

tab2.header('RFM Analysis')
rfm_data = load_rfm_data(df_mem)

with tab2.expander(label='RFM DataFrame', expanded=False):
    st.write(rfm_data[:100])

tab2.subheader('Condition of segmentation')
tab2.write('''
As RFM dataframe, customer who has Segment (the sequence order, 1 which means the best and so on)
- **111 -> Best**
- **211 -> Almost Lost**
- **311 -> Lost**
- **X1X -> Loyal**
- **XX1 -> Spender**
- **333 -> Lost Cheap**
- **the rest -> Regular**
''', unsafe_allow_html=True)
tab2.caption("'X' which means any segment order")

rfm_data['Segment_label'] = rfm_data.apply(segment_label, axis=1)
stat_segment = rfm_data.groupby('Segment_label').agg({
    'Recency':'mean',
    'Frequency':'mean',
    'MonetaryValue':['mean','count']
}).round(1)
stat_segment.columns = ['Recency_mean', 'Frequency_mean', 'MonetaryValue_mean', 'Count']
tab2.subheader('Statistics')
tab2.write(stat_segment)

fig = go.Figure(data=go.Pie(values=stat_segment.Count.values, labels=stat_segment.index.values,))
fig.update_traces(hole=.4, hoverinfo="label+percent+name", marker=dict(colors=px.colors.sequential.Blues))
fig.update_layout(
    title=dict(text='Pie Chart ratio of each segment',font=dict(size=24)),
    )
tab2.plotly_chart(fig)

df_mem = df_mem.merge(rfm_data['Segment_label'], how='inner',left_on='CustomerID',right_index=True)
segments_label = df_mem.Segment_label.unique()

dfs = df_mem.groupby(['Segment_label', 'CustomerID', 'InvoiceNo'], as_index=False)[['Amount']].sum()
selected_list = tab2.multiselect(
    label='Display on below Boxplot',
    options=segments_label,
    default=segments_label
    )
check = tab2.checkbox(label='Show Outlier')
dfs = dfs[dfs.Segment_label.isin(selected_list)]
fig = px.box(dfs, x='Segment_label', y="Amount", points='outliers' if check else False)
fig.update_layout(title=dict(text='Boxplot amount per bill group by segment', font=dict(size=24)), yaxis_title='Amount (Dollar)')
tab2.plotly_chart(fig)

dfs = df_mem.groupby('Segment_label')['Amount'].sum().sort_values(ascending=False)
fig = px.bar(dfs, x=dfs.index, y='Amount', title='Sum amount in each customer segment')
fig.update_layout(title=dict(text='Sum amount in each customer segment', font=dict(size=24)), yaxis_title='Amount (Dollar)')
tab2.plotly_chart(fig)

tab2.write('''
The above chart shows that the revenue we received from customer segments can lead to further budget decisions on marketing strategies. 

The strategies could be :
- **Best customers** -> Notice new products, Loyal programs
- **Spenders customers** -> Market the most expensive products
- **Almost lost/Lost customers** -> Price incentive
- **Lost cheap** -> Do not spend too much to re-acquire
''', unsafe_allow_html=True)
tab2.markdown('---')

tab2.header('EDA Segments')
tab2.subheader('Top products in a segment')
prod_count = df_mem.groupby('Segment_label')[['StockCode']].value_counts() \
    .sort_values(ascending=False).to_frame('Count')

c1, c2 = tab2.columns((2,1))
segment = c1.selectbox(label='Select Segment', options=segments_label, index=2)
topk = c2.number_input(label='Number of products', min_value=1, value=3)
df_top = prod_count.loc[segment][:topk]
df_top = df_top.assign(Product_Name = stockcode2desc.loc[df_top.index])
# df_top.sort_values('Count', ascending=True, inplace=True)

df_mem_seg = df_mem[df_mem.Segment_label==segment]
df_mem_seg = df_mem_seg[df_mem_seg.StockCode.isin(df_top.index)]
amount_top = df_mem_seg.groupby(['StockCode'])['Amount'].sum().to_frame('Amount') \
    .sort_values(by='Amount', ascending=False)
amount_top = amount_top.assign(Product_Name = stockcode2desc.loc[amount_top.index])

# variable = c1.selectbox(label='Select Segment', options=segments_label, index=2)
fig = px.bar(df_top, y='Count', x='Product_Name' ,)
fig.update_layout(title=dict(text=f'Top {topk} sales', font=dict(size=24)), yaxis_title='Sales (pieces)')
fig2 = px.bar(amount_top, y='Amount', x='Product_Name')
fig2.update_layout(yaxis_title='Amount (Dollar)')

tab2.plotly_chart(fig)
tab2.plotly_chart(fig2)

tab2.markdown('---')


# ------------------------tab3--------------------------

tab3.header('Customer Profile')
idx = tab3.selectbox(label='CustomerID',options=rfm_data.index)

trans = df_mem[df_mem.CustomerID==idx]
with tab3.expander(label='Transaction History', expanded=False):
    st.write(trans)

c1, c2, c3, c4 = tab3.columns(4)
c1.metric(label='CustomerID', value=idx)
c2.metric(label='Recency', value=rfm_data.loc[idx,'Recency'])
c3.metric(label='Frequency', value=rfm_data.loc[idx,'Frequency'])
c4.metric(label='MonetaryValue', value=rfm_data.loc[idx,'MonetaryValue'])



# tab3.write(rfm_data.loc[idx])
# c2.metric(label='Freq', value=trans.InvoiceNo.nunique())
# c2.metric(label='Freq', value=trans.InvoiceNo.nunique())
tab3.caption('Implementing...')
tab3.write('Churn model')
tab3.write('top3 repeat purchase products')
tab3.write('timeline plot the purchase days')

