import streamlit as st
import pandas as pd
import plotly.express as px
from plotly.graph_objects import Figure
from utils import load_css

@st.cache_data
def load_dataset() -> pd.DataFrame:
    '''Load example dataset and cache'''
    df = pd.read_csv('dataset/cleaned/cleaned_retail_transaction.csv', index_col=None, parse_dates=['InvoiceDate'])
    df = df.assign(year=df.InvoiceDate.dt.year)
    df = df.assign(month=df.InvoiceDate.dt.month)
    df = df.assign(day=df.InvoiceDate.dt.dayofweek)
    df = df.assign(hour=df.InvoiceDate.dt.hour)
    df = df.assign(is_member=~df.CustomerID.isna())

    return df

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

load_css()

st.title('Customer Segmentation')

df = load_dataset()
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
fig = px.bar(dfs, x=dfs.index, y='Amount', title='Compare sum amount between member and non-member customer')
tab1.plotly_chart(fig)

# dfs = df.groupby(['is_member', 'InvoiceDate'], as_index=False)[['Amount']].sum()
# dfs = dfs.assign(sum_amount=dfs.groupby('is_member')['Amount'].cumsum())
# fig = px.line(dfs, x='InvoiceDate', y='sum_amount', color='is_member', title='Cumsum of Amount base on member and non-member')
# tab1.plotly_chart(fig)

tab1.write('According to the revenue of member customer, that more than non-member. After this would analyze only member customer')

tab1.subheader('General questions which can find from transaction dataset')
tab1.write('''
- [When the most frequent customers purchases?](#heatmap-prime-time)
- [What is the retention of our customers? or When will the customers come back to buy our products again](#cohort-analysis)
''', unsafe_allow_html=True)
tab1.markdown('---')
#top 10 sales products




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

tab1.subheader('Cohort Analysis')
retention = heatmap_retention(df_mem)
fig = px.imshow(retention, text_auto=True)
tab1.plotly_chart(fig, theme='streamlit')
tab1.write('''Base on result from the Cohort chart, we gain an insights.
- Customers make repeat purchases with in about 6 month
- If we have longer period data, we may see the lifetime of customers
- If we did a marketing campaigns, we can observe the effectiveness of campaigns by tracking customer acquisition rates and customer lifetime value in another cohort.
''', unsafe_allow_html=True)
with tab1.expander(label='Approximation of 6 months', expanded=False):
    st.image("images/cohort_example.png")
    st.write('observe')
tab1.markdown('---')



# ------------------------tab2--------------------------

tab2.header('Segmentation process')
tab2.caption('Implementing...')

# ------------------------tab3--------------------------

tab2.header('Customer Profile')
tab2.caption('Implementing...')


