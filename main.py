import pandas as pd
import numpy as np
import LEBOURSIER_scraper as scraper
import os
import json
import streamlit as st
import threading
import time
import asyncio
import sys
from datetime import date, timedelta
from math import log , sqrt , ceil
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, dendrogram
import plotly.express as px


from_date = "2024-03-31"
to_date = str(date.today() - timedelta(days=1))



if sys.platform.startswith('win'):
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

with open("isin_codes.json" , 'r' ) as f1 :
    stocks = json.load(f1)

stocks_names = list(stocks.keys())
stocks_isins = list (stocks.values())
choice = st.selectbox("Stocks" , stocks_names)

if "df" not in st.session_state : #initilising the main data_frame
    
    st.session_state.df = pd.DataFrame()

if "portfolio" not in st.session_state :
    st.session_state.portfolio = {} 

new_df = pd.DataFrame()
default_df = pd.DataFrame({'date': pd.date_range(start=from_date, end=to_date, freq='D')})


if st.button("ADD STOCK TO POTFOLIO"):
    scraper.stock_scraper(stocks[choice], from_date, to_date)
     #adding stock to portfolio
    st.success("Stock data fetched!")

    if not st.session_state.portfolio  : # first_stock
        with open(stocks[choice]+'.json', 'r') as data_file:
            data = json.load(data_file)

        st.session_state.df = pd.DataFrame(data["result"])
        st.session_state.df = st.session_state.df[["date","value"]]
        st.session_state.portfolio[choice] = stocks [choice]
        st.session_state.df.rename(columns={"value" : choice} , inplace=True)
        st.session_state.df['date'] = pd.to_datetime(st.session_state.df['date'] , dayfirst=True)
        st.session_state.df = pd.merge (st.session_state.df , default_df ,left_on='date' , right_on='date' , how = 'outer')
        st.session_state.df[choice].interpolate(method = 'linear' , inplace =True)


    elif choice not in st.session_state.portfolio.keys(): #subsequent_stocks
        with open(stocks[choice]+'.json', 'r') as data_file:
            data = json.load(data_file)

        new_df = pd.DataFrame(data["result"])
        new_df = new_df[["date","value"]]
        new_df['date']= pd.to_datetime(new_df['date'] , dayfirst=True )
        st.session_state.portfolio[choice] = stocks [choice]
        new_df = pd.merge (new_df , default_df ,left_on='date' , right_on='date' , how = 'outer')
        st.session_state.df = pd.merge(new_df , st.session_state.df , on = "date" , suffixes=("",choice))
        st.session_state.df.rename(columns={"value" : choice} , inplace=True)
        st.session_state.df[choice].interpolate(method = 'linear' , inplace =True)

       

st.write ( "stocks price")
st.write (st.session_state.df)

#st.write ( st.session_state.portfolio)

#line chart

st.write("stocks_chart")
try :
    chart_df = st.session_state.df.set_index('date')
    chart = st.line_chart(chart_df)
except :
    pass

#calculating log returns

today_df = st.session_state.df.iloc[1: , 1:].reset_index(drop = True)

yesterday_df = st.session_state.df.iloc[ 0:-1 , 1:]

returns = today_df/yesterday_df
returns = returns.applymap(log)

st.write("stocks log returns")
st.write(returns)
dendrogram_labels = list(returns.columns)

#corr_matrix

correlation_matrix =  returns.corr()
st.write("correlation matrix")
plt.figure(figsize=(8, 6))

# Create the heatmap
try :
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
except :
    pass

st.pyplot(plt)

#calculatind distance matrix 
dis_matrix = correlation_matrix.applymap(lambda a : sqrt(1-a))


#altered distance matrix
dis_matrix = dis_matrix.values 

dis_matrix2 = np.zeros(dis_matrix.shape)
for i in range (dis_matrix2.shape[1]) :
    for j in range (dis_matrix2.shape[1]) : 
        dis_matrix2[i,j] = np.linalg.norm(dis_matrix[:, i ]  - dis_matrix[:, j ] )
#st.write (dis_matrix2)

#perform clustering
try :
    dist_condensed = squareform(dis_matrix2, checks=False)
    link = linkage(dist_condensed, method='ward')
    plt.figure(figsize=(8, 4))

except : 
    pass

fig, ax = plt.subplots(figsize=(10, 5))
dendrogram(link ,  ax=ax , labels=dendrogram_labels , leaf_font_size=8)
plt.title("Dendrogram")
plt.xlabel("Entreprise")

plt.show()

st.pyplot(fig)

#reorganising the covariance matrix :
def Quasidiag(link2):
    link2 = link2.astype(int)
    sortIx = pd.Series([link2[-1, 0], link2[-1, 1]])
    numItems = link2[-1, 3]
    while sortIx.max() >= numItems:
        sortIx.index = range(0, sortIx.shape[0] * 2, 2)
        df0 = sortIx[sortIx >= numItems]
        i = df0.index
        j = df0.values - numItems
        sortIx[i] = link2[j, 0]  # item 1
        df0 = pd.Series(link2[j, 1], index=i + 1)  # item 2
        sortIx = pd.concat([sortIx, df0])
        sortIx = sortIx.sort_index()  # re-sort
        sortIx.index = range(sortIx.shape[0])
    return sortIx.tolist()


def recursive_bisection(assets_list, n):
    # assets_list: list of column indices (e.g., [0, 1, 2, 3])
    # n: number of total assets

    L_list = [assets_list]
    weights = [1.0 for _ in assets_list]

    while len(L_list) != n:
        temp_list = []
        k = len(L_list)
        counter = 0

        for i in range(k):
            ki = len(L_list[i])
            if ki != 1:
                # Split sub-portfolio
                L1 = L_list[i][:ceil(ki / 2)]
                L2 = L_list[i][ceil(ki / 2):]
                temp_list.append(L1)
                temp_list.append(L2)

                # Extract corresponding columns (assets) from returns
                L1_pd = returns.iloc[:, L1]
                L2_pd = returns.iloc[:, L2]

                # Covariance matrices
                L1_cov = L1_pd.cov().to_numpy()
                L2_cov = L2_pd.cov().to_numpy()

                # Inverse variances (proxy for risk budgeting)
                L1_weights = np.diag(np.linalg.pinv(L1_cov))
                L2_weights = np.diag(np.linalg.pinv(L2_cov))

                # Proxy risk measure (can be adapted)
                L1_risk = L1_weights.T @L1_cov@ L1_weights
                L2_risk = L2_weights.T @L2_cov@ L2_weights

                alpha_1 = 1 - L1_risk / (L1_risk + L2_risk)
                alpha_2 = 1 - L2_risk / (L1_risk + L2_risk)

                for j in range(len(L1)):
                    weights[counter + j] *= alpha_1
                for j in range(len(L2)):
                    weights[counter + len(L1) + j] *= alpha_2

                counter += len(L_list[i])

            else:
                temp_list.append(L_list[i])
                counter += 1

        L_list = temp_list

    return weights


#st.write(Quasidiag(link))
#st.write(Quasidiag(link) [0])
#st.write(recursive_bisection(Quasidiag(link) , len(st.session_state.portfolio.keys())))
#weights for capital allocation
try :
    weights = recursive_bisection(Quasidiag(link) , len(st.session_state.portfolio.keys()))
except :
    pass
st.write ("Optimal Capital Allocation")
#pie chart
labels = st.session_state.df.columns.tolist()[1:]
try :
    labels = [labels[i] for i in Quasidiag(link) ]
except :
    pass
fig =plt.figure(figsize=(6,6))
try :
    plt.pie(weights, labels=labels, autopct='%1.1f%%', startangle=140)
except :
    pass
st.pyplot(fig)

#mothly frequency backtesting
def backtest (assets ,weights ,start_date , end_date) :
    df = pd.DataFrame(stocks = assets )
    df[returns] = st.session_state[date]

