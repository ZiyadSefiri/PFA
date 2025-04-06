import pandas as pd
import numpy as np
import LEBOURSIER_scraper as scraper
import os
import  json
import streamlit as st
import threading
import time
import asyncio
import sys

from_date = "2024-03-31"
to_date = "2025-03-31"



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


    else: #subsequent_stocks
        with open(stocks[choice]+'.json', 'r') as data_file:
            data = json.load(data_file)

        new_df = pd.DataFrame(data["result"])
        new_df = new_df[["date","value"]]
        st.session_state.portfolio[choice] = stocks [choice]
    
        st.session_state.df = pd.merge(new_df , st.session_state.df , on = "date")

        



st.write(st.session_state.df)
st.write(st.session_state.portfolio)