
import requests
from bs4 import BeautifulSoup
import playwright 
from playwright.sync_api import sync_playwright , Playwright
import time
import json

def run (playwright :Playwright  ) :
    isin_list = {}
    url = "https://www.cdgcapitalbourse.ma/bourse/vuemarche"
    chrome = playwright.chromium
    browser = chrome.launch(headless=False)
    page = browser.new_page()
    page.goto(url)
    
    
    for i in range (1,100) :
        try :
        # Wait for the link element to appear
            selector = "#instrument-search-table-STOCK > tbody > tr:nth-child("+str(i)+") > td.longName > a"
            page.wait_for_selector(selector)

            # Extract the displayed text and the href attribute
            element = page.locator(selector)
            link_text = element.inner_text().strip()  # Get visible text
            link_href = element.evaluate("el => el.href")  # Get the href attribute
            for i in range(len(link_href)) :
                if link_href[i:i+6] == 'MA0000'  :
                     link_href=link_href[i:i+12]

            isin_list [link_text] = link_href

            with open ("isin_codes.json" , 'w' ) as f :
                json.dump(isin_list ,f)
        except :
            break
 
    print(isin_list)

    browser.close()
    

with sync_playwright() as playwright :
    run (playwright )