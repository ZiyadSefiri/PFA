import requests
from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright
import json
import asyncio


from_date = "2024-03-31"
to_date = "2025-03-31"
stock_isin = "MA0000012296"

def run(playwright, stock_isin, from_date, to_date):
    url = "https://medias24.com/content/api?method=getPriceHistory&ISIN="+stock_isin+"&format=json&from="+from_date+"&to="+to_date
    browser = playwright.chromium.launch(headless=False)
    page = browser.new_page()

    response = page.goto(url)
    json_text = response.text()

    try:
        data = json.loads(json_text)
        with open(stock_isin + '.json', "w") as file:
            json.dump(data, file, indent=4)
    except json.JSONDecodeError:
        print("Failed to parse JSON")
    
    browser.close()

def stock_scraper(stock_isin, from_date, to_date):
    with sync_playwright() as playwright:
        run(playwright, stock_isin, from_date, to_date)
# To run the scraper asynchronously
#asyncio.run(stock_scraper(stock_isin, from_date, to_date))

