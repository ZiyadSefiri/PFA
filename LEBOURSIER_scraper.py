import requests
from bs4 import BeautifulSoup
import playwright 
from playwright.sync_api import sync_playwright , Playwright
import json

stock ='cmt'

def run (playwright :Playwright , stock) :
    url = "https://medias24.com/content/api?method=getPriceHistory&ISIN=MA0000011793&format=json&from=2018-06-29&to=2025-03-31"
    chrome = playwright.chromium
    browser = chrome.launch(headless=False)
    page = browser.new_page()
    response =page.goto(url)
    json_text = response.text()
    try:
        data = json.loads(json_text)  #
        with open("data.json", "w") as file:
            json.dump(data, file, indent=4)
    except json.JSONDecodeError:
        print("Failed to parse JSON")
    pass
    browser.close()

with sync_playwright() as playwright :
    run (playwright,stock )

