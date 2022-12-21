import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time 
import random
import os
from bs4 import BeautifulSoup

chrome_options = Options()
# chrome_options.add_argument("--headless")
# chrome_options.add_argument("--window-size=1920x1080")
driver = webdriver.Chrome(options=chrome_options, executable_path="/home/johannes/googleScrape/chromedriver")

sentenceFiles = ["unlabelled_test_data", "training_data"]
sentences0 = pd.read_csv(sentenceFiles[0] + ".csv")
sentences0 = sentences0["sentence"].tolist()
# if a sentence has more than 25 words, shorten it to 25 words
sentences0 = [" ".join(sentence.split()[:25]) for sentence in sentences0]


sentences0 = ['"' + sentence + '"' for sentence in sentences0]
sentences1 = pd.read_csv(sentenceFiles[1] + ".csv")
sentences1 = sentences1["sentence"].tolist()
sentences1 = [" ".join(sentence.split()[:25]) for sentence in sentences1]
sentences1 = ['"' + sentence + '"' for sentence in sentences1]
sentences = [sentences0, sentences1]


driver.get('https://www.google.com')
time.sleep(2)
driver.find_element_by_id('L2AGLb').click()
time.sleep(2)
driver.find_element_by_xpath("//*[contains(text(), 'English')]").click()

def check_load_success():
    requestSuccessful = False
    while not requestSuccessful:
        try:
            element = WebDriverWait(driver, 5).until(
                EC.presence_of_element_located((By.NAME, "q"))
            )
            search_bar = driver.find_element_by_name('q')
            search_bar.clear()
            requestSuccessful = True
        except:
            # print("fatal error")
            # exit()
            print("waiting for user to solve captcha")
            input()

for i in range(len(sentences)):
    for j in range(len(sentences[i])):
        outputFileAPI = "results/" + sentenceFiles[i] + str(j) + ".txt"
        outputFile = "results/" + sentenceFiles[i] + str(j) + "_scraped.txt"
        assert os.path.isfile(outputFileAPI)
        # read the text file as raw text string
        with open(outputFileAPI, 'r') as fin:
            results = fin.read()
        if (len(results) <= 4) and (not os.path.isfile(outputFile)):
            term = sentences[i][j]
            print(sentenceFiles[i], j)
            # wait for search bar to load and enter search term
            check_load_success()
            # enter search term
            search_bar = driver.find_element_by_name('q')
            search_bar.clear()
            search_bar.send_keys(term)
            search_bar.send_keys(Keys.RETURN)

            # check that the request was successful
            check_load_success()

            # wait for search results to load
            resultText = ""
            try:
                element = WebDriverWait(driver, 2).until(
                    EC.presence_of_element_located((By.CLASS_NAME, "yuRUbf"))
                )
                
                # remove the "people also ask" results
                soup = BeautifulSoup(driver.page_source, 'html.parser')
                for div in soup.find_all('div', {'class': 'Wt5Tfe'}):
                    div.decompose()
                
                # save to link name and full link to resultText in class yuRUbf
                for div in soup.find_all('div', {'class': 'yuRUbf'}):
                    resultText += div.find('h3').text.rstrip() + ' --- '
                    resultText += div.find('a').get('href').rstrip() + '\n'
                    
                time.sleep(random.randint(1, 2))
            except:
                print("No results found for term: " + term)
            finally:
                with open(outputFile, 'w') as f:
                    f.write(resultText)
