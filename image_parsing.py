# -*- coding: utf-8 -*-


import requests
from bs4 import BeautifulSoup
import csv
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import re
import warnings
warnings.filterwarnings("ignore")
from google.colab import drive
drive.mount('/content/drive')
from tqdm import tqdm
from urllib.parse import urljoin, urlparse
import time
from multiprocessing import Pool
from fake_useragent import UserAgent
UserAgent().chrome

Links = pd.read_csv('movies_pr.csv')

def download(url, filename):
    '''
    Downloading images
    '''
    response = requests.get(url, stream=True, headers={'User-Agent': UserAgent().chrome} ) 
    with open(filename, "wb") as f:
        for data in response.iter_content(1024):
            f.write(data)
    return filename

def get_page(html):
    '''
    Getting page
    '''
    soup = BeautifulSoup(html, 'lxml')
    try:
        div = soup.find("a", class_="ipc-lockup-overlay ipc-focusable").get('href')
        return div
    except AttributeError:
        return 1
    
def get_image(html):
    '''
    Getting Image
    '''
    soup = BeautifulSoup(html, 'lxml')
    divs = soup.findAll("img")
    div = divs[1].get('src')
    return div

def main_image(df):
    k=[]
    start = df.index[0]
    stop = df.index[-1]
    base_url = 'https://www.imdb.com'
    for i in tqdm(range(start, stop+1)):
        time.sleep(0.01)
        url  = base_url + df.Url[i] 
        if i%1000==0:
            print(i, url)
        response = requests.get(url, headers={'User-Agent': UserAgent().chrome})
        html = response.text
        next_page = get_page(html)
        if next_page == 1:
            print(i, url)
            k.append(i)
        else:
            url = base_url + next_page
            response = requests.get(url, headers={'User-Agent': UserAgent().chrome})
            html = response.text
            img_url = get_image(html)
            img_name = df.Images[i]
            filename = download(img_url, img_name)
