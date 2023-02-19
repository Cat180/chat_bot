# -*- coding: utf-8 -*-
'''
Parsing information from IMDB.com
'''


import requests
from bs4 import BeautifulSoup
import csv
import pandas as pd
import numpy as np
import os
import re
import warnings
warnings.filterwarnings("ignore")
from google.colab import drive
drive.mount('/content/drive')
from tqdm import tqdm
from urllib.parse import urljoin, urlparse
import time
from fake_useragent import UserAgent
UserAgent().chrome
from multiprocessing import Pool


def write_parse_data(data):
    '''
    Writing parsed data in csv dile
    '''
    with open('movies.csv','a') as f:
        writer = csv.writer(f)
        writer.writerow((
                         data['Title'],
                         data['Type'], 
                         data['Year'],
                         data['Age Limit'], 
                         data['Episodes'],
                         data['Time'],
                         data['Genre'],
                         data['IMDB'],
                         data['Metascore'],
                         data['Overview'],
                         data['Directed by'],
                         data['Starring'],
                         data['Box Office'],
                         data['Url']
                         ))

def get_next_page(html):
    '''
    Getting next page address from 'next' button
    '''
    soup = BeautifulSoup(html, 'lxml')
    divs = soup.find('div', class_='desc')
    div = divs.findChildren('a')
    value = []
    for cell in div:
      value.append(cell.get('href'))
    page = value[-1]
    return page

def get_type(html):
    '''
    Getting info from the main movie page
    '''
    soup = BeautifulSoup(html, 'lxml')
    ul = soup.find('ul', class_="ipc-metadata-list ipc-metadata-list--dividers-all title-pc-list ipc-metadata-list--baseAlt")
    divs = ul.findChildren('li', class_='ipc-metadata-list__item')
    #Getting directors or creators
    if len(divs) != 1:
      div = divs[0].findChildren('li')
      value = []
      for cell in div:
        value.append(cell.text)
      dir = ', '.join(value)
    else:
      dir = ''
    
    epi = soup.find("span", class_="ipc-title__subtext").text  #Getting number of episodes
    d = soup.find('div', class_="sc-80d4314-1 hbFqAr")
    typ = d.find('div').find('li', class_="ipc-inline-list__item").text   #Getting type (movie/TVSeries, etc.)
    return epi, typ, dir

def get_page_data(html):
    '''
    Parsing information
    '''
    soup = BeautifulSoup(html, 'lxml').find('div', class_='lister-list')
    films = soup.find_all('div', class_="lister-item-content")
    for film in films:
      episodes, types = '', 'Movie'
      try:
        try:
            url = film.find("h3", class_="lister-item-header").find('a').get('href')   
        except:
            url = ''
        try:
          title = film.find("h3", class_="lister-item-header").find('a').text
        except:
          title = ''
        try:
          year = film.find('span', class_="lister-item-year text-muted unbold").text
          year = re.sub(r'\(M{0,3}(CM|CD|D?C{0,3})?(XC|XL|L?X{0,3})?(IX|IV|V?I{0,3})\)', '', year).replace('() ', '')
        except:
          year = ''
        try:
          time = film.find('span', class_="runtime").text
        except:
          time = ''
        try:
          genre = film.find('span', class_="genre").text.strip()
        except:
          genre = ''
        try:
          descr = film.findChildren('p', class_="text-muted")[1].text.strip()
        except:
          descr = ''
        try:
          imdb = film.find('div', class_='inline-block ratings-imdb-rating').attrs['data-value']
        except:
          imdb = ''
        try:
          metascore = film.find('div', class_='inline-block ratings-metascore').text
          metascore = re.sub('Metascore', '', metascore).strip()
        except:
          metascore = ''
        try:
          votes, box_office = '', ''
          divs = film.find('p', class_="sort-num_votes-visible").findChildren('span')
          if len(divs) == 2: votes = divs[1].text
          elif len(divs) == 5:
            votes = divs[1].text
            box_office = divs[-1].text
        except:
          votes, box_office = '', ''
        try:
          age = film.find('span', class_='certificate').text
        except:
          age=''
        try:
          part = film.findChildren('p')[2].text
          director, actors = ['']*2
          if part.find('Director') != -1:
            director = re.sub('\n', '', part.split('|')[0].split(':')[-1].strip())
          if part.find('Star')!=-1:
            actors = re.sub('\n', '', part.rsplit(':')[-1].strip())
        except:
            director, actors = ['']*2
        if director == '':
          try:
            p_url = 'https://www.imdb.com' + url
            html = requests.get(p_url, headers={'User-Agent': UserAgent().chrome}).text
            episodes, types, director = get_type(html)
          except:
            episodes, types, director = '', 'Movie', ''

        data = {
            'Title': title,'Type': types, 'Year': year, 'Age Limit': age, 
            'Episodes': episodes, 'Time': time, 'Genre': genre, 'IMDB':imdb, 
            'Metascore': metascore, 'Overview': descr, 'Directed by': director,
            'Starring': actors, 'Box Office': box_office, "Url": url
            }
        write_parse_data(data)
      except UnicodeEncodeError:
        print('UnicodeEncodeError: ', data)
        continue

def main(genre, i=1, end=2001):
  '''
  Main part of collecting data
  '''
  base_url = 'https://www.imdb.com/'
  query_part = 'search/title/?'
  page_part = 'start='
  end_part = '&explore=title_type,genres&ref_=adv_nxt'
  genre_part = 'genres=' + genre + '&'
  url = base_url + query_part  + genre_part + page_part + str(i) + end_part
  print(genre, '\n', i, url)
  response = requests.get(url, headers={'User-Agent': UserAgent().chrome})   #Setting fake agent
  html = response.text
  get_page_data(html)
    
  for i in tqdm(range(51, end, 50)):
    next_page = get_next_page(html)
    url  = base_url + str(next_page) + '&ref_=adv_nxt'
    response = requests.get(url, headers={'User-Agent': UserAgent().chrome})
    html = response.text
    get_page_data(html)
    if i % 1000 == 0:
        print(i, url)

"""#actors"""

def write_data(data):
    with open('actors2.csv','a') as f:
        writer = csv.writer(f)
        writer.writerow((
                         data['Url'],
                         data['Directed by'],
                         data['Starring'],
                         ))

def get_data(html, url):
    #Getting full cast & directors
    soup = BeautifulSoup(html, 'lxml')
    ul = soup.find('ul', class_="ipc-metadata-list ipc-metadata-list--dividers-all title-pc-list ipc-metadata-list--baseAlt")
    try:
        divs = ul.findChildren('li', class_='ipc-metadata-list__item')
        if len(divs) == 1:
            dir = ''
        else:
            div = divs[0].findChildren('li')
            value = []
            for cell in div:
                value.append(cell.text)
            dir = ', '.join(value)
    except AttributeError:
        dir = ''
    a = soup.find('div', class_="ipc-sub-grid ipc-sub-grid--page-span-2 ipc-sub-grid--wraps-at-above-l ipc-shoveler__grid")
    try:
        al = a.findChildren('div', class_="sc-bfec09a1-5 kUzsHJ")
        act = []
        for aw in al:
            act.append(aw.find('div', class_="sc-bfec09a1-7 dpBDvu").find('a').text)
        acts = ', '.join(act)
    except AttributeError:
        acts = ''
    data = {'Url':url, 'Directed by':dir, 'Starring':acts}
    write_data(data)

def main(df, start=0, stop=0):
    stop = len(df)
    base_url = 'https://www.imdb.com'
    for i in tqdm(range(start, stop)):
        url = base_url + df.Url[i] 
        if i % 500 == 0:
            print(i, url)
        response = requests.get(url, headers={'User-Agent': UserAgent().chrome})
        html = response.text
        get_data(html, df.Url[i])

