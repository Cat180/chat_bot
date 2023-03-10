# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import os
import re
import warnings
warnings.filterwarnings("ignore")
import sqlite3
import h5py
from google.colab.patches import cv2_imshow
import cv2
import matplotlib.pyplot as plt
import telebot
import random
from telebot import types
from multiprocessing import Pool
from fuzzywuzzy import fuzz
from fuzzywuzzy import process
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from google.colab import auth
from oauth2client.client import GoogleCredentials
import sys



bot = telebot.TeleBot('YOUR TOKEN')
#Authentification on Google Collab
auth.authenticate_user()
gauth = GoogleAuth()
gauth.credentials = GoogleCredentials.get_application_default()
drive = GoogleDrive(gauth)

#Handling command 'start'
@bot.message_handler(commands=['start'])
def start(message):
    markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
    btn = types.KeyboardButton("Greet 👋")
    markup.add(btn)
    bot.send_message(message.chat.id, "Hey! I am MovieDragon 🐉. I recommend films! 🎬", reply_markup=markup)
    
#Handling command 'again'
@bot.message_handler(commands=['again'])
def again(message):
    markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
    btn1 = types.KeyboardButton("Continue")
    btn2 = types.KeyboardButton("No, I don't")
    markup.add(btn1, btn2)
    bot.send_message(message.chat.id, "Hey! Do you want to continue with recommendations? 🤔", reply_markup=markup)
    
#Handling command 'help'
@bot.message_handler(commands=['help'])
def help(message):
    markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
    btn1 = types.KeyboardButton("Continue")
    btn2 = types.KeyboardButton("No, I don't")
    markup.add(btn1, btn2)
    msg = "Here are the commands that you can use:"
    coms = "/start - launch the chat\n/help - open the help menu\n/again - restart the search of movies when errors found\n"
    recs = "About recommendations \n"
    rec = "Overview-based - I will find movies similar to the chosen movie's plot"
    rec1 = "Actor-based - I will find movies with the chosen actor" 
    rec2 = 'Genre-based - I will find movies with the chosen genre'
    rec3 = "Movie of the Day - Movie on my choice"
    bot.send_message(message.chat.id, '\n'.join([msg, coms, recs, rec, rec1, rec2, rec3]), reply_markup=markup)
    bot.send_message(message.chat.id, "Do you want to continue?", reply_markup=markup)

#Handling user answers
@bot.message_handler(content_types=['text'])
def get_text_messages(message):
    markup = types.ReplyKeyboardMarkup(resize_keyboard=True) 
    btn1 = types.KeyboardButton('Overview-based')
    btn2 = types.KeyboardButton('Actor-based')
    btn3 = types.KeyboardButton('Genre-based')
    btn4 = types.KeyboardButton('Movie of the Day')
    markup.add(btn1, btn2, btn3, btn4)
    if message.text == 'Greet 👋':
        bot.send_message(message.chat.id, '✅ Choose category of recommendation', reply_markup=markup)
    elif message.text == 'help':
        bot.send_message(message.chat.id, 'Choose suitable category of recommendation and receive movies.', reply_markup=markup) 
    elif message.text == 'Overview-based':
        bot.send_message(message.from_user.id, 'Enter the title of the movie', reply_markup=markup)
        bot.register_next_step_handler(message, get_movie_title)
    elif message.text == 'Actor-based':
        bot.send_message(message.chat.id, "Enter actor's name", reply_markup=markup)
        bot.register_next_step_handler(message, get_actor_name)
    elif message.text == 'Genre-based':
        bot.send_message(message.chat.id, "Enter genre", reply_markup=markup)
        bot.register_next_step_handler(message, get_genre)   
    elif message.text == 'Movie of the Day':
        bot.send_message(message.chat.id, "Here's your movie of the day 🎞", reply_markup=markup)
        output(message, flag=np.random.randint(0, 15568)) 
    elif message.text == 'Continue':
        bot.send_message(message.from_user.id, '✅ Choose category of recommendation', reply_markup=markup)
        bot.register_next_step_handler(message, get_text_messages)
    elif message.text == "No, I don't":
        bot.send_message(message.from_user.id, 'OK, see you later!', reply_markup=markup)
    else:
        markup1 = types.ReplyKeyboardMarkup(resize_keyboard=True)
        btn1 = types.KeyboardButton("/again")
        markup1.add(btn1)
        bot.send_message(message.from_user.id, 'Something went wrong. Please enter /again.', reply_markup=markup1)


def get_movie_title(message):
    '''
    Processing movie title for recommendations
    '''
    global films, mapping
    movie_title = message.text.strip()
    films = pd.read_csv('movies_pr.csv')
    mapping = pd.Series(films.index, index=films["Title"])
    c = films.Title.values.tolist().count(movie_title)
    
    if c == 1:     #Movie title exists in dataset
        movie_index = mapping[movie_title]
        bot.send_message(message.from_user.id, 'Here are the movies you might enjoy 🎥')
        top_movies, movie_indices = recommend_movies_based_on_plot(movie_index)   #Getting recommendations
        markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
        btn0, btn1 = types.KeyboardButton('Yes'), types.KeyboardButton('No')
        markup.add(btn0, btn1)
        mes_id = bot.send_message(message.from_user.id, '\n'.join(top_movies)).message_id
        bot.send_message(message.from_user.id, 'Do you want to know more about these movies? 🎬',\
                         reply_to_message_id=mes_id, reply_markup=markup)
        bot.register_next_step_handler(message, answer, mes_id, top_movies, movie_indices)  #Getting response from user
        
    elif c == 0:   #Movie title doesn't exist in the dataset
        bot.send_message(message.from_user.id, 'Please wait ⏳')
        ts, ms = films.Title.values, []
        for x in range(len(ts)):   #Getting similar titles
            ms.append([x, ts[x], fuzz.WRatio(movie_title, ts[x])])
        ms = sorted(ms, key=lambda x: x[2], reverse=True)  #Sorting and getting 7 similar titles
        ms = [[x[0], x[1]] for x in ms[:7]]
        opt = [x[1] + ', ' + str(films.Year[x[0]]) for x in ms]
        markup2 = types.ReplyKeyboardMarkup(resize_keyboard=True)
        btn00, btn10 = types.KeyboardButton(opt[0]), types.KeyboardButton(opt[1])
        btn20, btn30 = types.KeyboardButton(opt[2]), types.KeyboardButton(opt[3])
        btn40, btn50, btn60 = types.KeyboardButton(opt[4]), types.KeyboardButton(opt[5]), types.KeyboardButton(opt[6])
        markup2.add(btn00, btn10, btn20, btn30, btn40, btn50, btn60)
        mes_id = bot.send_message(message.from_user.id, '❌ Movie not found. Found similar: \n\n' + '\n'.join(opt)).message_id
        bot.send_message(message.from_user.id, 'Enter preferred movie with year as written in the message',\
                         reply_to_message_id=mes_id, reply_markup=markup2)
        bot.register_next_step_handler(message, get_corrected_movie_title, mes_id)  #Suggesting user to choose one
        
    else:  #Found more than one movie with given title
        bot.send_message(message.from_user.id,'There are more than 1 movie with that title. Please choose one below to specify.')
        k = []
        movie_index = mapping[movie_title] 
        markup2 = types.ReplyKeyboardMarkup(resize_keyboard=True)
        for i in movie_index:
            k.append(f'{films["Title"][i]}, {films["Year"][i]}')
            markup2.add(films["Year"][i])
        mes_id = bot.send_message(message.from_user.id, '\n'.join(k)).message_id
        bot.send_message(message.from_user.id, 'Enter preferred year as written in the message above',\
                         reply_to_message_id=mes_id, reply_markup=markup2)   #suggesting to choose year
        bot.register_next_step_handler(message, get_movie_year, movie_title, mes_id)
   
def recommend_movies_based_on_plot(movie_index):
    '''
    Recommendation based on overview similarity
    '''
    with h5py.File('Sim_mat_15569.hdf5', 'r') as f:
        data_set = f['default']
        similarity_matrix = data_set[movie_index].tolist()
    similarity_score = list(enumerate(similarity_matrix))
    similarity_score = sorted(similarity_score, key=lambda x: x[1], reverse=True)
    similarity_score = similarity_score[1:11]
    movie_indices = [i[0] for i in similarity_score]
    top_movies = films[['Title', 'Year']].iloc[movie_indices].values 
    top_movies = [t[0] + ', ' + t[1] for t in top_movies]
    return top_movies, movie_indices

def output(message, flag, message_id=False, top_movies=False, movie_indices=False):
    '''
    Making output
    '''
    movie_title = message.text.title()
    films = pd.read_csv('movies_pr.csv')
    m_title = message.text.strip()
    if flag == True:   #If flag==True, that means that we futher suggest user to know more about each movie
        markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
        btn0, btn1 = types.KeyboardButton('Yes'), types.KeyboardButton('No')
        markup.add(btn0, btn1)
        
        try:    #Preparing the data for the output
            movie_index = movie_indices[list(top_movies).index(m_title)]
            movie = films[movie_index : movie_index + 1]
            movie.dropna(axis=1, inplace=True)
            if movie.Overview.values[0] in ['Add a Plot', 'Plot Unknown']:
                movie.drop(['Overview'], axis=1, inplace=True)
            if movie.Type.values[0] in ['Movie']:
                movie.drop(['Type'], axis=1, inplace=True)
            img = movie["Images"][movie_index]
            movie.drop(['Images', 'Url'], axis=1, inplace=True)
            vals = movie.values.tolist()[0]
            names = movie.columns.tolist()
            out = [str(n) + ': ' + str(v) if n not in ['Title', 'Year', 'Type'] else v for n, v in zip(names, vals)]
            out[0] = out[0] + '\n'
            out = ['\n' + k if 'Overview' in k  or 'Directed' in k else k for k in out ]
            
            try:    #If Photo exists in folder
                file_list = drive.ListFile({'q': "'folder_of_photos' in parents and title = '%s'" %img}).GetList()
                url = file_list[0]['alternateLink']
                medias = [types.InputMediaPhoto(url, '\n'.join(out))]
                bot.send_media_group(message.from_user.id, medias)
                
            except Exception:
                bot.send_message(message.from_user.id, '\n'.join(out))
                pass
            
            bot.send_message(message.from_user.id, 'Do you want to know more about these movies? 🎬',\
                            reply_to_message_id=message_id, reply_markup=markup)
            bot.register_next_step_handler(message, answer, message_id, top_movies, movie_indices)
        except ValueError:
            bot.send_message(message.from_user.id, 'Choose movie from the list below: 📃\n\n' + '\n'.join(top_movies),\
                             reply_markup=markup)
            bot.register_next_step_handler(message, output, True, message_id, top_movies, movie_indices)
    
    else:    #"handling output for 'Movie of the day'. Here suggestions for more films is not needed
        movie_index = flag
        movie = films[movie_index : movie_index + 1]
        movie.dropna(axis=1, inplace=True)
        if movie.Overview.values[0] in ['Add a Plot', 'Plot Unknown']:
            movie.drop(['Overview'], axis=1, inplace=True)
        if movie.Type.values[0] in ['Movie']:
            movie.drop(['Type'], axis=1, inplace=True)
        img = movie["Images"][movie_index]
        movie.drop(['Images', 'Url'], axis=1, inplace=True)
        vals = movie.values.tolist()[0]
        names = movie.columns.tolist()
        out = [str(n) + ': ' + str(v) if n not in ['Title', 'Year', 'Type'] else v for n, v in zip(names, vals)]
        out[0] = out[0] + '\n'
        out = ['\n' + k if 'Overview' in k or 'Directed' in k else k for k in out ]
        
        try:
            file_list = drive.ListFile({'q': "'folder_of_photos' in parents and title = '%s'" %img}).GetList()
            url = file_list[0]['alternateLink']
            medias = [types.InputMediaPhoto(url, '\n'.join(out))]
            bot.send_media_group(message.from_user.id, medias)
        except Exception:
            bot.send_message(message.from_user.id, '\n'.join(out))
            pass
        

def get_movie_year(message, movie_title, message_id):
    '''
    Getting movie year if there the title is not unique
    '''
    movie_year = message.text.strip()
    markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
    btn0, btn1 = types.KeyboardButton('Yes'), types.KeyboardButton('No')
    markup.add(btn0, btn1)
    
    if str(movie_year) == 'nan':  #Year is not defined in dataset
        movie_index = films[(films.Title == movie_title) & (films.Year.isna())].index[0]
        movie_index = films[(films.Title == movie_title) & (films.Year == movie_year)].index[0]
        top_movies, movie_indices = recommend_movies_based_on_plot(movie_index)
        bot.send_message(message.from_user.id, 'Here are the movies you might enjoy 🎥')
        mes_id = bot.send_message(message.from_user.id, '\n'.join(top_movies)).message_id
        bot.send_message(message.from_user.id, 'Do you want to know more about these movies? 🎬',\
                         reply_to_message_id=mes_id, reply_markup=markup)
        bot.register_next_step_handler(message, answer, mes_id, top_movies, movie_indices)
    else:
        try:
            movie_index = films[(films.Title == movie_title) & (films.Year == movie_year)].index[0]
            top_movies, movie_indices = recommend_movies_based_on_plot(movie_index)
            bot.send_message(message.from_user.id, 'Here are the movies you might enjoy 🎥')
            mes_id = bot.send_message(message.from_user.id, '\n'.join(top_movies)).message_id
            bot.send_message(message.from_user.id, 'Do you want to know more about these movies? 🎬',\
                         reply_to_message_id=mes_id, reply_markup=markup)
            bot.register_next_step_handler(message, answer, mes_id, top_movies, movie_indices)
        except IndexError:
            bot.send_message(message.from_user.id, 'Enter only year as written in the message', reply_to_message_id=message_id)
            bot.register_next_step_handler(message, get_movie_year, movie_title, message_id)
        
def get_corrected_movie_title(message, message_id):
    '''
    Getting the year and the title
    '''
    markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
    btn0, btn1 = types.KeyboardButton('Yes'), types.KeyboardButton('No')
    markup.add(btn0, btn1)
    
    try: 
        movie_title, movie_year = message.text.strip().split(',')
        movie_title, movie_year = movie_title.strip(), movie_year.strip()
        if str(movie_year) == 'nan':
            movie_index = films[(films.Title == movie_title) & (films.Year.isna())].index[0]
            top_movies, movie_indices = recommend_movies_based_on_plot(movie_index)
            bot.send_message(message.from_user.id, 'Here are the movies you might enjoy 🎥')
            mes_id = bot.send_message(message.from_user.id, '\n'.join(top_movies)).message_id
            bot.send_message(message.from_user.id, 'Do you want to know more about these movies? 🎬',\
                         reply_to_message_id=mes_id, reply_markup=markup)
            bot.register_next_step_handler(message, answer, mes_id, top_movies, movie_indices)
        else:
            try:
                movie_index = films[(films.Title == movie_title) & (films.Year == movie_year)].index[0]
                top_movies, movie_indices = recommend_movies_based_on_plot(movie_index)
                bot.send_message(message.from_user.id, 'Here are the movies you might enjoy 🎥')
                mes_id = bot.send_message(message.from_user.id, '\n'.join(top_movies)).message_id
                bot.send_message(message.from_user.id, 'Do you want to know more about these movies? 🎬',\
                         reply_to_message_id=mes_id, reply_markup=markup)
                bot.register_next_step_handler(message, answer, mes_id, top_movies, movie_indices)
            except IndexError:
                bot.send_message(message.from_user.id, 'Enter movie title with year as written in the message', reply_to_message_id=message_id)
                bot.register_next_step_handler(message, get_corrected_movie_title, message_id)
    except ValueError:
        bot.send_message(message.from_user.id, 'Enter movie title with year as written in the message', reply_to_message_id=message_id)
        bot.register_next_step_handler(message, get_corrected_movie_title, message_id)

def answer(message, message_id, top_movies, movie_indices):
    '''
    Handling user's answer for bot's suggestion for knowing more about films
    '''
    ans = message.text.title().strip()
    markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
    try: 
        btn0 = types.KeyboardButton(top_movies[0])
    except IndexError: 
        btn0 = ''
    try: 
        btn1 = types.KeyboardButton(top_movies[1])
    except IndexError: 
        btn1 = ''
    try: 
        btn2 = types.KeyboardButton(top_movies[2])
    except IndexError: 
        btn2 = ''
    try: 
        btn3 = types.KeyboardButton(top_movies[3])
    except IndexError: 
        btn3 = ''
    try: 
        btn4 = types.KeyboardButton(top_movies[4])
    except IndexError: 
        btn4 = '' 
    try: 
        btn5 = types.KeyboardButton(top_movies[5])
    except IndexError: 
        btn5 = ''
    try: 
        btn6 = types.KeyboardButton(top_movies[6])
    except IndexError: 
        btn6 = ''
    try: 
        btn7 = types.KeyboardButton(top_movies[7])
    except IndexError: 
        btn7 = ''
    try: 
        btn8 = types.KeyboardButton(top_movies[8])
    except IndexError: 
        btn8 = ''
    try: 
        btn9 = types.KeyboardButton(top_movies[9])
    except IndexError: 
        btn9 = ''
    btns = [btn0, btn1, btn2, btn3, btn4, btn5, btn6, btn7, btn8, btn9]
    btns = [b for b in btns if b != '']
    if len(btns) == 10:
        markup.add(btn0, btn1, btn2, btn3, btn4, btn5, btn6, btn7, btn8, btn9)
    else:
        for btn in btns: markup.add(btn)
    markup1 = types.ReplyKeyboardMarkup(resize_keyboard=True) 
    btn1, btn2 = types.KeyboardButton('Overview-based'), types.KeyboardButton('Actor-based')
    btn3, btn4 = types.KeyboardButton('Genre-based'), types.KeyboardButton('Movie of the Day')
    markup1.add(btn1, btn2, btn3, btn4)
    
    if ans.replace('.', '') in ["Yes", 'Yeah', 'Yap', 'Yep', 'Sure', 'Ok']:
        bot.send_message(message.from_user.id, 'Enter the title of one of the movies above',\
                         reply_to_message_id=message_id, reply_markup=markup)
        bot.register_next_step_handler(message, output, True, message_id, top_movies, movie_indices)
    elif ans.replace('.', '') in ['Nope', 'No', 'Meh', "Nah", 'Not Really', 'Neh']:
        bot.send_message(message.from_user.id, 'OK 👌', reply_markup=markup1) 
        bot.register_next_step_handler(message, get_text_messages)
    else:
        bot.send_message(message.from_user.id, 'Reenter answer') 
        bot.register_next_step_handler(message, answer, message_id, top_movies, movie_indices)

def get_actor_name(message, message_id=False, number=10):
    '''
    Getting actor name
    '''
    actor_name = message.text.title().strip()
    markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
    btn0, btn1 = types.KeyboardButton('Yes'), types.KeyboardButton('No')
    markup.add(btn0, btn1)
    
    #Preparing request to the database
    sql  = '''
    SELECT DISTINCT title, year 
    FROM movies as m 
    JOIN mov_act as ma ON m.id = ma.movie_id 
    JOIN stars as s ON ma.act_id = s.id
    WHERE s.name=:actor_name 
    ORDER BY rating ASC, imdb DESC, metascore DESC
    '''
    conn = sqlite3.connect('Movies.db')
    curr = conn.cursor()
    curr.execute(sql, {'actor_name':actor_name})
    movs = curr.fetchall()
    curr.close()
    conn.close()
    movie_title = message.text.title()
    global films, mapping
    films = pd.read_csv('movies_pr.csv')
    mapping = pd.Series(films.index, index=films["Title"])
    
    if movs != []:  #If actor found
        top_movies = movs[:number]
        movie_indices = [films[(films.Title == m) & (films.Year == y)].index[0]\
                         if str(y)!='None' else films[(films.Title == m) & (films.Year.isna())].index[0] for m, y in top_movies] 
        top_movies = [t[0] + ', ' + t[1] for t in top_movies]
        bot.send_message(message.from_user.id, 'Here are the movies you might enjoy 🎥')
        mes_id = bot.send_message(message.from_user.id, '\n'.join(top_movies)).message_id
        bot.send_message(message.from_user.id, 'Do you want to know more about these movies? 🎬',\
                         reply_to_message_id=mes_id, reply_markup=markup)
        bot.register_next_step_handler(message, answer, mes_id, top_movies, movie_indices)
        
    else:  # If actor not found
        bot.send_message(message.from_user.id, 'Please wait ⏳')
        # Finding similar name
        sqld  = '''
        SELECT DISTINCT name FROM stars as s 
        '''
        conn = sqlite3.connect('Movies.db')
        curr = conn.cursor()
        curr.execute(sqld)
        a = curr.fetchall()
        curr.close()
        conn.close()
        ms = []
        for j in range(len(a)):
            ms.append([a[j], fuzz.WRatio(movie_title, a[j])])
        ms = sorted(ms, key=lambda x: x[-1], reverse=True)
        ms = [x[0][0] for x in ms[:10]]
        
        markup1 = types.ReplyKeyboardMarkup(resize_keyboard=True)
        btn00, btn10 = types.KeyboardButton(ms[0]), types.KeyboardButton(ms[1])
        btn20, btn30 = types.KeyboardButton(ms[2]), types.KeyboardButton(ms[3])
        btn40, btn50 = types.KeyboardButton(ms[4]), types.KeyboardButton(ms[5])
        btn60, btn70 = types.KeyboardButton(ms[6]), types.KeyboardButton(ms[7])
        btn80, btn90 = types.KeyboardButton(ms[8]), types.KeyboardButton(ms[9])
        markup1.add(btn00, btn10, btn20, btn30, btn40, btn50, btn60, btn70, btn80, btn90)
        mes_id = bot.send_message(message.from_user.id, '❌ Actor not found. Found similar: \n\n' + '\n'.join(ms)).message_id
        bot.send_message(message.from_user.id, 'Enter preferred actor as written in the message above',\
                         reply_to_message_id=mes_id, reply_markup=markup1)
        bot.register_next_step_handler(message, get_actor_name, mes_id)

def get_genre(message, message_id=False, number=10):
    '''
    Getting genre
    '''
    genre = message.text.title().strip()
    markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
    btn0, btn1 = types.KeyboardButton('Yes'), types.KeyboardButton('No')
    markup.add(btn0, btn1)
    
    # Requesting data from the database
    sql  = '''
    SELECT DISTINCT title, year FROM movies as m 
    JOIN mov_gen as mg ON m.id = mg.movie_id 
    JOIN genres as g ON mg.genre_id = g.id
    WHERE g.genre=:genre
    '''
    conn = sqlite3.connect('Movies.db')
    curr = conn.cursor()
    curr.execute(sql, {'genre':genre})
    movs = curr.fetchall()
    curr.close()
    conn.close()
    movie_title = message.text.title()
    global films, mapping
    films = pd.read_csv('movies_pr.csv')
    mapping = pd.Series(films.index, index=films["Title"])
    
    if movs != []:  #If genre found
        inx = np.random.randint(0, len(movs), number)
        top_movies = [movs[i] for i in inx]
        movie_indices = [films[(films.Title == m) & (films.Year == y)].index[0] if str(y)!='None'\
                         else films[(films.Title == m) & (films.Year.isna())].index[0] for m, y in top_movies] 
        top_movies = [t[0] + ', ' + t[1] for t in top_movies]
        bot.send_message(message.from_user.id, 'Here are the movies you might enjoy 🎥')
        mes_id = bot.send_message(message.from_user.id, '\n'.join(top_movies)).message_id
        bot.send_message(message.from_user.id, 'Do you want to know more about these movies? 🎬',\
                         reply_to_message_id=mes_id, reply_markup=markup)
        bot.register_next_step_handler(message, answer, mes_id, top_movies, movie_indices)
        
    else:
        bot.send_message(message.from_user.id, 'Please wait ⏳')
        sqld  = '''
        SELECT DISTINCT genre FROM genres
        '''
        conn = sqlite3.connect('Movies.db')
        curr = conn.cursor()
        curr.execute(sqld)
        a = curr.fetchall()
        curr.close()
        conn.close()
        ms = []
        for j in range(len(a)):
            ms.append([a[j], fuzz.WRatio(movie_title, a[j])])
        ms = sorted(ms, key=lambda x: x[-1], reverse=True)
        ms = [x[0][0] for x in ms[:5]]
        
        markup1 = types.ReplyKeyboardMarkup(resize_keyboard=True)
        btn00, btn10 = types.KeyboardButton(ms[0]), types.KeyboardButton(ms[1])
        btn20, btn30, btn40 = types.KeyboardButton(ms[2]), types.KeyboardButton(ms[3]), types.KeyboardButton(ms[4])
        markup1.add(btn00, btn10, btn20, btn30, btn40)
        mes_id = bot.send_message(message.from_user.id, '❌ Genre not found. Found similar:\n\n' + '\n'.join(ms))
        bot.send_message(message.from_user.id, 'Enter preferred genre as written in the message above',\
                         reply_to_message_id=mes_id, reply_markup=markup1)
        bot.register_next_step_handler(message, get_genre, mes_id)

bot.polling(none_stop=True, interval=0)
