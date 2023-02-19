# -*- coding: utf-8 -*-


import pandas as pd
import numpy as np
import os
from google.colab import drive
drive.mount('/content/drive')
import re
import warnings
warnings.filterwarnings("ignore")
import sqlite3
from tqdm import tqdm
from IPython.display import Audio, display
from sqlite3.dbapi2 import OperationalError, IntegrityError
from multiprocessing import Pool

"""##part 1

###subpart1
"""

dt = pd.read_csv('movies_pr.csv')
dt.drop_duplicates(inplace=True)
dt.reset_index(inplace=True, drop=True)

for i in tqdm(range(len(dt))):
  if str(dt.Year[i]).__contains__('TV Special') == True:
    dt.Year[i] = str(dt.Year[i]).replace(' TV Special', '')
    dt.Type[i] = 'TV Special'
  elif str(dt.Year[i]).__contains__('TV Movie') == True:
    dt.Year[i] = str(dt.Year[i]).replace(' TV Movie', '')
    dt.Type[i] = 'TV Movie'
  elif str(dt.Year[i]).__contains__('TV Short') == True:
    dt.Year[i] = str(dt.Year[i]).replace(' TV Short', '')
    dt.Type[i] = 'TV Short'

ind = dt[dt.IMDB.isna()].index
dt.drop(index=ind, inplace=True)
dt.reset_index(inplace=True)

ind11 = dt[dt.Overview.str.contains('Plot|Post-production|Filming|See full summary')].index.tolist()
ind12 = dt[dt.IMDB.isna()].index.tolist()
ind13 = dt[(dt.Starring.isna())&(dt['Genre(s)'].str.contains('Animation')==False)].index.tolist()
ind14 = dt[(dt.Year.str.contains(r'Podcast Series|Video Game|Music Video|Video', na=False))].index.tolist()
inx = list(set(ind11 + ind12 + ind13 + ind14))
dt.drop(index=inx, inplace=True)
dt.drop_duplicates(inplace=True)
dt.shape

# dt.drop('index', axis=1, inplace=True)
dt.drop(index=ind, inplace=True)
dt.reset_index(drop=True, inplace=True)
dt.shape

ind = [5021, 6510, 6554, 5631, 6613, 750, 4831, 6624, 970, 5210,\
       1217, 5236, 6393, 13510, 6426, 6716, 6724, 5355, 9243, 10220,\
       13389, 3155, 5080, 4161, 9249, 6858, 6151, 4566, 5214, 5579,\
       5629, 6176, 9281, 5398, 9414, 6725, 6058, 15704, 13623, 15703,\
       9178, 13447, 13337, 13757]

dt[dt.Overview.str.contains('When the brilliant but unorthodox scientist')]

dt[dt.duplicated(['Overview'])][7:]

"""###Overviews"""

dt = pd.read_csv('movies_pr.csv')
dt.shape

ind, k = [], []
for i in tqdm(range(len(dt))):
  if dt.Overview[i].__contains__('?') and dt.Overview[i][-1]!='?':
    ind.append(i)
    words = dt.Overview[i].split()
    for word in words:
      if word.__contains__('?') and word not in k:
        k.append(word)

len(ind), len(k)

dt.Title[26] = '8½'
dt.Title[2389] = '90 Day Fiancé'
dt.Title[4616] = 'Ófærð'
dt.Title[6117] = 'Kona fer í stríð'
dt.Title[7255] = 'Le Samouraï'
dt.Title[8153] = 'The Protégé'
dt.Title[10491] = 'Project Mc²'
dt.Title[10885] = 'Liberté'
dt.Title[11408] = 'Charité'
dt.Title[11556] = 'Marketa Lazarová'
dt.Title[12163] = 'Alien³'
dt.Title[12843] = 'Be Natural: The Untold Story of Alice Guy-Blaché'
dt.Title[13037] = 'Salomé'
dt.Title[13528] = '[Rec]²'
dt.Title[14352] = 'A White Dress for Marialé'
dt.Title[15465] = 'Ranma ½'
dt.Overview[14705] = 'War-hungry teddy bears journey from bootcamp to the psychedelic terrors of the Magic Forest in this darkly beautiful horror animation.'

dt.Overview[dt.Overview.str.contains('unicorn', )]

start, step = 14705, 1
dt[start:start+step]

start, step = 230, 5
k[start:start+step]

start, step = 250, 5
dt.loc[ind, ['Title', 'Year', 'Overview']][start:start+step].values

d = {'prot?g?':'protégé', 'T?r':'Tár', 'fianc?':'fiancé', 'Bogot?': 'Bogotá', 'Sof?a':'Sofía', '?milie':'Émilie', '?lex':'Álex', 'na?ve':'naïve',\
     'L?o':'Léo', "R?mi":"Rémi", 'Mal?na':'Malèna', 'Ad?le':'Adèle', "d'?tat":"d'état",'Op?ra':'Opéra', 'proteg?':'protégé', 'Merl?':'Merlí',\
     'J?gerst?tter':'Jägerstätter', ' caf?':' café', 'd?butante':'débutante', 'client?le':'clientèle', 'divorc?':'divorcé', 'H?l?ne':'Hélène',\
     '?dith':'Édith', 'Ars?ne':'Arsène', "d\'?garement":"d'égarement", 'Chlo?':'Chloé', 'S?o Paulo':'São Paulo', 'fa?ade':'façade', 'attach?':'attaché',\
     'S?nchez':'Sánchez','Mu?ez':'Muñez', 'Atticus P?nd':'Atticus Pünd','?ber rich':'über rich', 'Jacob St?rr':'Jacob Störr','the Le?ncio':'the Leôncio',\
     'the Marru?':'the Marruá', 'Am?lia':'Amélia', 'm?nage ? trois':'ménage à trois', 'M?tley Cr?e':'Mötley Crüe','9? Weeks':'9½ Weeks', 'M?ria':'Mária',\
     'Jos?':'José', 'Mart?n':'Martín', 'Ana?s':'Anaïs', 'Caf?':'Café', 'Fr?d?ric':'Frédéric', 'Ra?l':'Raúl', 'Ragnar?k': 'Ragnarök',\
     'imp Blitz?':'imp Blitzø', 'Pok?mon': 'Pokémon', '\x92':"'", '?50k':'£50k', 'Egbert Sous?':'Egbert Sousé', 'Son Gok?':'Son Gokû', ' Ren?':' René',\
     'Diakit?':'Diakité', 'Ast?rix':'Astérix', 'Ob?lix':'Obélix', 'Num?robis':'Numérobis', 'Ram?n': 'Ramón', 'D?sseldorf':'Düsseldorf', 'Br?no':'Brüno',\
     'Medell?n':'Medellín','Saga Nor?n':'Saga Norén', 'Ko?ovali':'Koçovali', '?eki? ve G?l':'Çekiç ve Gül', 'Rapha?l':'Raphaël', '?mer U?ar':'Ömer Uçar',\
     'Mill?n':'Millán','Malm?':'Malmö', 'doppelg?nger':'doppelgänger','Th?r?se':'Thérèse','Jens M?ller':'Jens Møller', 'Bj?rn Andresen':'Björn Andresen',\
     'Bj?rn':'Bjørn', '?on Flux':'Æon Flux', 'soup?on':'soupçon', 'Neuk?lln':'Neukölln', 'Carl M?rck':'Carl Mørck', 'Bouvet?ya':'Bouvetøya',\
     'Faberg?':'Fabergé', "Tot? u Curtu":'Totò u Curtu', 'expos?':'exposé', 'Montr?al':'Montréal', 'H?ctor':'Héctor', 'Belascoar?n':'Belascoarán',\
     '?ngela':'Ángela', 'Jordn?ra':'Jordnära', 'Padm?':'Padmé', 'Khlo?':'Khloé', 'In?s':'Inés', 'clich?':'cliché', 'Z? Carioca':'Zé Carioca',\
     'Kung F?hrer':'Kung Führer', 'Bu?uel':'Buñuel', 'Dal?':'Dalí', 'Lou?s':'Louÿs', 'R?gnier':'Régnier', '?rp?d':'Árpád', 'Gl?ria':'Glória',\
     'Jo?o':'João','Per?n':'Perón', 'G?k Han':'Gök Han', 'D?az':'Díaz', 'Sebasti?o':'Sebastião', 'Bront?':'Brontë','?1 million':'£1 million',\
     'H?kon':'Håkon', 'Sobib?r':'Sobibór', 'Charit?':'Charité', 'Mar?a':'María', 'Due?as':'Dueñas', 'C?vennes':'Cévennes','Cond?':'Condé',\
     'Hern?n':'Hernán', 'Cort?s':'Cortés', 'M?xico':'México', 'Sm?land':'Småland', 'S?mi':'Sámi', 'Asunci?n':'Asunción', 'Ern?':'Ernõ', 'F?in':'Féin',\
     '?guila':'Águila', 'Lo?e':'Loïe', 'Berg?res':'Bergères', 'Lumi?re':'Lumière', 'P?ricourt':'Péricourt', 'C?line':'Céline', 'G?ring':'Göring',\
     'Assembl?e':'Assemblée', 'F?lix':'Félix', 'Ang?lique':'Angélique', 'derg?h':'dergâh', 'C?ndido':'Cândido', 'Kitzb?hel':'Kitzbühel',\
     'Superman?s':"Superman's", '(n?e ':'(née ', 'Z?topek':'Zátopek', 'Salom?':'Salomé', 'Fran?ois':'François', 'Guy-Blach?':'Guy-Blaché',\
     'Hor?kov?':'Horáková', 'Ant?nio':'António', '?lvares':'Álvares', 'Ic?ar Bolla?n':' Icíar Bollaín', 'M?rbult':'Mörbult', 'Midg?rd':'Midgård',\
     'H?j?':'Hōjō', '(K?ki)':'(Kôki)', 'N?dasdy':'Nádasdy', 'Ib??ez':'Ibáñez', 'Nausica?':'Nausicaä', 'Guant?namo':'Guantánamo', 'h?mophage':'hæmophage',\
     'Sl?born':'løborn', 'Ch?nin':'Chûnin', 'Bat?':'Batô', 'C?lestine':'Célestine','G? Mifune':'Gô Mifune', 'T?ngere':'Tángere' }

for i in tqdm(range(len(dt))):
  dt.Overview[i] = dt.Overview[i].replace("\\'", "'")

start, step = 240, 10
dt.loc[ind, ['Title', 'Year', 'Overview']][start:start+step].values
print(dt.loc[(dt.Overview.str[-1]!='?') &(dt.Overview.str.contains('\?')),['Title', 'Overview']].values.shape)
dt.loc[(dt.Overview.str[-1]!='?') &(dt.Overview.str.contains('\?')),['Title', 'Overview']].values
dt.to_csv('movies_pr.csv', index=False)



"""### full cast"""

dt = pd.read_csv('movies_pr.csv')
dt.head(1)
dt.shape

names = ['Url', 'Directed', 'Cast']
df = pd.concat([pd.read_csv('actors.csv', names=names),\
                pd.read_csv('actors1.csv', names=names),\
                pd.read_csv('actors2.csv', names=names)])
df.reset_index(drop=True, inplace=True)
df.shape

dt['Directed'] = [0]*len(dt)
dt['Cast'] = [0]*len(dt)

for i in tqdm(range(len(dt))):
    if dt.Url[i] != df.Url[i]:
        print(i)
        break

dt.Directed[:11986] = df.Directed[:11986]
dt.Cast[:11986] = df.Cast[:11986]   
dt.Directed[11996:] = df.Directed[11986:]
dt.Cast[11996:] = df.Cast[11986:]

dt.Type[11986], dt.Episodes[11986] = 'TV Mini Series', 3
dt.Directed[11986], dt.Cast[11986] = np.nan, np.nan
dt.Directed[11987] = 'Henry Hathaway'
dt.Cast[11987] = 'Tyrone Power, Orson Welles, Cécile Aubry, Jack Hawkins, Michael Rennie, Finlay Currie, Herbert Lom, Mary Clare, Robert Blake, Alfonso Bedoya, Gibb McLaughlin, James Robertson Justice, Henry Oscar, Laurence Harvey, Itto Bent Lahcen, Rufus Cruickshank, Peter Drury, Valéry Inkijinoff'
dt.Directed[11988] = 'Alice Winocour'
dt.Cast[11988] = 'Vincent Lindon, Stéphan Wojtowicz, Soko, Chiara Mastroianni, Olivier Rabourdin, Roxane Duran, Lise Lamétrie, Ange Ruzé, Grégory Gadebois, Valentine Herrenschmidt, Aliénor de Mézamat, Audrey Bonnet, Jeanne Cohendy, Jean-Claude Baudracco, Julie Ravix, Victoire Gonin-Labat, Jacob Vouters, Damien Bonnard'
dt.Directed[11989] = 'Patrick McGoohan'
dt.Cast[11989] = 'Patrick McGoohan, Angelo Muscat, Peter Swanwick, Peter Brace, Leo McKern, Christopher Benjamin, Michael Miller, Alexis Kanner, Bill Cummings, Frank Maher, Patrick Cargill, Colin Gordon, Kenneth Griffith, Georgina Cookson, Harold Berens, John Cazabon, Bee Duffell, John Maxim'
dt.Directed[11990] = np.nan
dt.Cast[11990] = 'Mark Cousins, Juan Diego Botto, Aleksandr Sokurov, Norman Lloyd, Lars von Trier, Paul Schrader, Haskell Wexler, Woo-Ping Yuen, Robert Towne, Samira Makhmalbaf, Jean-Michel Frodon, Stanley Donen, Sharmila Tagore, Mani Kaul, Youssef Chahine, Kyôko Kagawa, Donald Richie, Gaston Kaboré'
dt.Directed[11991] = 'George Schaefer'
dt.Cast[11991] = 'Anthony Hopkins, Richard Jordan, Cliff Gorman, James Naughton, Michael Lonsdale, Martin Jarvis, Michael Kitchen, Andrew Ray, Piper Laurie, Susan Blakely, Robert Austin, Geoffrey Bateman, Graham Bishop, Kevin Bishop, Nathalie Boulmer, Yves Brainville, Jane Carr, Georges Corraface'
dt.Directed[11992] = 'Raoul Peck'
dt.Cast[11992] = 'August Diehl, Stefan Konarske, Vicky Krieps, Olivier Gourmet, Hannah Steele, Alexander Scheer, Hans-Uwe Bauer, Michael Brandner, Ivan Franek, Peter Benedict, Niels-Bruno Schmidt, Marie Meinzenbach, Wiebke Adam, Aran Bert, Ronald Beurms, Cédric Boulanger, Ulrich Brandhoff, Sarah Christoyannis'
dt.Directed[11993] = np.nan
dt.Cast[11993] = 'Christa Théret, Jannis Niewöhner, Alix Poisson, Jean-Hugues Anglade, Miriam Fussenegger, Stefan Pohl, Nicolas Wanczycki, Sylvie Testud, André Penvern, Raphaël Lenglet, Johannes Krisch, Fritz Karl, Harald Windisch, Thierry Piétra, Lili Epply, Max Baissette de Malglaive, Sebastian Blomberg, Christoph Luser'
dt.Directed[11994] = 'James Ivory'
dt.Cast[11994] = 'Nick Nolte, Greta Scacchi, Gwyneth Paltrow, Estelle Eonnet, Thandiwe Newton, Seth Gilliam, Todd Boyce, Nigel Whitmey, Nicolas Silberg, Catherine Samie, Lionel Robert, Stanislas Carré de Malberg, Jean Rupert, Yvette Petit, Paolo Mantini, Frédéric van den Driessche, Humbert Balsan, Nichel Rois'
dt.Directed[11995] = 'Stéphanie Di Giusto'
dt.Cast[11995] = 'Soko, Gaspard Ulliel, Mélanie Thierry, Lily-Rose Depp, François Damiens, Louis-Do de Lencquesaing, Amanda Plummer, Denis Ménochet, Charlie Morgan, Tamzin Merchant, William Houston, Bert Haelvoet, Camille Rutherford, Laurent Manzoni, Matilda Kime, Christian Erickson, Nicolas Helpiquet, Daniel Kramer'

dt.drop(['Directed by', 'Starring'], axis=1, inplace=True)
dt.head(1)

dt[11995:12000]

ind = []
for i in tqdm(range(len(dt))):
    if str(dt.Directed[i]) != 'nan':
        if '(' in dt.Directed[i]:
            dir = dt.Directed[i].split(', ')
            dir = [t.split('(')[0] for t in dir if '(' in t]
            a = ', '.join(dir)
            dt.Directed[i] = a

dt[dt.Type.str.contains('Series')][10:15]

dt.to_csv('movies_pr.csv', index=False)

"""##Images"""

y = []
for i in tqdm(range(len(dt))):
    t = dt.Title[i]
    t = re.sub("[A-z0-9?\/\\';:'\|* ]", '', t)
    for j in t:
        if j not in y:
            y.append(j)

dt['Images'] = [0]*len(dt)

k = []
for i in tqdm(range(len(dt))):
    if str(dt.Year[i]) != 'nan':
        y = '_' + re.sub('[\(\)]', '', str(dt.Year[i])).strip()
    else: y = ''
    im = re.sub("[?\/\\';:\*,!.]", '', dt.Title[i]).replace('"', '')
    im = im.replace('½', '1_2').replace('²', '2').replace('·', '-').replace('³', '3')
    im = '_'.join(im.lower().split(' ')) + y +'.jpg'
    if im in k:
        im = im[:-4] + '_1.jpg'
        dt['Images'][i] = im
        k.append(im)
    else:
        dt['Images'][i] = im
        k.append(im)

dt[dt.Title=='The Cursed']

dt.to_csv('movies_pr.csv', index=False)

dt[dt.duplicated('Images')]



"""#Tables preparing

##movies
"""

films = dt.copy()
films.drop(['Genre(s)', 'Cast', 'Directed', 'Url'], axis=1, inplace=True)

n = [str(i) for i in range(10)]
ind = []
flag = False
for i in tqdm(range(len(films))):
    line = films.Title[i]
    line = re.sub(r"[\W]", '', line)
    line = re.sub(r"[ÀÁÂÄÅÆ]", 'A', line).replace('Ç', 'C').replace('Þ', 'Th')
    line = re.sub(r"[ÈÉÊ]", 'E', line).replace('Í', 'I').replace('ß', 'Ss')
    line = re.sub(r"[ÒÓÔÕÖØ]", 'O', line).replace('Î', 'I')
    line = re.sub(r"[ÚÜ]", 'U', line).replace('Ý', 'Y')
    l_part = re.sub(r'(a|à|á|â|ã|ä|å|æ|e|è|é|ê|ë|i|ì|í|î|ï|o|ò|ó|ô|õ|ö|ő|ø|u|ù|ú|ū|û|ü|y|ý|ÿ)', '', line).lower()
    l_part = l_part.replace('þ', 'th').replace('ñ', 'n').replace('ð', 'd').replace('½', '12').replace('ç', 'c')
    d_part = ''.join(np.random.choice(n, 5))
    index = l_part + d_part
    if index not in ind:
        ind.append(index)
    else:
       index = index + '1'
       ind.append(index)

len(ind), films.shape

films['id'] = ind

films.to_csv('movies.csv', index=False)

df = pd.read_csv('movies.csv')
df.head(1)

dt = pd.read_csv('movies_pr.csv')

df.Year[4237] = '(2010) (1)'
df.Year[7810] = '(2021) (1)'
df.Year[7941] = '(2018) (1)'
df.Year[9930] = '(1989–1996) (1)'
df.Year[11505] = '(1976) (1)'
df.Year[11605] = '(2005) (1)'
df.Year[11745] = '(2019) (1)'
df.Year[12220] = '(1980) (1)'
df.Year[12521] = '(1990) (1)'
df.Year[13606] = '(2022) (1)'
df.Year[13650] = '(2020) (1)'
df.Year[14139] = '(2017) (1)'
df.Year[14150] = '(2020) (1)'
df.Year[14291] = '(2005) (1)'
df.Year[14315] = '(2019) (1)'
df.Year[14697] = '(2021) (1)'

df.drop(index=13309, inplace=True)

df[11605:]

print(df[df.duplicated(['Title', 'Year'])].shape)
df[df.duplicated(['Title', 'Year'])]

dt.to_csv('movies_pr.csv', index=False)

df.to_csv('movies.csv', index=False)

"""##stars"""

dt = pd.read_csv('Prmovies10.csv')

st =[]
for i in tqdm(range(len(dt))):
    if str(dt['Cast'][i]) != 'nan':
        st = st + str(dt['Cast'][i]).split(', ')
    if  str(dt['Directed'][i]) !='nan':
        st = st + str(dt['Directed'][i]).split(', ')

s = pd.read_csv('Stars.csv')
s.head(1)

stars = list(set(st))
# stars.remove('nan')
stars.sort(key=lambda x: x.split()[-1])
len(stars)

star = pd.DataFrame({'id':[0]*len(stars), 'name':stars})
star.head(2)

for i in tqdm(range(len(star))):
    if star.name[i] in s.name.values:
        ind = s[s.name == star.name[i]].index[0]
        star.id[i] = s.id[ind]

n = [str(i) for i in range(10)]
ind = []
flag = False
for i in tqdm(range(len(star))):
    line = str(star.name[i])
    line = re.sub(r"[&?.,\- \'!\$\(\) ’]", '', line)
    line = re.sub(r"[ÀÁÂÄÅÆ]", 'A', line).replace('Ç', 'C').replace('Þ', 'Th')
    line = re.sub(r"[ÈÉÊ]", 'E', line).replace('Í', 'I').replace('ß', 'Ss')
    line = re.sub(r"[ÒÓÔÕÖØ]", 'O', line).replace('Î', 'I')
    line = re.sub(r"[ÚÜ]", 'U', line).replace('Ý', 'Y')
    l_part = re.sub(r'(a|à|á|â|ã|ä|å|æ|e|è|é|ê|ë|i|ì|í|î|ï|o|ò|ó|ô|õ|ö|ő|ø|u|ù|ú|ū|û|ü|y|ý|ÿ)', '', line).lower()
    l_part = l_part.replace('þ', 'th').replace('ñ', 'n').replace('ð', 'd').replace('ß', 'ss').replace('ç', 'c')
    d_part = ''.join(np.random.choice(n, 5))
    index = l_part.lower() + d_part
    if index not in ind and index not in star.id.values:
        ind.append(index)
    else:
        index = index + '1'
        ind.append(index)

len(ind), star.shape

star.id = ind

start, step  = 112210, 5
star[start:start+step]

star.to_csv('stars.csv', index=False)

"""## movies-genres"""

k, ind = 0, []
for i in range(len(dt)):
    p = dt['Genre(s)'][i].split(',')
    k += len(p)
    for j in p:
        if j.strip() not in ind:
            ind.append(j.strip())
ind = sorted(ind)

g = pd.DataFrame({'id':len(ind), 'genre':ind})

n = [str(i) for i in range(10)]
ind = []
flag = False
for i in tqdm(range(len(g))):
    line = str(g.genre[i])
    line = re.sub(r"[aouiye\-]", '', line).lower()
    d_part = ''.join(np.random.choice(n, 2))
    g.id[i] = line + d_part

g.to_csv('genres.csv', index=False)

mov_gen = pd.DataFrame({'movie_id':[0]*k, 'genre_id':[0]*k})

gen, mov = [], []
for i in tqdm(range(len(dt))):
    jh = dt['Genre(s)'][i].split(', ')
    for j in range(len(jh)):
        x = g.id[g.genre == jh[j]].values[0]
        gen.append(x)
        mov.append(movies.id[i])

mov_gen['movie_id'] = mov
mov_gen['genre_id'] = gen
mov_gen.head(5)

mov_gen.to_csv('movies-genres.csv', index=False)

"""##movies-actors"""

dt = pd.read_csv('movies_pr.csv')
movies = pd.read_csv('movies.csv')
star = pd.read_csv('movie-stars.csv')
mov_act = pd.read_csv('movie-actor.csv')

fwna = dt[['Title', 'Cast']][dt.Cast.isna()==False]
fwna['id'] = movies['id']
fwna.reset_index(drop=True, inplace=True)
fwna.tail(5)

k = 0
for i in tqdm(range(len(fwna))):
    k += len(fwna['Cast'][i].split(','))
k

act, mov, ind = [], [], []
for i in tqdm(range(len(fwna))):
    jh = fwna['Cast'][i].split(', ')
    act = act + jh
    mov = mov + [fwna.id[i]]*len(jh)
    ind = ind + [j for j in range(1, len(jh) + 1)]

m_a = pd.DataFrame({'movie_id':mov, 'actor_id':act, 'rating': ind})
print(m_a.shape)
m_a.drop_duplicates(['movie_id', 'actor_id'], keep='first', inplace=True)
m_a.reset_index(drop=True, inplace=True)
m_a.shape

mov_act['rating'] = m_a['rating']

for i in tqdm(range(len(mov_act))):
    mov_act.actor_id[i] = star.id[star.name==mov_act.actor_id[i]].values[0]

m_a[m_a.actor_id=='Masako Watanabe']

mov_act[-5:], m_a[-5:]

print(mov_act.shape)
mov_act.drop_duplicates(inplace=True, )
mov_act.shape

st = 269006
m_a[st:st+5], mov_act[st:st+5]



mov_act.to_csv('movie-actor.csv', index=False)

"""##movie-director"""

fdwna = dt[['Title', 'Directed by']][dt['Directed by'].isna()==False]
fdwna['id'] = movies['id']
fdwna.reset_index(drop=True, inplace=True)
fdwna.tail(5)

k, c = 0, 0
for i in range(len(fdwna)):
    k += len(fdwna['Directed by'][i].split(','))
    if len(fdwna['Directed by'][i].split(',')) > c:
        c = len(fdwna['Directed by'][i].split(','))
        print(f'{i}) {fdwna.Title[i]} | {len(fdwna["Directed by"][i].split(", "))} | {fdwna["Directed by"][i]}')
k

mov_dir = pd.DataFrame({'movie_id':[0]*k, 'director_id':[0]*k, 'rating':[0]*k})

dir, m, r = [], [], []
for i in tqdm(range(len(fdwna))):
    jh = fdwna['Directed by'][i].split(', ')
    for j in range(len(jh)):
        x = star.id[star.name == jh[j]].values[0]
        dir.append(x)
        m.append(fdwna.id[i])
    r = r + [y for y in range(1, len(jh) + 1)]
len(m), len(dir), len(r)

mov_dir['movie_id'] = m
mov_dir['director_id'] = dir
mov_dir['rating'] = r
mov_dir.tail(5)

print(mov_dir.shape)
mov_dir.drop_duplicates(['movie_id', 'director_id'], inplace=True)
mov_dir.shape

mov_dir.tail(15)

fdwna.tail(10)

mov_dir.to_csv('movies-directors.csv', index=False)

"""#database"""

fm = pd.read_csv('movies.csv')
s = pd.read_csv('movie-stars.csv')
m_d = pd.read_csv('movies-directors.csv')
m_a = pd.read_csv('movie-actor.csv')
f_g = pd.read_csv('movies-genres.csv')
g = pd.read_csv('genres.csv')

print(f'Movie: {fm.head(1)}\n Stars: {s.head(1)}\n Genres: {g.head(1)}\n Mov-gen{f_g.head(1)}\n Mov-div{m_d.head(1)}\n Mov-act{m_a.head(1)}')

conn = sqlite3.connect('Movies.db')
conn.execute("PRAGMA foreign_keys = 1")
conn.commit()

curr = conn.cursor()
curr.execute("CREATE TABLE IF NOT EXISTS movies(\
    id TEXT PRIMARY KEY NOT NULL,\
    title TEXT NOT NULL, mtype TEXT, year TEXT,\
    age_limit TEXT, episodes INT, time TEXT, imdb REAL,\
    metascore INT, overview TEXT NOT NULL,\
    box_office TEXT, images TEXT NOT NULL)")
conn.commit()

curr.execute("CREATE TABLE IF NOT EXISTS mov_gen( \
              movie_id TEXT NOT NULL,\
              genre_id TEXT NOT NULL,\
              PRIMARY KEY (movie_id, genre_id))")
curr.execute("CREATE TABLE IF NOT EXISTS genres( \
              id TEXT PRIMARY KEY NOT NULL,\
              genre TEXT NOT NULL)")
curr.execute("CREATE TABLE IF NOT EXISTS mov_dir( \
              movie_id TEXT NOT NULL,\
              dir_id TEXT NOT NULL, rating INT NOT NULL,\
              PRIMARY KEY (movie_id, dir_id))")
curr.execute("CREATE TABLE IF NOT EXISTS stars( \
              id TEXT PRIMARY KEY NOT NULL,\
              name TEXT NOT NULL)")
curr.execute("CREATE TABLE IF NOT EXISTS mov_act( \
              movie_id TEXT NOT NULL,\
              act_id TEXT NOT NULL, rating INT NOT NULL,\
              PRIMARY KEY (movie_id, act_id))")
conn.commit()

curr.execute("SELECT name FROM sqlite_master WHERE type='table';")
curr.fetchall()

"""##stars"""

conn = sqlite3.connect('Movies.db')
curr = conn.cursor()
curr.execute('Delete from mov_dir')
x = curr.fetchall()
curr.close()
conn.commit()
x

conn = sqlite3.connect('Movies.db')
curr = conn.cursor()
curr.execute('Select count(*) from stars')
x = curr.fetchall()
curr.close()
x

s.shape

step=10000
l = [[step*k+j-step for k in range(2)] for j in range(1*step, 14*step+1, step)]
l[-1][1] = s.shape[0]
l

# Stars table
def f(a):
    start, stop = a
    conn = sqlite3.connect('Movies.db')
    for j in tqdm(range(start, stop), desc="Progress" ):
        curr = conn.cursor()
        curr.execute("INSERT INTO stars VALUES(?, ?);", \
                (s.id[j], s['name'][j]))
        curr.close()
    conn.commit()
try:
    print(Pool(1).map(f, l))
except (OperationalError, IntegrityError):
    pass
display(Audio('1.wav', autoplay=True))

"""## movies"""

fm.columns.tolist()

fm.shape

step=10000
l = [[step*k+j-step for k in range(2)] for j in range(1*step, 2*step+1, step)]
l[-1][1] = fm.shape[0]
l

conn = sqlite3.connect('Movies.db')
curr = conn.cursor()
curr.execute('Delete from movies')
x = curr.fetchall()
curr.close()
conn.commit()
x

conn = sqlite3.connect('Movies.db')
curr = conn.cursor()
curr.execute('Select * from movies Limit 5')
x = curr.fetchall()
curr.close()
x

# Films table
def f(a):
    start, stop = a
    conn = sqlite3.connect('Movies.db')
    for i in tqdm(range(start, stop), desc="Progress"):
        curr = conn.cursor()
        try:
            epi = int(fm.Episodes[i])
        except ValueError:
            epi = fm.Episodes[i]
        try:
            meta = int(fm.Metascore[i])
        except ValueError:
            meta = fm.Metascore[i]
        curr.execute("INSERT INTO movies VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);", \
                (fm.id[i], fm.Title[i], fm['Type'][i], fm.Year[i], fm['Age Limit'][i],\
                epi, fm.Time[i], fm.IMDB[i], meta, fm['Overview'][i],\
                fm['Box Office'][i], fm.Images[i], ))
        curr.close()
    conn.commit()
try:
    print(Pool(1).map(f, l))
except (OperationalError, IntegrityError):
    pass
display(Audio('1.wav', autoplay=True))

"""##genres

"""

g.columns.tolist()

# Genre table
conn = sqlite3.connect('Movies.db')
curr = conn.cursor()
for j in tqdm(range(len(g)), desc="Progress" ):
  curr.execute("INSERT INTO genres VALUES(?, ?);", \
              (g.id[j], g.genre[j]))
  conn.commit()
curr.close()

"""##mov_gen"""

f_g.columns.tolist()

f_g.shape

step=10000
l = [[step*k+j-step for k in range(2)] for j in range(1*step, 5*step+1, step)]
l[-1][1] = f_g.shape[0]
l

# Genre_ids table
def f(a):
    start, stop = a
    conn = sqlite3.connect('Movies.db')
    for j in tqdm(range(start, stop), desc="Progress" ):
        curr = conn.cursor()
        curr.execute("INSERT INTO mov_gen VALUES(?, ?);", \
              (f_g.movie_id[j], f_g.genre_id[j]))
        curr.close()
    conn.commit()
try:
    print(Pool(1).map(f, l))
except (OperationalError, IntegrityError):
    pass
display(Audio('1.wav', autoplay=True))

"""##mov_act"""

conn = sqlite3.connect('Movies.db')
curr = conn.cursor()
curr.execute('Delete from mov_act')
x = curr.fetchall()
curr.close()
conn.commit()
x

conn = sqlite3.connect('Movies.db')
curr = conn.cursor()
curr.execute('Select * from mov_act limit 5')
x = curr.fetchall()
curr.close()
x

m_a.shape

m_a.columns.tolist()

step=10000
l = [[step*k+j-step for k in range(2)] for j in range(1*step, 27*step+1, step)]
l[-1][1] = m_a.shape[0]
l

def f(a):
    start, stop = a
    conn = sqlite3.connect('Movies.db')
    for j in tqdm(range(start, stop), desc="Progress" ):
        curr = conn.cursor()
        curr.execute("INSERT INTO mov_act VALUES(?, ?, ?);", \
                (m_a.movie_id[j], m_a.actor_id[j], int(m_a.rating[j])))
        curr.close()
    conn.commit()
try:
    print(Pool(1).map(f, l))
except (OperationalError):
    pass
display(Audio('1.wav', autoplay=True))

"""##mov_dir"""

conn = sqlite3.connect('Movies.db')
curr = conn.cursor()
curr.execute('Delete from mov_dir')
x = curr.fetchall()
curr.close()
conn.commit()
x

conn = sqlite3.connect('Movies.db')
curr = conn.cursor()
curr.execute('Select count(*) from mov_dir')
x = curr.fetchall()
curr.close()
x

m_d.shape

m_d.columns.tolist()

step = 10000
l = [[step*k + j - step for k in range(2)] for j in range(1*step, 2*step + 1, step)]
l[-1][1] = m_d.shape[0]
l

# Directors table
def f(a):
    start, stop = a
    conn = sqlite3.connect('Movies.db')
    for j in tqdm(range(start, stop), desc="Progress"):
        curr = conn.cursor()
        curr.execute("INSERT INTO mov_dir VALUES(?, ?, ?);", \
                (m_d['movie_id'][j], m_d['director_id'][j], int(m_d['rating'][j])))
        curr.close()
    conn.commit()
try:
    print(Pool(1).map(f, l))
except (OperationalError, IntegrityError):
    pass
display(Audio('1.wav', autoplay=True))

"""##queries"""

sql = '''
Select *
from movies as m
inner join mov_dir as md on m.id = md.movie_id
inner join stars as s on ma.act_id = s.id
inner join mov_act as ma on m.id = ma.movie_id
inner join mov_gen as mg on m.id = mg.movie_id
inner join genres as g on mg.movie_id = m.id
Order by name, title, year, rating
limit 2
'''
conn = sqlite3.connect('Movies.db')
curr = conn.cursor()
curr.execute(sql)
x = curr.fetchall()
curr.close()
conn.commit()
for i in x:
    print(i)

sql = '''
Select title, year, imdb, metascore, rating,  name
from movies as m
inner join mov_act as ma on m.id = ma.movie_id
inner join stars as s on ma.act_id = s.id
where name like "%Selena %"
Order by name, title, year, rating
'''
conn = sqlite3.connect('Movies.db')
curr = conn.cursor()
curr.execute(sql)
x = curr.fetchall()
curr.close()
conn.commit()
for i in x:
    print(i)
