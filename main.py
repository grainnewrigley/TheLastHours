import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager
import matplotlib.pyplot as plt
import numpy as np
import spacy
import networkx as nx
import os
import re

#Create Driver
service = Service(ChromeDriverManager().install())
driver = webdriver.Chrome(service=service)

#Go to character page
page_url = "https://shadowhunters.fandom.com/wiki/Category:The_Last_Hours_characters"
driver.get(page_url)

#Do not need to click on accept cookies

#Find books
book_categories = driver.find_elements(By.CLASS_NAME, 'category-page__member-link')


books = []
for category in book_categories:
    book_url = category.get_attribute('href')
    book_name = category.text
    books.append({'book_name': book_name, 'url': book_url})

character_list = []

for book in books:
    #go to book page
    driver.get(book['url'])

    character_elems = driver.find_elements(By.CLASS_NAME, 'category-page__member-link')

    for elem in character_elems:
        character_list.append({'book': book['book_name'],'character': elem.text}) 

#Create data fram of characters
character_df = pd.DataFrame(character_list)

character_df['character'].value_counts().plot(kind="bar")

#load spacy english language model
NER = spacy.load('en_core_web_sm')
NER.max_length = 2_000_000


#get all book fiules in the data directory
files = sorted(os.listdir('data'))
all_books = [b for b in os.scandir('data') if '.txt' in b.name]

book = all_books[2]
book_text = open(book).read()
book_doc = NER(book_text)

#remove text with brackets
character_df['character'] = character_df['character'].apply(lambda x: re.sub(r"\(.*?\)", '', x))
character_df['character_firstname'] = character_df['character'].apply(lambda x: x.split(' ', 1)[0])

pd.set_option('display.max_rows', None)
character_df

#get named entity per sentence
sent_entity_df = []

#loop through sentences, stored names entity list for each sentence
for sent in book_doc.sents:
    entity_list = [ent.text for ent in sent.ents]
    sent_entity_df.append({'sentence':sent,'entities':entity_list})

sent_entity_df = pd.DataFrame(sent_entity_df)

#filter non character entities

def filter_entity(ent_list, character_df):
    return [ent for ent in ent_list
            if ent in list(character_df.character)
            or ent in list(character_df.character_firstname)]

filter_entity(['James','2'], character_df)

sent_entity_df['character_entities'] = sent_entity_df['entities'].apply(lambda x: filter_entity(x, character_df))

#filter out sentences without characters
sent_entity_df_filtered = sent_entity_df[sent_entity_df['character_entities'].map(len) > 0]
sent_entity_df_filtered.head(10)

#take only first name
sent_entity_df_filtered['character_entities'] = sent_entity_df_filtered['character_entities'].apply(lambda x: [item.split()[0] for item in x])

pd.reset_option('^display.', silent=True)
sent_entity_df_filtered

#create realtionships

window_size = 2
relationships = []

for i in range (sent_entity_df_filtered.index[-1]):
    end_i = min(i+5, sent_entity_df_filtered.index[-1])
    char_list = sum((sent_entity_df_filtered.loc[i: end_i].character_entities), [])

    #remove duplicate characters
    char_unique = [char_list[i] for i in range(len(char_list))
                  if (i==0) or char_list[i] !=char_list[i-1]]

    for idx, a in enumerate(char_unique[:-1]):
        b = char_unique[idx + 1]
        relationships.append({'source':a, 'target':b})

relationship_df = pd.DataFrame(relationships)

pd.set_option('display.max_rows', None)

#combine a->b and b->a
relationship_df = pd.DataFrame(np.sort(relationship_df.values, axis = 1), columns = relationship_df.columns)

relationship_df['value'] = 1
relationship_df = relationship_df.groupby(['source','target'], sort=False, as_index=False).sum()

relationship_df.head(10)

#graph analysis
G = nx.from_pandas_edgelist(relationship_df,
                            source = 'source',
                            target = 'target',
                            edge_attr = 'value',
                            create_using = nx.Graph())

#graph visualisation

pos = nx.kamada_kawai_layout(G)
nx.draw(G, with_labels = True, node_color = 'skyblue', edge_cmap=plt.cm.Blues, pos=pos)

plt.show

#graph visualistion using pyvis

from pyvis.network import Network

net = Network(notebook = True, width = '1000px', height = '700px', bgcolor = '#222222', font_color = 'white')

node_degree = dict(G.degree)

# Disable physics so nodes stay put
net.toggle_physics(False)

nx.set_node_attributes(G, node_degree, 'size')
net.from_nx(G)
net.show('witcher.html')        