import pandas as pd

path = '/Users/zhaojianbo/Desktop/Alphabet/Week2/animelist.csv'
animelist = pd.read_csv('/Users/zhaojianbo/Desktop/Alphabet/Week2/animelist.csv')
anime = pd.read_csv('/Users/zhaojianbo/Desktop/Alphabet/Week2/anime.csv')

print(animelist.head())
print(anime.head())
