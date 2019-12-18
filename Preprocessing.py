# Importing the libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler 
import json
from sklearn.tree import tree
from sklearn.svm import SVC

import seaborn as sns
import random
from sklearn.preprocessing import LabelEncoder ,OneHotEncoder
from sklearn import preprocessing
from sklearn.preprocessing import Imputer
from sklearn import linear_model
from sklearn import metrics
from  sklearn.decomposition import PCA
from sklearn.preprocessing import Normalizer
import time
from sklearn import svm

from sklearn.feature_extraction import FeatureHasher

##############################################################


def ReadFiles_ConcatenateTwoFiles():
    datapath_movie = 'tmdb_5000_movies_classification.csv'
    datapath_credits = 'tmdb_5000_credits.csv'
    dataset_movie = pd.read_csv(datapath_movie)
    dataset_credits = pd.read_csv(datapath_credits)
    del dataset_credits['title']
    del dataset_credits['movie_id']
    movie=pd.concat([dataset_movie,dataset_credits],axis = 1 )
    return movie



##############################################################
def PrepareData(movie):
    movie ['profit']=movie['revenue'] - movie['budget']
    dataClean = ['id','title','homepage','genres','keywords', 'original_language', 'original_title', 'tagline' ,'cast','crew',
             'production_companies','production_countries','release_date','revenue','budget','popularity',
             'profit','runtime','spoken_languages', 'vote_count' ,'rate']
    
    
    movie_dataset_clean = movie[dataClean]
    movie_dataset_clean.dropna(thresh=15  , inplace =True)
    return  movie_dataset_clean 


###############################################################

def SolveNanData(movie_clean):
    # tagline  475
    #numofnantagline=movie_clean['tagline'].isnull().sum()
    # 1 nan runtime 
    #numofnanruntime=movie_clean['runtime'].isnull().sum()
    #handel tagline 
    a=movie_clean['tagline']
    rand=random.randrange(0, len(movie_clean))
    movie_clean['tagline']= movie_clean['tagline'].fillna(a[rand])

    
    #handel runtime 
    imputer = Imputer(missing_values = 'NaN',strategy='mean' , axis=0  )
    imputer=imputer.fit(movie_clean[['runtime']])
    movie_clean['runtime'] =imputer.transform(movie_clean[['runtime']])
    
    return movie_clean


##################################################################

def Handelrelease_date(movie):
   
    year = pd.DatetimeIndex(movie['release_date']).year
    month = pd.DatetimeIndex(movie['release_date']).month
    day = pd.DatetimeIndex(movie['release_date']).day
    movie['year'] = year.astype(np.float64)
    movie['month'] = month.astype(np.float64)
    movie['day'] = day.astype(np.float64)
    del (movie['release_date'] )

    
    
    return movie


##################################################################
    
def Feature_Encoder(X,cols):
    for c in cols:
        lbl = LabelEncoder()
        lbl.fit(list(X[c].values))
        X[c] = lbl.transform(list(X[c].values))
    return X

def names_genres(a):
    y= [x['name'] for x in a]
    
    s='&'
    return s.join(y)
def names_country(a):
    y= [x['iso_3166_1'] for x in a]
    
    s='&'
    return s.join(y)

def names_id(a):
    y= [x['id'] for x in a]
    s='&'
    return s.join(y)



def check (x):
     if(x==True):
          x =1
         
     else :
          
          x =0
     return x

#############################################################
def EncodeNonNumericForGeners_to_Numeric(movie):
    
    movie['genres'] = [json.loads(i) for i in movie['genres'] if i]   
    movie['genres'] = movie['genres'].apply(names_genres)
    a= movie['genres']
   
    st =set()
    for i in a.str.split('&'):
        st=set(st.union(i))
    
    st = list(st)
    dummy_geners = movie[['title']]
        
    for i in st:
        dummy_geners[i]=movie['genres'].str.contains(i).apply(check)
        
    del (dummy_geners[''])
    
    del (dummy_geners['title'] )
    movie   = pd.concat([movie,dummy_geners],axis = 1 )
    del (movie['genres'])
    
    return movie
    
    











'''
def PCA_Function(movie):
    
    
    x = movie.iloc[:, 0:20].values
    y = movie['rate']
    x = StandardScaler().fit_transform(x)
    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(x)
    print(pca.explained_variance_ratio_)
    return principalComponents

'''
    


#def EncodeNonNumericForKeywords_to_Numeric(movie):
def EncodeNonNumericForcountry_to_Numeric(movie):
    
    movie['production_countries'] = [json.loads(i) for i in movie['production_countries'] if i]   
    movie['production_countries'] = movie['production_countries'].apply(names_country)
    a= movie['production_countries']
   
    st =set()
    a=movie['production_countries']
    for i in a.str.split('&'):
        st=set(st.union(i))
    
    st = list(st)
    dummy_geners = movie[['title']]
        
    for i in st:
        dummy_geners[i]=movie['production_countries'].str.contains(i).apply(check)
        
    del (dummy_geners[''])
    
    del (dummy_geners['title'] )
    movie   = pd.concat([movie,dummy_geners],axis = 1 )
    del (movie['production_countries'])
    
    return movie   






movie = ReadFiles_ConcatenateTwoFiles()
movie= PrepareData(movie)   
movie = SolveNanData(movie)
movie = Handelrelease_date(movie)


movie = Feature_Encoder(movie , ['rate'])
movie =  EncodeNonNumericForGeners_to_Numeric(movie) 
movie = Feature_Encoder(movie , ['original_language'])
movie = EncodeNonNumericForcountry_to_Numeric(movie)


corr = movie.corr()
plt.subplots(figsize=(12, 8))
labels=[] #xticklabels= labels
sns.heatmap(corr, annot=True)
plt.show()


del(movie['crew'])
del(movie['id'])
del(movie[ 'title'])
del(movie[ 'tagline'])
del(movie[ 'production_companies'])
del(movie[ 'keywords'])
#del(movie['original_language'])
del(movie['original_title'])
del(movie['homepage'])
del(movie['spoken_languages'])
del(movie['cast'])

y = movie['rate']
del(movie['rate'])
x= movie.iloc[:,:].values

#x = StandardScaler().fit_transform(x)


print (x)
test_size =.20

X_train, X_test, y_train, y_test = train_test_split(pd.DataFrame(x), y, test_size = test_size, shuffle=True)




#x_word = EncodeNonNumericForKeywords_to_Numeric ( movie)
#print(movie)

#del(movie['crew'])
#del(movie['id'])
#del(movie[ 'title'])
#del(movie[ 'tagline'])
#del(movie[ 'production_countries'])
#del(movie[ 'budget'])
#del(movie['original_language'])
#del(movie['original_title'])
#del(movie['homepage'])
#del(movie['spoken_languages'])

