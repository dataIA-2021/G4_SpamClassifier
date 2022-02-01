from flask import Flask, render_template,request,redirect,url_for

import pandas as pd
import re
import string
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score,confusion_matrix,classification_report

app = Flask(__name__)
app.config.from_object('config') # Configuration flask

class Preprocess():
    
    # Attribut de classe
    wordsList = ['prize','urgent','free','mobile','please','cash','chat','win','txt','reply','phone','new',
        'now','call','tone','claim','won','chance']
    
    # constructeur
    def __init__(self,df):
        self.df_ = df
        self.df_['text'] = self.df_['text'].apply(self.text_cleaning_encoding)
        self.df_['length'] = self.df_['text'].str.len()
        self.df_['words_count'] = self.df_['text'].apply(lambda x: len(x.split(" ")))
        self.df_['uppercases_count'] = self.df_['text'].str.findall(r'[A-Z]').str.len()
        self.df_['lowercases_count'] = self.df_['text'].str.findall(r'[a-z]').str.len()
        self.df_['number_sequence'] = self.df_['text'].map(self.isNumberSequenceInText)
        self.df_['url_or_mail'] = self.df_['text'].map(self.isUrlMailInText)
        self.df_['special_char_count'] = self.df_['text'].map(self.countSpecialCharInText)
        self.df_['words_fateful_count'] = self.df_['text'].map(self.countWordsInText)
        
        # Suppresion de la colonne text, inutile pour le ML
        self.df_.drop('text',axis=1,inplace=True)
        
        # Preparation des features et de la target pour entrainement
        # ------------------------
        # Est-ce le dataframe d'entrainement ou celui de la prediction ?
        self.y = None # Pas de target si c'est pour une prediction
        if 'target' in self.df_.columns:
            self.X = self.df_.drop(columns=['target','special_char_count','words_fateful_count'])
            
            # et encodage de la target (1 ou 0)
            self.lb_encod = LabelEncoder()
            self.y = self.lb_encod.fit_transform(self.df_['target'])
            
        else:
            self.X = self.df_.drop(columns=['special_char_count','words_fateful_count'])
        
        # Preparation des colonnes pour normalisation
        self.column_num = ['length','words_count','uppercases_count','lowercases_count']
        self.transfo_num = Pipeline(steps=[('scaling', MinMaxScaler())])
        #self.transfo_num = Pipeline(steps=[('imputer',SimpleImputer())])
        

    def text_cleaning_encoding(self,text):
        text = re.sub('&lt;#&gt;',"",text) #Removing square brackets from the text
        return(text) 
        
    def isNumberSequenceInText(self,txt):
        if bool(re.search("(\d{2})", txt)):
            return 1
        else:
            return 0

    def isUrlMailInText(self,txt):
        if bool(re.search("((https?:\/\/|www\.)[a-zA-Z0-9-_\.]+)|(\w+@\w+)", txt)):
            return 1
        else:
            return 0


    def countSpecialCharInText(self,txt):
        category = 0
        nbSpecialChar = len(re.findall("[$&+:;=?@#|'<>^*()%!-]", txt))
        if nbSpecialChar == 1:
            category = 1
        elif nbSpecialChar > 1 and nbSpecialChar < 7:
            category = 2
        elif nbSpecialChar > 6 and nbSpecialChar < 13:
            category = 3
        elif nbSpecialChar > 12:
            category = 4

        return category

    def countWordsInText(self,txt):
        res = 0
        nb = 0
        for i in Preprocess.wordsList:
            nb += len(re.findall(r"\b" + i + r"\b", txt))

        if nb > 3:
            res = 1
        else:
            res = 0

        return res
    
    def getDf(self):
        return self.df_
    
    def getPipeline(self):
        return self.transfo_num
    
    def getFeatures(self):
        return self.X
    
    def getTarget(self):
        return self.y

class ModelEvaluation():
    
    # constructeur
    def __init__(self,pipeline,model,listSplit):
        self.model = model
        self.pipeline = pipeline
        self.X_train = listSplit.get('X_train')
        self.X_test = listSplit.get('X_test')
        self.y_train = listSplit.get('y_train')
        self.y_test = listSplit.get('y_test')
        self.addStepModel()
        self.fitModel()
        self.predict(self.X_test)
        
    def addStepModel(self):
        self.pipeline.steps.append(['model',self.model])
        print(self.pipeline)
        
    def fitModel(self):
        self.pipeline.fit(self.X_train, self.y_train)
    
    def predict(self,X_test):
        self.y_pred = self.pipeline.predict(X_test)
        
    def predict_proba(self,X_test):
        print(self.pipeline.predict_proba(X_test)) 
        
    def getScore(self):
        self.score = accuracy_score(self.y_test, self.y_pred)
        return round(self.score, 5)
        
    def getTargetPred(self):
        return self.y_pred
    
    def getModel(self):
        return self.model


df = pd.read_csv('spam.csv',encoding = "latin-1")
# Nettoyage du dataframe
df.drop(['Unnamed: 2', 'Unnamed: 3','Unnamed: 4'],axis=1,inplace=True)
df.rename(columns={"v1": "target", "v2": "text"},inplace=True)
df.drop_duplicates(inplace=True) # Suppression des doublons
preproc = Preprocess(df)
df = preproc.getDf()
X = preproc.getFeatures()
y = preproc.getTarget()
pipeline = preproc.getPipeline()

# Preparation pour entrainement
X_train, X_test, y_train, y_test = train_test_split(X, y,stratify=y, test_size=0.2, random_state=42)

listSplit = {
    'X_train' : X_train,
    'X_test' : X_test,
    'y_train' : y_train,
    'y_test' : y_test
}

DecisionTree = ModelEvaluation(pipeline,RandomForestClassifier(),listSplit)

score = DecisionTree.getScore()

@app.route('/', methods=['GET'])
def index():
    form = request.args.get('myForm')
    return render_template('index.html', data=score)
    
if __name__ == "__main__":
    app.run()
    
