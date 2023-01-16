from flask import Flask,request, url_for, redirect, render_template
import pickle
import numpy as np
import keras
import pandas as pd
import forest_fire as model

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


app = Flask(__name__)


# model = keras.models.load_weights('model_esp')
#  load weights
ml_model = keras.models.load_model('model_esp')
# model.load_weights('model_esp.h5')

amino_map = {'A': 1,
 'K': 2,
 'L': 3,
 'G': 4,
 'E': 5,
 'P': 6,
 'V': 7,
 'D': 8,
 'I': 9,
 'W': 10,
 'R': 11,
 'Q': 12,
 'F': 13,
 'M': 14,
 'T': 15,
 'H': 16,
 'S': 17,
 'N': 18,
 'Y': 19,
 'C': 20,
 None: 0}

def encode(df):
    df = df.drop(columns=['protein_sequence'])
    return df

def preprocessing(list_data):
    # Convert the data into a pandas dataframe of 1 row with headers
    df = pd.DataFrame([list_data], columns=['protein_sequence', 'pH'])
    sequences_df = pd.DataFrame(df['protein_sequence'].apply(list).tolist())
    sequences_df = sequences_df.replace(amino_map)
    df.join(sequences_df)
    df = encode(df)
    return df

def preprocessingPD(df):
    df["protein_sequence_len"] = df["protein_sequence"].apply(lambda x: len(x))
    # make to list till 1024 columns and add None
    # df['protein_sequence'] = df['protein_sequence'].apply(lambda x: x + [None] * (1024 - len(x)))
    listx = df['protein_sequence'].apply(list).tolist()
    listx = [x + [None] * (1024 - len(x)) for x in listx]
    sequences_df = pd.DataFrame(listx)
    sequences_df.to_csv('sequences_df.csv')

    # sequences_df = pd.DataFrame(if len(df['protein_sequence'].apply(list).tolist()) < 1024: df['protein_sequence'].apply(list).tolist().append(None) for i in range(1024-len(df['protein_sequence'].apply(list).tolist())) else df['protein_sequence'].apply(list).tolist())
    # sequences_df = pd.DataFrame(df['protein_sequence'].apply(list).tolist())

    sequences_df = sequences_df.replace(amino_map)
    df = df.join(sequences_df)
    df = encode(df)
    return df



def preprocess_model_inputs(X):
    X0 = X['pH'].values
    X1 = X.drop(columns=['pH']).values
    return X0, X1

@app.route('/')
def hello_world():
    return render_template("forest_fire.html")

@app.route('/predict',methods=['POST','GET'])
def predict():
    df = pd.read_csv('x_val.csv')
    features=[x for x in request.form.values()]
    # features[-1] = float(features[-1])
    x = model.predict(features[0], features[1])
    print(x)
    output='{0:.{1}f}'.format(x, 1)
    return render_template('forest_fire.html',pred='The thermal stability of your mutated chemical is {}'.format(output),bhai="kuch karna hain iska ab?")

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)
