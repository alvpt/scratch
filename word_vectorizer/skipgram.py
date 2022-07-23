# Simple skipgram

if __name__ == "__main__":
    
    # Corpus: Legal Dataset -> https://archive.ics.uci.edu/ml/machine-learning-databases/00239/corpus.zip

    import numpy as np
    import pandas as pd
    from processing import preprocess, training_data

    df = pd.read_json('News_Category_Dataset_v2.json', lines=True)
    df = pd.DataFrame(df['short_description']).head(200)

    df['cleanText'] = df['short_description'].map(lambda s:preprocess(s)).str.split() 

    print(df['cleanText'])

    df['bigrams'] = df['cleanText'].map(lambda s:training_data(s))

    print(df.bigrams.head())


