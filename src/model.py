from gensim import corpora
from gensim.models import LdaModel
from gensim.models.coherencemodel import CoherenceModel
import mlflow
import numpy as np
import pandas as pd

def set_up_model(processed_df):
    # Create dictionary
    dictionary = corpora.Dictionary(processed_df['clean text'])
    # Apply filter to remove unknown words
    dictionary.filter_extremes(no_below = 15,  no_above = 0.5)
    # Convert documents to BoW
    corpus = [dictionary.doc2bow(doc) for doc in processed_df['clean text']]

    return dictionary, corpus

def coherence_grid(dictionary, corpus, texts, limit, start, step, random_state, chunksize, alpha):

    coherence_values = []
    model_list = []

    for num_topics in range(start, limit, step):

        model = LdaModel(corpus = corpus,
                     id2word = dictionary,
                     num_topics = num_topics,
                     random_state = random_state,
                     update_every = 1,
                     chunksize=chunksize,
                     passes = 10,
                     alpha = alpha,
                     per_word_topics = True)
    
    
        model_list.append(model)


        coherencemodel = CoherenceModel(model=model,
                                        texts=texts,
                                        dictionary=dictionary,
                                        coherence='c_v',
                                        window_size=2,
                                        topn=5)

        coherence_values.append(coherencemodel.get_coherence())


    return model_list, coherence_values


def get_best_model(model_list, coherence_list):
    idx_best = np.argmax(coherence_list)
    return model_list[idx_best], coherence_list[idx_best]

def train_model(experiment_name, 
                corpus, 
                dictionary, 
                processed_df,
                start, 
                limit, 
                step, 
                random_state,
                chunksize,
                version,
                alpha = 'auto'):
    

    mlflow.set_experiment(experiment_name)

    # Start run
    with mlflow.start_run():

        # Get models and coherence values
        model_list, coherence_values = coherence_grid(dictionary, corpus, processed_df['clean text'], limit, start, step, random_state, chunksize, alpha)

        # Find the best model & coherence
        model, coherence_score = get_best_model(model_list, coherence_values)

        # Log Params
        mlflow.log_param("chunksize", chunksize)
        mlflow.log_param("alpha", alpha)

        # Log Model & Coherence
        model_path = f'./../models/model_topic_{version}'
        model.save(model_path)
        mlflow.log_artifact(model_path)
        mlflow.log_metric("coherence_score", coherence_score)


    return model

def get_results(model, processed_df, corpus, show_results = True):

    if show_results:
        # Show the most important words of each topic
        topics = model.print_topics(num_words = 10)
        for topic in topics:
            print(topic)

    dominant_topics = []
    for i, row_list in enumerate(model[corpus]):
        row = row_list[0] if model.per_word_topics else row_list            
        row = sorted(row, key=lambda x: (x[1]), reverse=True)
        topic_num, prop_topic = row[0]
        dominant_topics.append((i, topic_num, prop_topic))

    # Create DataFrame
    df_topics = pd.DataFrame(dominant_topics, columns=['Document_Id', 'Dominant_Topic', 'Percentage_Contribution'])

    # Combine
    final_df = pd.concat([processed_df, df_topics], axis=1)

    return final_df

def model_run(processed_df, config, show_results):

    dictionary, corpus = set_up_model(processed_df)

    model = train_model(config.experiment_name, corpus, dictionary, processed_df, config.start, 
                        config.limit, config.step, config.random_state, config.chunksize, config.version)
    
    
    final_df = get_results(model, processed_df, corpus, show_results = show_results)

    return final_df



