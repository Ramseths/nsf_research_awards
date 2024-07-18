import mlflow
import numpy as np
import pandas as pd
from gensim import corpora
from gensim.models import LdaModel
from gensim.models.coherencemodel import CoherenceModel


def set_up_model(processed_df):
    """
    This function helps to create a Dictionary & Corpus.

    Parameters
    ----------
    processed_df: DataFrame with text cleaned.

    Returns
    -------
    dictionary: Dictionary with each token/word (frequency).
    corpus: Documents converted into BoW.
    """

    # Create dictionary
    dictionary = corpora.Dictionary(processed_df['clean text'])
    # Apply filter to remove unknown words
    dictionary.filter_extremes(no_below = 15,  no_above = 0.5)
    # Convert documents to BoW
    corpus = [dictionary.doc2bow(doc) for doc in processed_df['clean text']]

    return dictionary, corpus

def coherence_grid(dictionary, corpus, texts, limit, start, step, random_state, chunksize, alpha):
    """
    Coherence grid helps to train multiple models with diferent number of topics and evaluate with Coherence Model to get a determinate score.

    Parameters
    ----------
    dictionary: gensim.corpora.Dictionary
        Dictionary containing the vocabulary of the texts.
    corpus: List of list
        Corpus in Bag-of-Words format.
    texts: List of list 
        Sample texts to evaluate the coherence score.
    limit: int
        Indicate the max. num of topics.
    start: int
        Indicate the min. num of topics.
    step: int
        The step size to increment the number of topics.
    random_state: int
        Random state for reproducibility.
    chunksize: int
        Number of documents to be used in each training chunk.
    alpha: float or str
        Hyperparameter that affects the sparsity of the document-topic (theta) distribution.

    Returns
    -------
    model_list: list of gensim.models.LdaModel
        List of trained LDA models.
    coherence_values: list of float
        Coherence values corresponding to each model.

    """

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
    """
    Selects the best LDA model based on the highest coherence score.

    Parameters
    ----------
    model_list: list of models (LDA)
        List of trained LDA models.
    coherence_list: list of float
        List of coherence scores corresponding to each model.

    Returns
    -------
    best_model: LDA Model
        The LDA model with the highest coherence score.
    best_coherence: float
        The highest coherence score.
    """
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
    
    """
    Trains multiple LDA models with different numbers of topics, evaluates their coherence, 
    and logs the best model and its coherence score to MLflow.

    Parameters
    ----------
    experiment_name: str
        Name of the MLflow experiment.
    corpus: list of list
        Corpus in Bag-of-Words format.
    dictionary: gensim.corpora.Dictionary
        Dictionary containing the vocabulary of the texts.
    processed_df: pandas.DataFrame
        DataFrame containing the processed texts. Expects a column named 'clean text'.
    start: int
        The minimum number of topics to evaluate.
    limit: int
        The maximum number of topics to evaluate.
    step: int
        The step size to increment the number of topics.
    random_state: int
        Random state for reproducibility.
    chunksize: int
        Number of documents to be used in each training chunk.
    version: str
        Version identifier for the model.
    alpha: float or str, optional
        Hyperparameter that affects the sparsity of the document-topic (theta) distribution. Default is 'auto'.

    Returns
    -------
    model: LDA Model
        The LDA model with the highest coherence score.

    """
    

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

    """
    This function helps to extract the dominant topic for each document in the corpus and optionally prints the most important words for each topic.

    Parameters
    ----------
    model: LDA Model
        Trained LDA model.
    processed_df: pandas.DataFrame
        DataFrame containing the processed texts.
    corpus: list of list
        Corpus in Bag-of-Words format.
    show_results: bool, optional
        If True, prints the most important words of each topic. Default is True.

    Returns
    -------
    final_df: pandas.DataFrame
        DataFrame containing the original processed texts and their dominant topics with percentage contributions.
    """

    if show_results:
        # Show the most important words of each topic
        topics = model.print_topics(num_words = 10)
        for topic in topics:
            print(topic)

    dominant_topics = []
    for row_list in model[corpus]:
        row = row_list[0] if model.per_word_topics else row_list            
        row = sorted(row, key=lambda x: (x[1]), reverse=True)
        # Get dominant topic, contribution percentage and keywords.
        topic_num, prop_topic = row[0]
        # Append in dominant topics
        dominant_topics.append((topic_num, prop_topic))

    # Create a DataFrame with topics
    df_topics = pd.DataFrame(dominant_topics, columns=['dominant_topic', 'percentage_contribution'])

    # Combine
    final_df = pd.concat([processed_df, df_topics], axis=1)

    return final_df

def model_run(processed_df, config, show_results):
    """
    This function helps to run the entire topic modeling pipeline: setting up the model, training it, and extracting the results.

    Parameters
    ----------
    processed_df: pandas.DataFrame
        DataFrame containing the processed texts.
    config: object
        Configuration object containing the attributes.
    show_results: bool
        If True, prints the most important words of each topic.

    Returns
    -------
    final_df: pandas.DataFrame
        DataFrame containing the original processed texts and their dominant topics with percentage contributions.
    """

    dictionary, corpus = set_up_model(processed_df)

    model = train_model(config.experiment_name, corpus, dictionary, processed_df, config.start, 
                        config.limit, config.step, config.random_state, config.chunksize, config.version)
    
    
    final_df = get_results(model, processed_df, corpus, show_results = show_results)

    final_df['topic_name'] = final_df['dominant_topic'].map(config.topic_ideal_name)
    final_df.drop(columns=['text', 'clean text'], inplace=True)

    return final_df



