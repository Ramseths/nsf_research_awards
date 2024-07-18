from datetime import datetime
import os
import pandas as pd
from nltk.corpus import stopwords
import spacy
import re
import xml.etree.ElementTree as ET


def get_xml_path(directory_path):
    """
    This function helps to find all files with xml extension.

    Parameters
    ----------
    directory_path: Primary path to find files.

    Returns
    -------
    xml_files: List with all paths.
    """
    # Primary Path
    directory_path = './../data/raw/2020/'

    # Get all files to process
    xml_files = [os.path.join(directory_path, f) for f in os.listdir(directory_path) if f.endswith('.xml')]

    return xml_files

def extract_data(xml_file, vars_to_process):
    """
    This function helps to extract data for each file.

    Parameters
    ----------
    xml_file: XML File.

    Returns
    -------
    list: List with all info.
    """
    tree = ET.parse(xml_file)
    root = tree.getroot()
    return [root.find(element).text if root.find(element) is not None else None for element in vars_to_process]


def processing_dataframe(list_xml_files, vars_to_process):
    """
    This function helps to process all info and make a DataFrame.

    Parameters
    ----------
    list_xml_files: List with all paths of XML files.

    Returns
    -------
    df: DataFrame with all info processed.
    """
    # Get all info for each XML file
    data_list = [extract_data(xml_file, vars_to_process) for xml_file in list_xml_files]

    # Create DataFrame
    df = pd.DataFrame(data_list, columns=['date', 'country', 'researcher', 'total_amount', 'text'])

    return df

def processing(config):
    print('1. Getting files...')
    xml_files = get_xml_path(directory_path=config.data.directory_path)

    print('2. Processing Data...')
    df = processing_dataframe(list_xml_files = xml_files, vars_to_process = config.data.vars_to_extract)

    print('3. DataFrame Shape...')
    print(df.shape)
    try:
        current_date = str(datetime.today().date()).replace('-', '')
        # Save files
        df.to_csv(config.data.output_path + f'{current_date}_nsf_research_awards_abstracts.csv', index = False)
        df.to_csv(config.data.output_path + 'nsf_research_awards_abstracts.csv', index = False)
    except Exception as e:
        print('Error Processing: {e}')


def data_load(path):
    """
    This function helps to load data for modeling.

    Parameters
    ----------
    path: path for get file.

    Returns
    -------
    df: DataFrame without NaNs.
    """
    original_data = pd.read_csv(path)

    # Get total NaNs by Column
    original_data.isnull().sum()

    # Remove NaNs
    df = original_data.dropna()

    return df


def clean_text(document, model_spacy, stopwords):
    """
    Applying lemmatization and certain regex to remove special characters and html tags.

    Parameters
    ----------
    document: Raw text to be cleaned.

    Returns
    -------
    tokens: Texto split into tokens.
    """  
    # Convert to Spacy Doc
    lemma = model_spacy(document)

    # Lemmatization
    lemmas = ' '.join([word.lemma_ for word in lemma])

    # Just alphabetics & remove html labels
    words = re.sub('<br/>', '', lemmas)
    words = re.sub(r'[^a-zA-ZáéíóúüñÁÉÍÓÚÜÑ]', ' ', lemmas)

    # Get tokens
    tokens = words.lower().split()

    # Remove stopwords & keep tokens with size 5.
    tokens = [token for token in tokens if token not in stopwords and len(token) > 5]
    return tokens

def apply_transformation(df):
    """
    This function helps to apply clean_text function.

    Parameters
    ---------
    df: DataFramt to process.

    Returns
    -------
    data: DataFramwe with new column called 'clean text'.
    """
    data = df.copy()

    # Load model
    model_spacy = spacy.load('en_core_web_lg')
    mystopwords = stopwords.words('english')

    # Apply data clean
    df_clean = [clean_text(document, model_spacy, mystopwords) for document in data.text]
    # Create column  
    data['clean text'] = df_clean

    return data