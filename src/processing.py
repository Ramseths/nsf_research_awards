from datetime import datetime
import os
import pandas as pd
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

