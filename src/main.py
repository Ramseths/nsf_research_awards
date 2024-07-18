from datetime import datetime

import hydra
import pandas as pd

from model import model_run
from processing import apply_transformation, data_load, processing


@hydra.main(config_path='./../config/', config_name='main', version_base='1.2')
def main(config):
    print("----- PROCESSING RAW DATA -----")
    processing(config.process)

    print("\n----- PROCESSING TRUSTED DATA -----")
    df = data_load(config.data.modeling_data)
    processed_df = apply_transformation(df)
    processed_df.reset_index(drop=True, inplace=True)
    print(processed_df.head())

    print("\n----- MODELING -----")
    results_df = model_run(processed_df, config.model, show_results=True)

    try:
        current_date = str(datetime.today().date()).replace('-', '')
        # Save files
        results_df.to_csv(config.data.output_path + f'{current_date}_nsf_with_topics.csv', index = False)
        results_df.to_csv(config.data.output_path + 'nsf_with_topics.csv', index = False)
        print("\n----- SAVED SUCCESSFULLY -----")
    except Exception as e:
        print('Error Processing: {e}')

if __name__ == '__main__':
    main()