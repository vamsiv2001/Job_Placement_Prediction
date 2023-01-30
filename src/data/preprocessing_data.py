from src.data.load_data import load_job_placement_data,save_preprocessed_data
import logging
import os



gender_to_int = {'M':0, 'F':1}
work_experience_to_int = {'Yes':0, 'No':1}
specialisation_to_int = {'Mkt&HR':0, 'Mkt&Fin':1}
status_to_int = {'Placed':0, 'Not Placed':1}

def preprocessing_data():
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

    df = load_job_placement_data()
    df['gender'] = df['gender'].map(gender_to_int)
    df['work_experience'] = df['work_experience'].map(work_experience_to_int)
    df['specialisation'] = df['specialisation'].map(specialisation_to_int)
    df['status'] = df['status'].map(status_to_int)      
    df = df.drop(columns=["emp_test_percentage","mba_percent", "ssc_board","hsc_board","hsc_subject","undergrad_degree"])

    save_preprocessed_data(df)
    return

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    print(os.getcwd())
    preprocessing_data()