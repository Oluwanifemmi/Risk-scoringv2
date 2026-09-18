from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import sys
import os

#import projectpath
from src.preprocessing import data_import,data_split
from src.Feature_engineering import calculate_iv 
from src.train import build_preprocessing_pipeline
from src.evaluate import load_processed_data


default_args = {
    'owner' : 'nify',
    'retries': 1
}

with DAG(
    dag_id = 'ml_pipeline',
    default_args=default_args,
    description = 'Credit Risk Model',
    schedule_interval ='@daily',
    catchup = False) as dag:

    preprocess = PythonOperator(
        task_id = 'data_preprocessing',
        python_callable = data_import
    )

    feature_engineering = PythonOperator(
        task_id = 'feature_engineering',
        python_callable = calculate_iv
    )