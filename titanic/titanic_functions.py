import os
import pandas as pd


def setup_environment(project_name):
    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)

    print(f"{project_name.upper()} KAGGLE COMPETITION")

    main_path = os.path.join(os.getcwd(), "..")
    data_path = os.path.join(main_path, "data", project_name)
    output_path = os.path.join(main_path, "output", project_name)

    if not os.path.isdir(output_path):
        print("Making output folder")
        os.mkdir(output_path)

    return data_path, output_path
