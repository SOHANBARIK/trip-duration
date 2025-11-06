# # import pandas as pd
# # import numpy as np
# # import pathlib

# # from sklearn.model_selection import train_test_split

# # from feature_definitions import feature_build

# # def load_data(data_path):
# #     # load your dataset from a given path
# #     df= pd.read_csv(data_path)
# #     return df

# # def split_data(df, test_split,seed):
# #     #Split the dataset into train and test sets
# #     train,test = train_test_split(df, test_size=test_split, random_state=seed)
# #     return train,test

# # def save_data(train,test, output_path):
# #     # Save the split datasets to the specified output path
# #     pathlib.Path(output_path).mkdir(parents=True, exist_ok=True)
# #     train.to_csv(output_path + '/train.csv', index=False)
# #     test.to_csv(output_path + '/test.csv', index=False)

# # if __name__ == "__main__":
# #     curr_dir = pathlib.Path(__file__)
# #     home_dir = curr_dir.parent.parent.parent
# #     test_path = home_dir.as_posix()+'/data/raw/test.csv'
# #     train_path= home_dir.as_posix()+'/data/raw/train.csv'
    
# #     train_data= pd.read_csv(train_path)
# #     test_data= pd.read_csv(test_path)
    
# #     do_not_use_for_training = ['id', 'pickup_datetime', 
# #                                'dropoff_datetime','check_trip_duration',
                               
# #                                'avg_speed_m', 'pickup_datetime_group']
# #     feature_names= [col for col in train_data.columns if col not in do_not_use_for_training]    
# #     print('We have %i features.' % len(feature_names))
    
# #     train_data = train_data[feature_names]
# #     test_data = test_data[feature_names]
    
# #     output_path = home_dir.as_posix()+ '/data/processed'





# import pandas as pd
# import numpy as np
# import pathlib
# from feature_definitions import feature_build

# if __name__ == "__main__":
#     curr_dir = pathlib.Path(__file__)
#     home_dir = curr_dir.parent.parent.parent

#     # Paths to raw data
#     train_path = home_dir / "data" / "raw" / "train.csv"
#     test_path = home_dir / "data" / "raw" / "test.csv"

#     # Read raw data
#     train_data = pd.read_csv(train_path)
#     test_data = pd.read_csv(test_path)

#     # Columns to exclude from training
#     do_not_use_for_training = [
#         "id",
#         "check_trip_duration",
#         "avg_speed_m",
#         "pickup_datetime_group",
#     ]

#     # Select useful features
#     feature_names = [col for col in train_data.columns if col not in do_not_use_for_training]
#     print(f"We have {len(feature_names)} features.")

#     train_data = train_data[feature_names]
#     test_data = test_data[[col for col in feature_names if col in test_data.columns]]

#     # Apply feature engineering
#     train_features = feature_build(train_data)
#     test_features = feature_build(test_data)

#     # Derive final usable feature names dynamically
#     feature_names = train_features.columns.tolist()

#     # Remove target column from train list if present
#     if "trip_duration" in feature_names:
#         feature_names.remove("trip_duration")

#     # Apply only valid features to test data
#     test_features = test_features[[col for col in feature_names if col in test_features.columns]]

#     # Save processed features
#     output_path = home_dir / "data" / "processed"
#     output_path.mkdir(parents=True, exist_ok=True)
#     train_features.to_csv(output_path / "train.csv", index=False)
#     test_features.to_csv(output_path / "test.csv", index=False)

#     print("✅ Features built successfully and saved to:", output_path)





import pathlib
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

from feature_definitions import feature_build

def load_data(data_path):
    # Load your dataset from a given path
    df = pd.read_csv(data_path)
    return df

def save_data(train, test, output_path):
    # Save the split datasets to the specified output path
    pathlib.Path(output_path).mkdir(parents=True, exist_ok=True)
    train.to_csv(output_path + '/train.csv', index=False)
    test.to_csv(output_path + '/test.csv', index=False)

if __name__ == '__main__':
    curr_dir = pathlib.Path(__file__)
    home_dir = curr_dir.parent.parent.parent
    
    train_path = home_dir.as_posix() + '/data/raw/train.csv'
    test_path = home_dir.as_posix() + '/data/raw/test.csv'
    
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)

    output_path = home_dir.as_posix() + '/data/processed'

    train_data = feature_build(train_data, 'train-data')
    test_data = feature_build(test_data, 'test-data')

    save_data(train_data, test_data, output_path)


    