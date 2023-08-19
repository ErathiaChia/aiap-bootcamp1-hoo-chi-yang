
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import numpy as np

# 'class_size', 'school_type'
#  'age', 'number_of_siblings' in categorical
# immpute and handle missing values



class Preprocessor:
    def __init__(self, df,seed=2023):
        self.df = df
        self.seed = seed

    def preprocess_dataframe(self, seed=2023):

        print("preprocess_dataframe() called, preprocessing dataframe...")

        data = self.df.copy()

        # Filter only to Secondary School age students
        data = data[data['age'] >= 15]

        # Process CCA column
        data['CCA'] = data['CCA'].str.lower().str.title()
        data.loc[data['CCA'].isin(['None', 'N/A']), 'CCA'] = 'None'
        data.loc[data['CCA'] == 'Arts', 'CCA'] = 'Arts'
        data.loc[data['CCA'] == 'Sports', 'CCA'] = 'Sports'
        data.loc[data['CCA'] == 'Clubs', 'CCA'] = 'Clubs'

        # Process tuition column
        data['tuition'] = data['tuition'].replace({'Y': 'Yes', 'N': 'No'})
        print("After processing CCA and tuition:", data.isnull().sum().sum())

        # Calculate sleep duration in hours
        data['hours_of_sleep'] = (pd.to_datetime(data['wake_time']) - pd.to_datetime(data['sleep_time'])).dt.seconds / 3600
        # drop sleep_time and wake_time
        data.drop(['sleep_time', 'wake_time'], axis=1, inplace=True)
        
        # drop student_id
        data.drop(['student_id'], axis=1, inplace=True)

        #drop index
        data.drop(['index'], axis=1, inplace=True)
        

        print("After calculating hours_of_sleep:", data.isnull().sum().sum())

        # Classify attendance type

        def classify_attendance_type(att_rate):
            if att_rate < 50:
                return 'C'
            elif att_rate < 90:
                return 'B'
            elif np.isnan(att_rate):
                return np.nan
            else:
                return 'A'
    
        
        data['att_type'] = data['attendance_rate'].apply(classify_attendance_type)

        categorical_vars = ['direct_admission', 'CCA', 'learning_style', 'gender', 'mode_of_transport', 'bag_color', 'tuition', 'att_type', 'age', 'number_of_siblings']
        

        # Handling nan and impute attendance rate
        data = data.dropna(subset=['final_test'])

        mean_attendance = data.groupby('att_type')['attendance_rate'].mean()
        list_att_mean = list(mean_attendance)
        mask_A = (data['att_type'].isna()) & (data['final_test'] > 50)
        mask_B = (data['att_type'].isna()) & (data['final_test'] > 45) & (data['final_test'] <= 50)
        mask_C = (data['att_type'].isna()) & (data['final_test'] <= 45)

        data.loc[mask_A, 'att_type'] = 'A'
        data.loc[mask_A, 'attendance_rate'] = list_att_mean[0]

        data.loc[mask_B, 'att_type'] = 'B'
        data.loc[mask_B, 'attendance_rate'] = list_att_mean[1]

        data.loc[mask_C, 'att_type'] = 'C'
        data.loc[mask_C, 'attendance_rate'] = list_att_mean[2]

        print("After dropping NaNs: & imputing attendance rate", data.isnull().sum().sum())

        # Feature engineering 'class_size' and 'school_type'  
        # 'class_size'
        data['class_size'] = data['n_male'] + data['n_female']

        # 'school_type'
        conditions = [
            (data['n_male'] > 0) & (data['n_female'] == 0),
            (data['n_female'] > 0) & (data['n_male'] == 0),
            (data['n_male'] > 0) & (data['n_female'] > 0)
        ]
        choices = ['boy_sch', 'girl_sch', 'mixed']
        data['school_type'] = np.select(conditions, choices, default='unknown')
        data.drop(['n_male','n_female'], axis=1, inplace=True)

        categorical_vars.append('school_type')

        print("After feature engineering:", data.isnull().sum().sum())

        print("Categorical variables:", categorical_vars)
        print("Numerical variables:", data.drop(categorical_vars, axis=1).columns.tolist())


        # One-hot encode categorical variables
        encoder = OneHotEncoder(drop='first')
        encoded_features = encoder.fit_transform(data[categorical_vars]).toarray()
        encoded_data = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(categorical_vars))

        # Reset the index of both dataframes before concatenating
        data.reset_index(drop=True, inplace=True)
        encoded_data.reset_index(drop=True, inplace=True)

        data = pd.concat([data, encoded_data], axis=1)
        data.drop(categorical_vars, axis=1, inplace=True)


        print("After one-hot encoding:", data.isnull().sum().sum())

        X = data.drop('final_test', axis=1)
        y = data['final_test']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=seed)

        print("X_train NaNs:", pd.DataFrame(X_train).isnull().sum().sum())
        print("X_test NaNs:", pd.DataFrame(X_test).isnull().sum().sum())
        print("y_train NaNs:", pd.DataFrame(y_train).isnull().sum().sum())
        print("y_test NaNs:", pd.DataFrame(y_test).isnull().sum().sum())

        # standardize numerical variables
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        X_train = pd.DataFrame(X_train, columns=X.columns)
        X_test = pd.DataFrame(X_test, columns=X.columns)

        print(f'Finished preprocessing dataframe, the X columns are: {X_train.columns.tolist()}. The y column is: {y_train.name}')
        print(f'The shapes of X_train, X_test, y_train, y_test are: {X_train.shape}, {X_test.shape}, {y_train.shape}, {y_test.shape}')

        return X_train, X_test, y_train, y_test, data
