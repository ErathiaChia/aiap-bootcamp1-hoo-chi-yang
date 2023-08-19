# aiap-bootcamp1-hoo-chi-yang

# U.A ML Pipeline

##  Details
- Name: Hoo Chi Yang
- Email: c-hi.yang@hotmail.sg

## Folder Overview
.\
├── eda.ipynb\
├── README.md\
├── requirements.txt\
├── run.sh\
├── .git\
├── .github\
├── data\
│   └── score.db\
└── src\
    ├── build_models.py\
    ├── dataloader.py\
    ├── model_mgr.py\
    ├── pipeline.py\
    ├── preprocessing.py\
    └── visualize.py

## Instructions
- Pipeline can be run 

## Pipeline

## Feature Preprocessing Summary

| Feature            | Processing Steps                                                                                                                                                                                                                                             |
|--------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `age`              | Filter only Secondary School age students 15-16.|
| `CCA`              | Text & Values standardized to 'None', 'Arts', 'Sports', 'Clubs'|
| `tuition`          | Values standardized to 'Yes' / 'No'|
| `hours_of_sleep`   | Duration between 'sleep_time', 'wake_time'. Original columns dropped.|
| `student_id`       | Dropped.|
| `index`            | Dropped.|
| `final_test`       | Missing values dropped.|
| `attendance_rate`  | Classified into `att_type` 'A', 'B', or 'C'. NaN values imputed based on the `final_test`.|
| `n_male` , `n_female` | Create features `class_size` (sum of male and female) and `school_type` (categorized into 'boy_sch', 'girl_sch', or 'mixed'). Original columns dropped.|
| Categorical vars   | Dummy variable encoded. Original categorical columns dropped.|
| Numerical vars     | Standardized using `StandardScaler`.|


## EDA Findings

Based on EDA findings, histogram of residuals and Q-Q plot shows that residuals are approximately normal, Residual vs Fitted plot shows that there is some non-linearities that are not captured if we use a linear model. We expect that non-linear models may be better in predicting 'final_test' scores. 


The variables effects on 'final_test' makes sense. \
We expect that smaller 'class_size' may lead to better test outcomes due to more focus on individual students. \
Students who do not have CCA are likely to have better test outcomes as they have more time to work on their studies. \
'attendance_rate' greatly impacts 'final_test' scores. We interpret that there may be some scores reserved for attendance. \
Students with access to 'tuition' are correlated to having better 'final_test' scores. 

'final_test' are less correlated to 'gender', 'mode_of_transport', 'school_type', 'bag_color', and 'age'.

Unexpectedly, 'hours_per_week' of study have a slight negative correlation to 'final_test.' Perhaps maybe these are struggling students. \
Also, it also seems like students with visual 'learning_style' are correlated to have better 'final_test.'


## Model Choices


## Model Evaluation

