from folktables import ACSDataSource, ACSIncome
year = '2018'
horizon = '1-Year'
survey = 'person'
state = 'CA'


data_source = ACSDataSource(
    survey_year=year,
    horizon=horizon,
    survey=survey
)

data = data_source.get_data(states=[state], download=True)

X, y, _ = ACSIncome.df_to_pandas(data)

X.to_csv(year + "_" + horizon + "_" + survey + "_" + state + "_features.csv", index=False)
y.to_csv(year + "_" + horizon + "_" + survey + "_" + state + "_labels.csv", index=False)