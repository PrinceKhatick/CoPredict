import flask
import numpy as np
import pandas as pd
import requests
from fbprophet import Prophet
from flask import jsonify
from pandas import json_normalize

app = flask.Flask(__name__)

# Extracting info directly from the APIs and converting them into json files---------------------------------------------------
url = 'https://api.covid19india.org/data.json'

r = requests.get(url)
j = r.json()

url1 = 'https://api.covid19india.org/raw_data1.json'

r = requests.get(url1)
j1 = r.json()

url2 = 'https://api.covid19india.org/raw_data2.json'

r = requests.get(url2)
j2 = r.json()

url3 = 'https://api.covid19india.org/raw_data3.json'

r = requests.get(url3)
j3 = r.json()

url4 = 'https://api.covid19india.org/raw_data4.json'

r = requests.get(url4)
j4 = r.json()

url5 = 'https://api.covid19india.org/state_test_data.json'

r1 = requests.get(url5)
j5 = r1.json()

url6 = 'https://api.covid19india.org/states_daily.json'

r = requests.get(url6)
j6 = r.json()

url7 = 'https://api.covid19india.org/raw_data5.json'

r = requests.get(url7)
j7 = r.json()

url8 = 'https://api.covid19india.org/zones.json'

r = requests.get(url8)
j8 = r.json()

url9 = 'https://api.covid19india.org/raw_data6.json'

r = requests.get(url9)
j9 = r.json()

url10 = 'https://api.covid19india.org/raw_data7.json'

r = requests.get(url10)
j10 = r.json()

url11 = 'https://api.covid19india.org/raw_data8.json'

r = requests.get(url11)
j11 = r.json()

# Dataframe from URL------------------------------------------------------------------------------------------------------------

cases = pd.DataFrame.from_dict(
    json_normalize(j, record_path='cases_time_series'))

states = pd.DataFrame.from_dict(json_normalize(j, record_path='statewise'))

testing = pd.DataFrame.from_dict(json_normalize(j, record_path='tested'))

# DataFrames from url

patient1 = pd.DataFrame.from_dict(json_normalize(j1, record_path='raw_data'))

patient2 = pd.DataFrame.from_dict(json_normalize(j2, record_path='raw_data'))

patient3 = pd.DataFrame.from_dict(json_normalize(j3, record_path='raw_data'))

patient4 = pd.DataFrame.from_dict(json_normalize(j4, record_path='raw_data'))

patient5 = pd.DataFrame.from_dict(json_normalize(j7, record_path='raw_data'))

patient6 = pd.DataFrame.from_dict(json_normalize(j9, record_path='raw_data'))

patient7 = pd.DataFrame.from_dict(json_normalize(j10, record_path='raw_data'))

patient8 = pd.DataFrame.from_dict(json_normalize(j11,record_path='raw_data'))

# Merging all the dataframes to build the master dataframe for the patient info

patient = pd.DataFrame()
patient = patient1.append(patient2).append(patient3).append(patient4).append(patient5).append(patient6).append(patient7).append(patient8).reset_index(drop=True)


# Depiction of the patients information by Gender
states_testing = pd.DataFrame.from_dict(
    json_normalize(j5, record_path='states_tested_data'))
states_daily = pd.DataFrame.from_dict(
    json_normalize(j6, record_path='states_daily'))
states_zones = pd.DataFrame.from_dict(json_normalize(j8, record_path='zones'))

df_gender = patient['gender'].value_counts()
df_gender = df_gender.reset_index()


@app.route('/gender_ratio', methods=['GET'])
def get_gender_radio():
    return jsonify({
        'Male': int(df_gender['gender'].iloc[1]),
        'Female': int(df_gender['gender'].iloc[2])
    })


# End Of Gender Ratio India -------------------------------------------------------------------------------------------

# Manipulating the patient dataframe to change the age values to the age ranges

patient['agebracket'] = patient['agebracket'].str.strip().replace('', np.nan).replace('28-35', '32').replace('6 Months', '0.5').replace('5 months', '0.45').replace('5 Months',
                                                                                                                                                                    '0.45').replace('9 Months', '0.75').replace('8 month', '0.8').replace('1 DAY', '0.1').replace('9 Month', '0.8').replace('18-28', '25')

patient['agebracket'] = patient['agebracket'].astype('float')

patient['agebracket'] = pd.cut(patient['agebracket'], [0, 10, 20, 30, 40, 50, 60, 70, 80, 90], labels=[
                               '0-10', '10-20', '20-30', '30-40', '40-50', '50-60', '60-70', '70-80', '80+'], include_lowest=True)
df_age = patient['agebracket'].value_counts().sort_index()


@app.route('/getAgeSample')
def get_age_sample():
    data_to_array = []
    for i in range(len(df_age)):
        data_to_array.append(
            {'range': str(df_age.index[i]), 'value': int(df_age[i])})
    return jsonify(data_to_array)


# End Of Age Range


# Manipulating the 'date' column of the cases dataframe in order to make it depictable

cases['date1'] = '2020'
cases['Date'] = cases['date'].str.cat(cases['date1'])
cases['Date'] = pd.to_datetime(cases['Date'])

# Depiction of the total confirmed, recovered and death cases

df_cases = cases.groupby(
    'Date')[['totalconfirmed', 'totaldeceased', 'totalrecovered']].sum()
df_cases['totalconfirmed'] = df_cases['totalconfirmed'].astype('int')
df_cases['totaldeceased'] = df_cases['totaldeceased'].astype('int')
df_cases['totalrecovered'] = df_cases['totalrecovered'].astype('int')
df_cases['totalactive'] = df_cases['totalconfirmed'] - \
    (df_cases['totaldeceased'] + df_cases['totalrecovered'])

x = df_cases.index
y1 = df_cases['totalconfirmed']
y2 = df_cases['totaldeceased']
y3 = df_cases['totalrecovered']
y4 = df_cases['totalactive']

totalconfirmed = []
for i in range(len(x)):
    totalconfirmed.append({
        'date': str(x[i]),
        'confirmed': str(y1[i]),
        'recovered': str(y3[i]),
        'deceased': str(y2[i]),
        'active': str(y4[i])
    })


@app.route('/gettotalcase')
def gettotalcase():
    return jsonify(totalconfirmed)


# End of Total Cases Trend---------------------------------------------------------------------------------------------------------

df_daily = cases.groupby(
    'Date')[['dailyconfirmed', 'dailydeceased', 'dailyrecovered']].sum()

df_daily['dailyconfirmed'] = df_daily['dailyconfirmed'].astype('int')
df_daily['dailydeceased'] = df_daily['dailydeceased'].astype('int')
df_daily['dailyrecovered'] = df_daily['dailyrecovered'].astype('int')
df_daily['dailyactive'] = df_daily['dailyconfirmed'] - \
    (df_daily['dailydeceased'] + df_daily['dailyrecovered'])

dx = df_daily.index
dy1 = df_daily['dailyconfirmed']
dy2 = df_daily['dailydeceased']
dy3 = df_daily['dailyrecovered']
dy4 = df_daily['dailyactive']

dailyconfirmed = []

for i in range(len(x)):
    dailyconfirmed.append({'date': str(dx[i]),
                           'confirmed': str(dy1[i]),
                           'recovered': str(dy3[i]),
                           'deceased': str(dy2[i]),
                           'active': str(dy4[i]),
                           })


@app.route('/getdailycase')
def getdailycase():
    return jsonify(dailyconfirmed)


# End of Daily Tend -----------------------------------------------------------------------------------------------------------


df_testing = testing[['updatetimestamp', 'testspermillion']]
df_testing.updatetimestamp = pd.to_datetime(
    df_testing['updatetimestamp'], format='%d/%m/%Y %H:%M:%S')
df_testing.set_index('updatetimestamp', inplace=True)
df_testing.testspermillion = df_testing['testspermillion'].str.strip().replace(
    '', np.nan).astype('float')
df_testing = df_testing.testspermillion.resample('SM').mean()


# df_testing = pd.concat([t1, t2], axis='columns')


@app.route('/getTestPerMil', methods=['GET'])
def getTestPerMil():
    test_per_mil = []
    for i in range(len(df_testing.index)):
        test_per_mil.append({
            "Date": str(df_testing.index[i]),
            "totalsamplestested": str(df_testing[i]),
            "testspermillion": str(df_testing[i])
        })
    return jsonify(test_per_mil)


# End of getTestPermil

df_states_testing = states_testing[['state', 'updatedon', 'testspermillion']]
df_states_testing.testspermillion = df_states_testing['testspermillion'].str.strip(
).replace('', np.nan).astype('float')
df_states_testing = df_states_testing.groupby(
    'state')['testspermillion'].mean()


@app.route('/statepermill', methods=['GET'])
def statepermill():
    data_list = []
    for i in range(len(df_states_testing.index)):
        data_list.append({"State": str(df_states_testing.index[i]),
                          "value": str(df_states_testing[i])
                          })

    return jsonify(data_list)


states_daily.date = pd.to_datetime(states_daily.date)

d = {'an': 'andaman and nicobar islands', 'ap': 'andhra pradesh', 'ar': 'arunachal pradesh', 'as': 'assam',
     'br': 'bihar', 'ch': 'chandigarh',
     'ct': 'chhattisgarh', 'dd': 'daman and diu', 'dl': 'delhi', 'dn': 'dadra and nagar haveli', 'ga': 'goa',
     'gj': 'gujarat', 'hp': 'himachal pradesh',
     'hr': 'haryana', 'jh': 'jharkhand', 'jk': 'jammu and kashmir', 'ka': 'karnataka', 'kl': 'kerala', 'la': 'ladakh',
     'ld': 'lakshadweep',
     'mh': 'maharashtra', 'ml': 'meghalaya', 'mn': 'manipur', 'mp': 'madhya pradesh', 'mz': 'mizoram', 'nl': 'nagaland',
     'or': 'odisha',
     'pb': 'punjab', 'py': 'puducherry', 'rj': 'rajasthan', 'sk': 'sikkim', 'tg': 'telangana', 'tn': 'tamil nadu',
     'tr': 'tripura',
     'up': 'uttar pradesh', 'ut': 'uttarakhand', 'wb': 'west bengal'}

states_daily.rename(columns=d, inplace=True)

c = np.array(states_daily.columns)
c = np.delete(c, [7, 32, 36, 37])

df_state_cases = pd.pivot_table(
    states_daily, values=c, index='date', columns='status', aggfunc=np.sum)

df_state_tes = states_testing[['state','updatedon','totaltested']]

df_state_tes.totaltested = df_state_tes['totaltested'].str.strip().replace('',np.nan).astype('float')
df_state_tes.state = df_state_tes['state'].str.replace('Andaman and Nicobar Islands','Andaman And Nicobar Islands').replace('Jammu and Kashmir','Jammu And Kashmir')
df_state_tes.updatedon = pd.to_datetime(df_state_tes['updatedon'],format='%d/%m/%Y')
df_state_tes = df_state_tes.groupby(['state','updatedon'])['totaltested'].sum()


@app.route('/getstatedata/<statename>', methods=['GET'])
def getstatedata(statename):

    x = df_state_cases.index
    state_choice_y1 = df_state_cases[statename].iloc[:, 0].astype('int64')
    state_choice_y2 = df_state_cases[statename].iloc[:, 1].astype('int64')
    state_choice_y3 = df_state_cases[statename].iloc[:, 2].astype('int64')
    state_choice_y4 = state_choice_y1 - (state_choice_y2 + state_choice_y3)

    xx = df_state_cases.index
    state_choice_y11 = df_state_cases[statename].iloc[:, 0].astype('int64').cumsum()
    state_choice_y22 = df_state_cases[statename].iloc[:, 1].astype('int64').cumsum()
    state_choice_y33 = df_state_cases[statename].iloc[:, 2].astype('int64').cumsum()
    state_choice_y44 = state_choice_y11 - (state_choice_y22 + state_choice_y33)

    x_state_tes = df_state_tes[statename.title()].index
    y_state_tes = df_state_tes[statename.title()]

    data_list = []
    data_list_test = []
    data_list_cumsum = []

    for i in range(len(x)):
        data_list_cumsum.append({
            "date": str(xx[i]),
            "Confirmed": str(state_choice_y11[i]),
            "Deaths": str(state_choice_y22[i]),
            "Recovered": str(state_choice_y33[i]),
            "Active": str(state_choice_y44[i])
        })

    for i in range(len(x)):
        data_list.append({
            "date": str(x[i]),
            "Confirmed": str(state_choice_y1[i]),
            "Deaths": str(state_choice_y2[i]),
            "Recovered": str(state_choice_y3[i]),
            "Active": str(state_choice_y4[i])
        })

    for i in range(len(x_state_tes)):
        data_list_test.append({
            'date': str(x_state_tes[i]),
            'value': str(y_state_tes[i])
        })

    return jsonify({
        'stateCase': data_list,
        'stateTest': data_list_test,
        'stateCasecumsum' : data_list_cumsum,
        'score':{
            'calculatedScore':calculateScore(statename),
            'zone':Zones(statename),
            'TestPositivityRate':TestPositivityRate(statename),
            'MortalityRate' :MortalityRate(statename),
            'TotalTestsPerMillion' : TotalTestsPerMillion(statename),
            'ConfirmedCasesPerMillion' : ConfirmedCasesPerMillion(statename)
        }
    })


patient['gender'] = patient['gender'].str.strip().replace('',np.nan)

patient['detectedstate'] = patient['detectedstate'].str.strip().replace('',np.nan).replace('Andaman and Nicobar Islands','Andaman And Nicobar Islands').replace('Jammu and Kashmir','Jammu And Kashmir')

patient['a'] = 1


@app.route('/infected_probability/<user_age>/<user_gender>/<user_dest>', methods=['GET'])
def infected_probability(user_age, user_gender, user_dest):
    user_age = int(user_age)
    user_dest = user_dest.title()
    user_df_gender = patient.groupby(['detectedstate', 'gender'])['a'].count()
    user_x = [user_df_gender[user_dest].iloc[1],
              user_df_gender[user_dest].iloc[0]]
    user_y = ['Male', 'Female']

    user_df_age = patient.groupby(['detectedstate', 'agebracket'])['a'].count()
    x_user = user_df_age[user_dest].index
    y_user = user_df_age[user_dest]

    z = patient.groupby(['detectedstate', 'agebracket', 'gender'])['a'].count()
    z = z[user_dest.title()]
    z = z.reset_index()
    a = z.pivot_table(values=['a'], index='agebracket', columns='gender')
    a = a.reset_index()

    x = a['agebracket']
    xpos = np.arange(len(x))
    # flatten() method is to convert the 2D numpy array into 1D, because graphs only take scalar values
    y1 = a['a'].iloc[:, 1].values.flatten()
    y2 = a['a'].iloc[:, 0].values.flatten()

    sample_size = sum(y1+y2)

    conf = df_state_cases[user_dest.lower()].iloc[:, 0].astype(
        'int64').cumsum().iloc[-1]
    df = pd.read_csv('./statesPopulation.csv', index_col='States')
    perc = conf / df.loc[user_dest.lower()].iloc[0]
    scaled_pop = sample_size / perc

    if (user_gender == 'M'):

        if 0 < user_age < 15:

            num = a['a'].iloc[0:2, 1].sum()
            d1 = df.loc[user_dest.lower()].iloc[2]
            d2 = df.loc[user_dest.lower()].iloc[0]

        elif 15 <= user_age < 60:

            num = a['a'].iloc[2:6, 1].sum()
            d1 = df.loc[user_dest.lower()].iloc[4]
            d2 = df.loc[user_dest.lower()].iloc[0]

        elif user_age >= 60:

            num = a['a'].iloc[6:9, 1].sum()
            d1 = df.loc[user_dest.lower()].iloc[6]
            d2 = df.loc[user_dest.lower()].iloc[0]

    else:

        if 0 < user_age < 15:

            num = a['a'].iloc[0:2, 0].sum()
            d1 = df.loc[user_dest.lower()].iloc[3]
            d2 = df.loc[user_dest.lower()].iloc[0]

        elif 15 <= user_age < 60:

            num = a['a'].iloc[2:6, 0].sum()
            d1 = df.loc[user_dest.lower()].iloc[5]
            d2 = df.loc[user_dest.lower()].iloc[0]

        elif user_age >= 60:

            num = a['a'].iloc[6:9, 0].sum()
            d1 = df.loc[user_dest.lower()].iloc[7]
            d2 = df.loc[user_dest.lower()].iloc[0]

    den = (d1 / d2) * scaled_pop
    prob = num / den * (1 - (calculateScore(user_dest)/50))

    infected_probability_age_range = []

    for i in range(len(x_user)):
        infected_probability_age_range.append({
            'ageRange': str(x_user[i]),
            'value': str(y_user[i])
        })

    return jsonify({
        'genderRatio': {
            'Male': str(user_x[0]),
            'Female': str(user_x[1])
        },
        'ageRange': infected_probability_age_range,
        'prob': {
            "0": "The probability of the User to get infected in {} is {:.6f}".format(user_dest, prob),
            "1": '{} in a million can get infected in {}'.format(int(prob*1000000), user_dest)
        }
    })


@app.route('/', methods=['GET'])
def home():
    return jsonify({"Hello": "World"})


df = cases[['Date', 'totalconfirmed']]
df['ds'] = df['Date']
df['y'] = df['totalconfirmed']
df.drop(['Date', 'totalconfirmed'], axis=1, inplace=True)
df = df.iloc[50:]

m = Prophet(changepoint_range=0.99)
m.fit(df)
future = m.make_future_dataframe(periods=5)
forecast = m.predict(future)
forecastData = []
for i in range(len(forecast)):
    forecastData.append({
        "ds": str(forecast['ds'][i]),
        "yhat": float(forecast['yhat'][i]),
        "yhat_lower": float(forecast['yhat_lower'][i]),
        "yhat_upper": float(forecast['yhat_upper'][i])
    })


@app.route('/prediction', methods=['GET'])
def prediction():
    return jsonify(forecastData)


states_zones['state'] = states_zones['state'].str.strip().replace('',np.nan).replace('Andaman and Nicobar Islands','Andaman And Nicobar Islands').replace('Jammu and Kashmir','Jammu And Kashmir')
states_testing['state'] = states_testing['state'].str.strip().replace('',np.nan).replace('Andaman and Nicobar Islands','Andaman And Nicobar Islands').replace('Jammu and Kashmir','Jammu And Kashmir')
states_testing['totaltested'] = states_testing['totaltested'].str.strip().replace('',np.nan).astype('float')


def ConfirmedCasesPerMillion(ch):
    """"
    It returns the score of the state taking into consideration the no. of confirmed cases of the state scaled to 1 million.

    """

    ch = ch.lower()

    df1 = df_state_cases[ch].iloc[:, 0].astype('int64').cumsum()
    df2 = pd.read_csv('statesPopulation.csv', index_col='States')

    x = (df1.iloc[-1] * 1000000) / df2.loc[ch].iloc[0]

    if 0 <= x <= 10:
        return 10
    elif 10 < x <= 100:
        return 8
    elif 100 < x <= 1000:
        return 6
    elif 1000 < x <= 10000:
        return 4
    elif 10000 < x <= 100000:
        return 2
    elif 100000 < x <= 1000000:
        return 0


def TotalTestsPerMillion(ch):
    """"
        It returns the score of the state taking into consideration the no. of tests of the state scaled to per 1 million people.

    """
    df1 = states_testing[['state', 'totaltested']]
    df1.set_index('state', inplace=True)
    df2 = pd.read_csv('statesPopulation.csv', index_col='States')

    y = (df1.loc[ch.title()].iloc[-1].values *
         1000000) / df2.loc[ch.lower()].iloc[0]

    if 0 <= y <= 1000:
        return 0
    elif 1000 < y <= 5000:
        return 2
    elif 5000 < y <= 10000:
        return 4
    elif 10000 < y <= 100000:
        return 6
    elif 100000 < y <= 500000:
        return 8
    elif 500000 < y <= 1000000:
        return 10


def MortalityRate(ch):
    """"
        It returns the score of the state taking into consideration the mortality rate.

    """

    df1 = df_state_cases[ch.lower()].iloc[:, 1].astype('int64').sum()
    df2 = df_state_cases[ch.lower()].iloc[:, 0].astype('int64').sum()

    z = (df1 * 100) / df2

    if 0 <= z <= 1:
        return 10
    elif 1 < z <= 2:
        return 8
    elif 2 < z <= 3:
        return 6
    elif 3 < z <= 4:
        return 4
    elif 4 < z <= 5:
        return 2
    elif z > 5:
        return 0


def TestPositivityRate(ch):
    """"
        It returns the score of the state taking into consideration the test positivity rate of the state.

    """
    df1 = df_state_cases[ch.lower()].iloc[:, 0].astype('int64').cumsum()
    df2 = states_testing[['state', 'positive', 'totaltested']]
    df2.set_index('state', inplace=True)

    q = (df1[-1] * 100) / df2.loc[ch.title()].iloc[-1:, 1].values

    if q <= 2:
        return 10
    elif 2 < q <= 3:
        return 8
    elif 3 < q <= 4:
        return 6
    elif 4 < q <= 10:
        return 2
    elif q > 10:
        return 0


def Zones(ch):
    """"
        It returns the score of the state taking into consideration the no. of red and orange zones in the state.

    """
    df1 = states_zones[['state','zone']]
    df1.set_index('state',inplace=True)
    df1 = df1.loc[ch.title()]
    try:
      orange = df1[df1['zone']=='Orange'].count().values
    except Exception as e:
      orange = 0
    try:
      red = df1[df1['zone']=='Red'].count().values
    except Exception as e:
      red = 0
    
    df2 = pd.read_csv('statesPopulation.csv',index_col='States')
    
    p = (orange + red) / df2.loc[ch.lower()].iloc[1]
    
    if 0<=p<=0.1:
        return 10
    elif 0.1<p<=0.3:
        return 8
    elif 0.3<p<=0.5:
        return 6
    elif 0.5<p<=0.7:
        return 4
    elif 0.7<p<=0.9:
        return 2
    elif 0.9<p<=1:
        return 0


def calculateScore(ch):
    return (ConfirmedCasesPerMillion(ch) + Zones(ch) + TestPositivityRate(ch) + MortalityRate(ch) + TotalTestsPerMillion(ch))


if __name__ == '__main__':
    app.run(threaded=True, port=5000)
