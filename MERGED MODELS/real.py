# Importing all necessary libraries.
import numpy as np
# TensorFlow must load before pandas on this Windows environment.  Pandas loads
# native libraries that can otherwise prevent TensorFlow's DLL from initializing.
from tensorflow.keras.models import load_model
import pandas as pd
import yfinance as yf
import streamlit as st
import matplotlib.pyplot as plt
import joblib

from datetime import date, timedelta

st.set_page_config(layout="wide")
alpha, beta, gamma = st.columns((4.1,5,3))
beta.title('Stock Price Prediction')
images0,image1,images1 = st.columns((2.8,6,2))
image1.image('https://miro.medium.com/max/620/0*dunTLlei47QWR7NR.gif')
alpha1, gamma1 = st.columns(2)

alpha1.markdown('Hello and Welcome to Stock Price Predictor. This WebApp is designed to predict the price of select stocks. To get started, Select a stock from the given list and select the starting date and ending date for the same to view its information.')
stock = gamma1.markdown('Stock price prediction is a very researched entity in the modern day world. It helps companies to raise capital, it helps people generate passive income, stock markets represent the state of the economy of the country and it is widely used soutce for people to invest money in companies with high growth potential')

list=['Select Stock','AAPL','AMZN','GOOGL','GOOG','TSLA','FB'
,'NVDA','V']

stock = st.selectbox('Select Stock', list)
today = date.today()
default_start = today - timedelta(days=5 * 365)
j,k= st.columns(2)
starting = j.date_input('Pick a starting Date', value=default_start, max_value=today)
ending = k.date_input('Pick an ending date', value=today, max_value=today)

if stock == 'Select Stock':
    st.info('Choose a stock to view its market data and predictions.')
    st.stop()

if starting >= ending:
    st.error('The ending date must be later than the starting date.')
    st.stop()

df1 = yf.Ticker(stock)

st.write("\n")
st.write("\n")
rhs, rh, rhs1 = st.columns((3,3,2))
rh.markdown("### Let's have a look at raw data")
def load_data(ticker):
    df = yf.download(ticker, start=starting, end=ending, auto_adjust=False, progress=False)
    df.reset_index(inplace=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [column[0] for column in df.columns]
    return df

df = load_data(stock)

if df.empty:
    st.error('No market data was found for the selected stock and dates.')
    st.stop()

if len(df) < 250:
    st.warning('Choose a date range with at least 250 trading days for all prediction models.')
    st.stop()

rds, rd, rds1 = st.columns((2.5,5,1))

raw_data = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']].head(10).copy()
raw_data['Date'] = pd.to_datetime(raw_data['Date']).dt.strftime('%d %b %Y')
raw_data[['Open', 'High', 'Low', 'Close']] = raw_data[['Open', 'High', 'Low', 'Close']].round(2)
raw_data['Volume'] = raw_data['Volume'].map('{:,.0f}'.format)
rd.dataframe(raw_data, hide_index=True, width="stretch", height=420)
tickerDF = df1.history(start=starting, end=ending)


st.markdown("""======================================================================================================================================================================""")

rhs, rh, rhs1 = st.columns((2.5,3,2))
rh.markdown("""### DATA IN GRAPHICAL FORM """)
st.write("\n")
st.write("\n")

graphs1, graphs3 = st.columns(2)
graphs1.markdown("""### Opening Price """)
graphs1.line_chart(tickerDF.Open)
graphs3.markdown("""### Volume Price """)
graphs3.line_chart(tickerDF.Volume)
graphs1.markdown("""### Closing Price """)
graphs1.line_chart(tickerDF.Close)
graphs3.markdown("""### Highest Price """)
graphs3.line_chart(tickerDF.High)
graphs1.markdown("""### Lowest Price""")
graphs1.line_chart(tickerDF.Low)
graphs3.markdown("""### Adjusted Closing Price """)
graphs3.line_chart(df['Close'])


st.markdown("""======================================================================================================================================================================""")
st.title('K-NEAREST NEIGHBOUR :-')
st.write('\n')

df['Open - Close']=df['Open'] - df['Close']
df['High - Low'] = df['High'] - df['Low']

X = df[['Open - Close', 'High - Low']]

Y= np.where(df['Close'].shift(-1)>df['Close'],1,-1)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y ,test_size=0.25, random_state=44)

from sklearn.neighbors import KNeighborsClassifier
from sklearn import neighbors
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

#using gridsearch to find the best parameter
params = {'n_neighbors': [2,3,4,5,6,7,8,9,10,11,12,13,14,15]}
knn = neighbors.KNeighborsClassifier()
model = GridSearchCV(knn, params, cv=5)

#fit the model
model.fit(X_train, y_train)

#Accuracy Score
accuracy_train = accuracy_score(y_train, model.predict(X_train))
accuracy_test = accuracy_score(y_test, model.predict(X_test))

rhs, rh, rhs1 = st.columns((2.5,3,2))
rhs.subheader('Train Data Accuracy :  %.2f' %accuracy_train)
rhs1.subheader('Test Data Accuracy :  %.2f' %accuracy_test)
st.write('\n')
st.write('\n')

predictions_classification = model.predict(X_test)

actual_predicted_data = pd.DataFrame({'Actual Class': y_test.ravel(), 'Predicted Class': predictions_classification.ravel()})

y = df['Close']


from sklearn.neighbors import KNeighborsRegressor
from sklearn import neighbors

X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X, y, test_size=0.25, random_state=44)

#using gridsearch to find the best parameter
params = {'n_neighbors': [2,3,4,5,6,7,8,9,10,11,12,13,14,15]}
knn_reg = neighbors.KNeighborsRegressor()
model_reg = GridSearchCV(knn_reg, params, cv=5)

#fit the model and make predictions
model_reg.fit(X_train_reg, y_train_reg)
predictions = model_reg.predict(X_test_reg)

valid = pd.DataFrame({'Actual Price': y_test_reg.values.ravel(), 'Predicted Price value': predictions.ravel()})

classifier_column, regression_column = st.columns(2, gap="large")

with classifier_column:
    st.subheader('KNN Classifier')
    st.dataframe(
        actual_predicted_data.head(10),
        hide_index=True,
        width="stretch",
        height=420,
    )

with regression_column:
    st.subheader('KNN Regression')
    st.dataframe(
        valid.head(10),
        hide_index=True,
        width="stretch",
        height=420,
    )

#rmse
rms=np.sqrt(np.mean(np.power((np.array(predictions)),2)))
st.write('RMSE : ',rms)

plt.figure(figsize=(12, 6))
plt.scatter(valid['Actual Price'], valid['Predicted Price value'], alpha=0.5)
plt.plot([valid['Actual Price'].min(), valid['Actual Price'].max()], [valid['Actual Price'].min(), valid['Actual Price'].max()], 'r--')
plt.title('Actual vs Predicted Price')
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.show()

st.markdown("""======================================================================================================================================================================""")

model = load_model("LSTM/STOCK.keras")
st.title('LONG-SHORT TERM MEMORY (LSTM) :-')
st.write('\n')

rhs, rh, rhs1 = st.columns((2.5,3,2))
rhs.subheader('Train Data Accuracy :  0.84')
rhs1.subheader('Test Data Accuracy :  0.72')

data_train = pd.DataFrame(df.Close[0: int(len(df) * 0.80)])
data_test = pd.DataFrame(df.Close[int(len(df) * 0.80): len(df)])

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))

past_100_days = data_train.tail(100)
data_test = pd.concat([past_100_days, data_test], ignore_index=True)
data_test_scale = scaler.fit_transform(data_test)

st.subheader('Price vs MA50')
ma_50_days = df.Close.rolling(50).mean()
fig1, ax1 = plt.subplots(figsize=(13, 6))
ax1.plot(ma_50_days, 'y')
ax1.plot(df.Close, 'g')
st.pyplot(fig1, width="stretch")

st.subheader('Price vs MA50 vs MA100')
ma_100_days = df.Close.rolling(100).mean()
fig2, ax2 = plt.subplots(figsize=(13, 6))
ax2.plot(ma_50_days, 'y')
ax2.plot(ma_100_days, 'b')
ax2.plot(df.Close, 'g')
st.pyplot(fig2, width="stretch")

st.subheader('Price vs MA100 vs MA200')
ma_200_days = df.Close.rolling(200).mean()
fig3, ax3 = plt.subplots(figsize=(13, 6))
ax3.plot(ma_100_days, 'b')
ax3.plot(ma_200_days, 'orange')
ax3.plot(df.Close, 'g')
st.pyplot(fig3, width="stretch")

x = []
y = []

for i in range(100, data_test_scale.shape[0]):
    x.append(data_test_scale[i - 100:i])
    y.append(data_test_scale[i, 0])

x, y = np.array(x), np.array(y)

predict = model.predict(x)

scale = 1 / scaler.scale_

predict = predict * scale
y = y * scale

st.subheader('Original Price vs Predicted Price')

fig4, ax4 = plt.subplots(figsize=(13, 6))
ax4.plot(ma_100_days, 'r', label='Predicted Price')
ax4.plot(df.Close, 'g', label='Original Price')
plt.xlabel('Time in days')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig4, width="stretch")
rm=np.sqrt(np.mean(np.power((np.array(predict)),2)))
st.write('RMSE : ',rm)

st.markdown("""======================================================================================================================================================================""")

df2 = df['Close']
df2 = pd.DataFrame(df2) 
df2.columns = ['Close']

loaded_tree = joblib.load("Decision Tree/decision_tree_model.joblib")
# Prediction 100 days into the future.
future_days = 100
df2['Prediction'] = df2['Close'].shift(-future_days)
df2.tail()

#LINEAR AND DECISION TREE REGRESSION

X = np.array(df2.drop(['Prediction'], axis=1))[:-future_days]
y = np.array(df2['Prediction'])[:-future_days]

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression

# Implementing Linear and Decision Tree Regression Algorithms.
st.success("Decision Tree predictions generated")
tree = DecisionTreeRegressor().fit(x_train, y_train)
lr = LinearRegression().fit(x_train, y_train)

x_future = df2.drop(['Prediction'], axis=1)[:-future_days]
x_future = x_future.tail(future_days)
x_future = np.array(x_future)

tree_prediction = tree.predict(x_future)
lr_prediction = lr.predict(x_future)
predictions = tree_prediction 
valid = df2[X.shape[0]:]
valid['Predictions'] = predictions

fig1 = plt.figure(figsize=(13,8))

st.title('DECISION TREE :-')
plt.xlabel('Years')
plt.ylabel('Close Price USD ($)')
plt.plot(df2['Close'])
plt.plot(valid['Predictions'], 'g')
plt.legend(["Original", "Predicted"])

st.success("Decision Tree predictions generated")
st.pyplot(fig1)

_, mid, _ = st.columns((3,4,3))
mid.subheader('ORIGINAL vs PREDICTED')

r = np.sqrt(np.mean((predictions - valid['Predictions'].values) ** 2))
st.write('RMSE : ', r)
