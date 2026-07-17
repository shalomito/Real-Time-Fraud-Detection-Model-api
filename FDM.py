import streamlit as st
import pandas as pd
import joblib
import time

model = joblib.load('my_fraud_detection_model.pkl')
# message.info("Model Loaded")
# time.sleep(3)
# message.empty()


st.title("🛡️Fraud Detection System")
st.write('\t' * 20, 'This AI system analyzes transaction patterns and predicts the probability of fraud')
st.info('Model Loaded')
st.markdown(
    """
    <style>
    h1 {
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True
)

with st.sidebar:
    st.title("Settings")
    st.write('FRAUD THRESHOLD')
    st.write('70%')
       )

hour_day = st.slider("Hours Day", 1, 24)
#is_weekend = st.selectbox("Is it weekend?\nTrue(1)/False(0)", [1, 0])
night_trx = st.selectbox(" Night Transaction? True(1)/False(0)", [1, 0])
#country = st.selectbox('Country', [
   # "Nigeria",
    #"Ghana",
    #"Kenya",
    #"South Africa",
    #"Egypt",
    #"United Kingdom",
    #"United States",
    #"Canada",
    #"Germany",
    #"France"
#])
#city = st.selectbox('City',
  #                  ['Kumasi', "Port Harcourt", "Mombasa", "Cape Town", "Cairo", "Birmingham", "Vancouver", "Frankfurt",
  #                   "Marseille"])

merchant_cat = st.text_input('What was purchased?')
#pay_method = st.text_input('Payment Method')
#dev_type = st.text_input('Device Type')
#customer_age = st.slider('Customer Age', 0, 120)
#credit_score = st.slider('Credit Score', 0, 100)
#account_age = st.slider('Account Age', 0, 150)
#account_balance = st.slider('Account Balance', 0, 100000)
trx_amount = st.slider('Transaction Amount', 0, 10000)
#num_prev_trx = st.slider('Number of previous transactions', 0, 100)
#trx_frequency = st.slider('Transaction Frequency', 0, 200)
#distance_from_home_meters = st.slider('Distance from Home Meters', 0, 400000)
is_international = st.selectbox('International Transaction\nTrue(1)/False(0)', [1, 0])
failed_attempts = st.selectbox('Failed Transaction\nTrue(1)/False(0)', [1, 0])
time_since_last_24hrs = st.slider('Time since last 24 hours', 0, 24)
pin_changed_recently = st.selectbox('Pin Changed Recently\nTrue(1)/False(0)', [1, 0])
trx_hour = st.slider('Transaction Hour', 0, 24)
#trx_min = st.slider('Transaction Minute', 0, 60)
#distance_from_home = st.slider('Distance from Home(km)', 0, 20000)
#trx_year = st.text_input('Transaction Year')
#trx_day = st.slider('Transaction Day', 0, 31)
#trx_month = st.slider('Transaction Month', 1, 12)


try:
    if st.button('Predict'):
        new_transaction = pd.DataFrame({
            'hour_of_day': [hour_day],
            #'is_weekend': [is_weekend],
            'is_night_transaction': [night_trx],
            #'country': [country],
            #'city': [city],
            'merchant_category': [merchant_cat],
            #'payment_method': [pay_method],
            #'device_type': [dev_type],
            #'customer_age': [customer_age],
            #'credit_score': [credit_score],
            #'account_age_years': [account_age],
            #'account_balance': [account_balance],
            'transaction_amount': [trx_amount],
            #'num_prev_transactions': [num_prev_trx],
            #'transaction_freq_monthly': [trx_frequency],
            #'distance_from_home_km': [distance_from_home],
            'time_since_last_txn_hrs': [time_since_last_24hrs],
            # 'distance_from_home': [distance_from_home_meters],
            'is_international': [is_international],
            'failed_attempts': [failed_attempts],
            'pin_changed_recently': [pin_changed_recently],
            #'transaction_year': [trx_year],
            #'transaction_month': [trx_month],
            #'transaction_day': [trx_day],
            'transaction_hour': [trx_hour]
           # 'transaction_min': [trx_min] 
        })
        result_predict = model.predict(new_transaction)[0]
        result_prob = model.predict_proba(new_transaction)[0, 1]
        result_prob2 = model.predict_proba(new_transaction)[0, 0]
        
        if result_prob > 0.7 :
            st.error(f'Your transaction is Fraudulent!\n\nTrx FRAUD rate {result_prob:.1%}')
            time.sleep(5)
            # st.error(f"Trx Fraudulent rate {result_prob:.2%}")
            # time.sleep(5)
            st.warning(f'Review this transaction')
            time.sleep(5)
        else:
            st.success(f"Your transaction is Legitimate!, Trx FRAUD rate {result_prob:.1%}")



 
except:
    st.warning("Fill the Required informations")
