import streamlit as st
import seaborn as sns
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error,mean_squared_error, r2_score
#page config #
st.set_page_config("Linear Regression App ",layout="centered")
# Load css #
def load_css(filename):
    base_dir = os.path.dirname(__file__)   # folder of app_linear_regression.py
    css_path = os.path.join(base_dir, filename)

    if os.path.exists(css_path):
        with open(css_path) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    else:
        st.warning("style.css not found. Using default Streamlit style.")

load_css("style.css")
# Title and description #
st.markdown("""
            <div class="card">
            <h1>Linear Regression Web App</h1>
            <p>Predict <b>Tip Amount </b> from <b> Total Bill </b> using Linear Regression...</p>

            </div>
            """,unsafe_allow_html=True)
# load data #
@st.cache_data
def load_data():
    return sns.load_dataset("tips")
df=load_data()

# Data Preview #
st.markdown('<div class="card"><h3><b>Data Preview</b></h2>',unsafe_allow_html=True)
#st.subheader("Data Preview")
st.dataframe(df.head())
st.markdown('</div>',unsafe_allow_html=True)

# prepare data #
X,y=df[['total_bill']],df['tip']
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
scaler=StandardScaler()
X_train_scaled=scaler.fit_transform(X_train)
X_test_scaled=scaler.transform(X_test)

# train model #
model=LinearRegression()
model.fit(X_train_scaled,y_train)
y_pred=model.predict(X_test_scaled)

# metric calculations #
mae=mean_absolute_error(y_test,y_pred)
mse=mean_squared_error(y_test,y_pred)
rmse=np.sqrt(mse)
r2=r2_score(y_test,y_pred)
adjusted_r2=1-(1-r2)*(len(y_test)-1)/(len(y_test)-2)

# visualizations #
st.markdown('<div class="card"><h3><b>Total Bill vs Tip</b></h3>',unsafe_allow_html=True)
# st.subheader("Total Bill vs Tip")
fig,ax=plt.subplots()
ax.scatter(df['total_bill'],df['tip'],alpha=0.6)
ax.plot(df['total_bill'],model.predict(scaler.transform(df[['total_bill']])),color='red')
ax.set_xlabel("Total Bill(Rs.)")
ax.set_ylabel("Tip(Rs.)")
st.pyplot(fig)
st.markdown('</div>',unsafe_allow_html=True)

# performance metrics #
st.markdown('<div class="card"><h3><b>Model Performance Metrics</b></h3>',unsafe_allow_html=True)
#st.subheader("Model Performance Metrics")
c1,c2=st.columns(2)
c1.metric("Mean Absolute Error (Mae)",f"{mae:.2f}")
c2.metric("Root Mean Squared Error (Rmse)",f"{rmse:.2f}")

c3,c4=st.columns(2)
c3.metric("R-squared (R2)",f"{r2:.2f}") 
c4.metric("Adjusted R-squared",f"{adjusted_r2:.2f}")
st.markdown('</div>',unsafe_allow_html=True)

# m and c #
st.markdown(f"""
            <div class="card">
            <h3>Model Interception</h3>
            <p><b>coefficient (m)</b>: {model.coef_[0]:.3f}<br>
            <b>Intercept: </b> {model.intercept_:.3f}</p>
            </div>
            """,unsafe_allow_html=True)

# prediction #
st.markdown('<div class = "card"><h3><b>Predict tip Amount</b></h3>',unsafe_allow_html=True)
#st.subheader("Predict tip Amount")
bill=st.slider("Total Bill Amount (Rs.)",float(df['total_bill'].min()),float(df['total_bill'].max()),30.0)
tip=model.predict(scaler.transform([[bill]]))[0]
st.markdown(f'<div class="prediction-box">Predicted Tip: (RS.) {tip:.2f}</div>',unsafe_allow_html=True)
st.markdown('</div>',unsafe_allow_html=True)
