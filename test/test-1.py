import joblib
model = joblib.load('DL_model.pkl')
x="great movie"
model.predict(x)