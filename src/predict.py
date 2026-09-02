from evaluate.py import pipe

def predict_model(X_test):
   prediction = pipe.predict_proba(X_test)[:,1][10]
   return prediction