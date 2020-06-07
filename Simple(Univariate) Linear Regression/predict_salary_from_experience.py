import joblib

model = joblib.load('predict_salary.pk1')

print('1. Estimate salary \n2. Exit')

while True:
    ch = int(input("Enter choice:"))
    if ch == 1:
        X = float(input("Enter experience:"))
        res = round(model.predict([[X]])[0],3)
        print("Estimated salary:",res)
    elif ch == 2:
        exit()
