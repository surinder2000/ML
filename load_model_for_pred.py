from sklearn.externals import joblib
model = joblib.load('Housepricepred.pk1')
print("Welcome")
inp = [[None]*5]
inp[0][0] = float(input("Enter average area income: "))
inp[0][1] = float(input("Enter average area house age: "))
inp[0][2] = float(input("Enter average area no of rooms: "))
inp[0][3] = float(input("Enter average area no of bedrooms: "))
inp[0][4] = float(input("Enter average area population: "))

res = model.predict(inp)[0]

print(round(res,3))


