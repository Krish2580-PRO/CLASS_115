import pandas as pa
import plotly.express as pe
import matplotlib.pyplot as plt
import numpy as nu
from sklearn.linear_model import LogisticRegression

# ---------------------------------------------------------------------

data = pa.read_csv("project.csv")

vel = data["Velocity"].tolist()
esc = data["Escaped"].tolist()

fig = pe.scatter(x = vel, y = esc)
#fig.show()


# ----------------------------------------------------------

X = nu.reshape(vel, (len(vel), 1))

Y = nu.reshape(esc, (len(esc), 1))

lr = LogisticRegression()
lr.fit(X,Y)

plt.figure()
plt.scatter(X.ravel(), Y, color = "red")

def model(x):
    return 1/ (1 + nu.exp(-x) )

Xtest = nu.linspace(0,100,200)
chances = model(Xtest * lr.coef_ + lr.intercept_).ravel()


# ----------------------------------------------------------


plt.plot(Xtest , chances , color = "red" , linewidth= 3)
plt.axhline(y=0 , color='k' , linestyle= '-')
plt.axhline(y=1 , color='k' , linestyle= '-')
plt.axhline(y=0.5 , color='b' , linestyle= '--')

plt.ylabel('y')
plt.xlabel('X')
plt.xlim(0, 30)
# plt.show()


# ----------------------------------------------------------------------

velocity = float(input("Enter the velocity :- "))

chances = model(velocity * lr.coef_ + lr.intercept_).ravel()

if chances < 0.01:
    print(" Will Not get Escape. ")
elif chances >= 1:
    print(" Will Escape. ")
elif chances < 0.5:
    print(" Might not Escape. ")
else: 
    print(" Might Escape. ")


















