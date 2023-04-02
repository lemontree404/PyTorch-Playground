from flask import Flask,request,render_template,redirect,send_file,jsonify
import mysql.connector
import torch
from io import BytesIO
import numpy as np
import matplotlib.pyplot as plt
import math
import json
import base64

connection = mysql.connector.connect(host="localhost",port=3306,database="mlcia",user="root",password="jojo")

cursor = connection.cursor(dictionary=True)

app = Flask(__name__,static_folder="static")

class Net(torch.nn.Module):
    def __init__(self, input_size, layers):
        super().__init__()
        self.layers = torch.nn.ModuleList()
        for i, layer in enumerate(layers):
            if i == 0:
                self.layers.append(torch.nn.Linear(input_size, layer["units"]))
            else:
                self.layers.append(torch.nn.Linear(layers[i-1]["units"], layer["units"]))
            if layer["activation"] == "relu":
                self.layers.append(torch.nn.ReLU())
            elif layer["activation"] == "sigmoid":
                self.layers.append(torch.nn.Sigmoid())
            elif layer["activation"] == "tanh":
                self.layers.append(torch.nn.Tanh())
        self.layers.append(torch.nn.Linear(layers[-1]["units"], 1))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

def create_model(problem, layers,epochs):
    if problem == "sin_cos":
        x_vals = np.arange(-2*math.pi, 2*math.pi, 0.1)
        y_vals = np.sin(2*x_vals) + np.cos(x_vals)
    elif problem == "sin_squared":
        x_vals = np.arange(-2*math.pi, 2*math.pi, 0.1)
        y_vals = x_vals * np.sin(x_vals**2 / 2)
    elif problem == "signal":
        x_vals = np.arange(-2*math.pi, 2*math.pi, 0.1)
        y_vals = np.sin(4*x_vals) + 0.8 * np.cos(2*x_vals) + 1.2 * np.cos(4*x_vals)    
    else:
        raise ValueError(f"Unknown problem: {problem}")
    
    x_tensor = torch.tensor(x_vals, dtype=torch.float32).view(-1, 1)
    y_tensor = torch.tensor(y_vals, dtype=torch.float32).view(-1, 1)

    net = Net(1, layers)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.01)

    for epoch in range(epochs):
        optimizer.zero_grad()
        y_pred = net(x_tensor)
        loss = criterion(y_pred, y_tensor)
        loss.backward()
        optimizer.step()

    net.eval()
    with torch.no_grad():
        y_pred = net(x_tensor)
        predictions = y_pred.numpy().flatten().tolist()
        ground_truth = y_tensor.numpy().flatten().tolist()

    return {
        "x_vals": x_vals.tolist(),
        "predictions": predictions,
        "ground_truth": ground_truth
    }

@app.route("/")
def home():
    return render_template("login.html")
    

@app.route("/login",methods=["GET","POST"])
def login():
    cursor.execute("select * from login")
    rows = cursor.fetchall()
    global db
    db= {}
    for i in rows:
        db[i["Usr"]] = i["pwd"]

    username = request.form["username"]
    password = request.form["password"]

    if username in db:
        if db[username] == password:
            return redirect("/home")
        else:
            return render_template("login.html",msg="Incorrect Password")
    else:
        return render_template("login.html",msg="Invalid User")

@app.route("/signup",methods=["POST","GET"])
def signup():
    return render_template("register.html")

@app.route("/register",methods=["POST","GET"])
def register():
    username = request.form["username"]
    password = request.form["password"]

    cursor.execute("select * from login")
    rows = cursor.fetchall()
    db= {}
    for i in rows:
        db[i["Usr"]] = i["pwd"]

    if username and password and username not in db:
        cursor.execute(f"insert into login values (%s,%s)",[username,password])
        connection.commit()
        return redirect("/home")
    
    return render_template("register.html",msg="Already Registered")


@app.route("/home")
def index():
    return render_template("index.html")

@app.route('/train', methods=['POST'])
def train():
    epochs = int(request.form["epochs"])
    problem = request.form['problem']
    layers = []
    # print(len(request.form[f'activation{i}']))
    i=0
    while True:
        try:
            activation = request.form[f'activation{i}']
            neurons = int(request.form[f'neurons{i}'])
            layers.append({"units":neurons, "activation":activation})
            i+=1
        except:
            break

    arrs = create_model(problem,layers,epochs)
    plt.clf()
    plt.plot(arrs["x_vals"],arrs["ground_truth"])
    plt.plot(arrs["x_vals"],arrs["predictions"])
    plt.legend(["Ground Truth","Predictions"])
    img = BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)

    encoded_img = base64.b64encode(img.getvalue()).decode('utf-8')
    html = '<img src="data:image/png;base64, {}">'.format(encoded_img)

    return render_template('index.html', graph_html=html)

if __name__=="__main__":
    app.run(debug=True,port=8000)