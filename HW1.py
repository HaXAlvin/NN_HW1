import sys
import matplotlib
from matplotlib import ticker
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog as fd
from tkinter.messagebox import showinfo
from random import random, sample
matplotlib.use("TkAgg")

# Config Setting
TRAIN_SIZE = 2/3
LIMIT_EPOCH = 512
LOSS_SELECT = 0
MSE_LOSS = False
# GUI Setting
FILE_PATH = ""
LEARNING_RATE = 0.01
END_LOSS_RANGE = 0.01

# GUI callback
def save_var_callback():
    if not FILE_PATH:
        showinfo(title="Warning", message="Please select a file first!")
        return
    global END_LOSS_RANGE
    global LEARNING_RATE
    END_LOSS_RANGE = train_end_gate.get()
    LEARNING_RATE = train_rate.get()
    window.destroy()

def select_file_callback():
    global FILE_PATH
    path = fd.askopenfilenames(
        title='Select a file',
        filetypes=(("text files", "*.txt"),("All files", "*.*"))
    )
    fill_path.set(path[0])
    FILE_PATH = path[0]

# Setting GUI
print("Loading...")
window = tk.Tk()
window.title("Setting")
window.protocol("WM_DELETE_WINDOW", lambda : sys.exit(0))

train_end_gate = tk.DoubleVar(value=END_LOSS_RANGE)
train_end_gate_label = tk.Label(window, text="Train End Gate:")
train_end_gate_label.grid(row=0, column=0)
train_end_gate_input = tk.Entry(window,textvariable=train_end_gate)
train_end_gate_input.grid(row=0, column=1)

train_rate = tk.DoubleVar(value=LEARNING_RATE)
train_rate_label = tk.Label(window, text="Training Rate:")
train_rate_label.grid(row=1, column=0)
train_rate_input = tk.Entry(window,textvariable=train_rate)
train_rate_input.grid(row=1, column=1)

fill_path = tk.StringVar(value="/")
select_file_label = tk.Label(window, text="Data Set Path:")
select_file_label.grid(row=2, column=0)
select_file_input = tk.Entry(window,textvariable=fill_path,state='disabled')
select_file_input.grid(row=2, column=1)
select_file_button = tk.Button(window, text="Select File", command=select_file_callback)
select_file_button.grid(row=2,column=2, columnspan=2)

button = tk.Button(window, text="Run", command=save_var_callback)
button.grid(row=3,column=0, columnspan=3)
window.mainloop()
# Setting GUI Closed

# Load origin data and split to training data and validate data
print("Training...")
with open(FILE_PATH) as f:
    datas = f.readlines()
datas = [[float(s) for s in data.split()] for data in sample(datas,len(datas))]
xs1, xs2, zs = zip(*datas)
classes = list(set(zs))
assert len(classes) == 2
train_data = datas[:int(len(datas)*TRAIN_SIZE)]
val_data = datas[int(len(datas)*TRAIN_SIZE):]

# Common Function
def matrix_mul(w,x,x_bios=True):
    if x_bios:
        x = [-1] + x
    return sum(wi*xi for wi, xi in zip(w, x))

def round_3(x): return round(x, 3)

# Training Function
def sgn(x):
    if x >= 0:
        return classes[1]  # class 1
    if x < 0:
        return classes[0]  # class 0

def loss_fc(data, w, mse=MSE_LOSS):
    error_count = 0
    se = 0
    for x1, x2, y in data:
        v = matrix_mul(w, [x1, x2])
        se += (-1 if y == classes[0] else 1)*v**2
        error_count += 1 if y != sgn(v) else 0
    if mse:
        return error_count/len(data)/2+se/len(data)/2
    return error_count/len(data)

# Model
class Perceptron():
    def __init__(self, lr):
        self.lr = lr
        self.w = [-1, random(), random()]

    def train(self, x: list, y: int, lr=None):
        if lr is None:
            lr = self.lr
        x = [-1]+x  # data bias(index=0) always be -1

        wt_x = sum(wi*xi for wi, xi in zip(self.w, x))
        cla = sgn(wt_x)  # get class 0 or 1
        change = y-cla
        self.w = [wi+lr*change*xi for wi, xi in zip(self.w, x)]


# Training
perceptron = Perceptron(lr=LEARNING_RATE)
mini_loss = loss_fc(train_data, perceptron.w)
error_rate_list = [mini_loss]
val_error_rate_list = [loss_fc(val_data, perceptron.w)]
error_rate_w = perceptron.w
epoch = 0
while(epoch < LIMIT_EPOCH):
    epoch += 1
    for data in sample(train_data,len(train_data)):
        perceptron.train(data[:2], data[2])

        # recode for plt draw
        error_rate_list.append(loss_fc(train_data, perceptron.w))
        val_error_rate_list.append(loss_fc(val_data, perceptron.w))
        if error_rate_list[-1] <= mini_loss:
            mini_loss = error_rate_list[-1]
            error_rate_w = perceptron.w
    print(f"epoch: {epoch:02d}", f"w: {list(map(round_3,perceptron.w))}",f"loss: {error_rate_list[-1]}", sep='\t')
    if mini_loss<END_LOSS_RANGE: # train enough
        break
    if len(set(error_rate_list[-len(train_data):]))==1 and epoch>=128: # train can't improve
        break

# Two plot 
fig, (plt_loss, plt_data) = plt.subplots(2)

# First plot - train loss
mini_train_loss_index = len(error_rate_list) - error_rate_list[::-1].index(mini_loss) - 1
plt_loss.plot(list(map(lambda x: x/len(train_data), range(len(error_rate_list)))), error_rate_list, '-', label="Train Loss")
plt_loss.plot((mini_train_loss_index)/len(train_data), mini_loss, 'o',c="violet", label=f"Mini Train Loss")
plt_loss.annotate(round_3(mini_loss), ((mini_train_loss_index)/len(train_data), mini_loss), textcoords="offset points", xytext=(0,10), ha='center', c="violet")
# First plot - val loss
mini_val_loss_index = len(val_error_rate_list) - val_error_rate_list[::-1].index(min(val_error_rate_list)) -1
plt_loss.plot(list(map(lambda x: x/len(train_data), range(len(error_rate_list)))), val_error_rate_list, '-', label="Validate Loss")
plt_loss.plot((mini_val_loss_index)/len(train_data), min(val_error_rate_list), 'o',c="red", label=f"Mini Validate Loss")
plt_loss.annotate(round_3(min(val_error_rate_list)), ((mini_val_loss_index)/len(train_data), min(val_error_rate_list)), textcoords="offset points", xytext=(0,15), ha='center', c="red")
plt_loss.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

# Second plot - data point
blue_data = [data[:2] for data in train_data if data[2] == classes[0]]
orange_data = [data[:2] for data in train_data if data[2] == classes[1]]
if blue_data : plt_data.scatter(*zip(*blue_data), c='blue', label=f"Train Label={int(classes[0])}", alpha=1)
if orange_data : plt_data.scatter(*zip(*orange_data), c='orange', label=f"Train Label={int(classes[1])}", alpha=1)
violet_data = [data[:2] for data in val_data if data[2] == classes[0]]
red_data = [data[:2] for data in val_data if data[2] == classes[1]]
if violet_data : plt_data.scatter(*zip(*violet_data), c='violet', label=f"Validate Label={int(classes[0])}", alpha=1)
if red_data : plt_data.scatter(*zip(*red_data), c='red', label=f"Validate Label={int(classes[1])}", alpha=1)
plt_data.set_ylim([min(xs2)-0.5,max(xs2)+0.5])
# Second plot - W line
x = [min(xs1)-3, max(xs1)+3]
error_rate_y = [(error_rate_w[0]-error_rate_w[1]*xi)/error_rate_w[2] for xi in x]
plt_data.plot(x, error_rate_y, '-g', label=f'${round_3(error_rate_w[1])}x{round_3(error_rate_w[2]):+}y{round_3(-error_rate_w[0]):+}=0$\n$W={list(map(round_3,error_rate_w))}$', alpha=0.8)

# Show two plot
plt_loss.set(xlabel=r'$Epoch$', ylabel=r'$Loss=Error Rate$', title='Training Loss & Validate Loss')
plt_loss.legend()
plt_loss.grid(True)
plt_data.set(xlabel=r'${x}_1$', ylabel=r'${x}_2$', title='Classification Result')
plt_data.legend()
plt_data.grid(True)
fig.canvas.manager.set_window_title(FILE_PATH)
plt.show()

input("Click Enter to exit.")