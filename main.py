import numpy as np
import random


INPUT_DIM = 3
OUT_DIM = 2
H_DIM = 15

dataset =   [(np.array([[0,0,0]]),0),
             (np.array([[0,0,1]]),1),
             (np.array([[0,1,0]]),1),
             (np.array([[0,1,1]]),0),
             (np.array([[1,0,0]]),1),
             (np.array([[1,0,1]]),0),
             (np.array([[1,1,0]]),0),
             (np.array([[1,1,1]]),1),
           
            ]


w1 = np.random.randn(INPUT_DIM, H_DIM)
b1 = np.random.randn(H_DIM)
w2 = np.random.randn(H_DIM, OUT_DIM)
b2 = np.random.randn(OUT_DIM)




def relu(t):
    return np.maximum(t, 0)

def softmax(t):
    out =  np.exp(t)
    return out / np.sum(out)

def sparce_cross_entropy(z, y):
    return -np.log(z[0,y])

def to_full(y, num_classes):
    y_full = np.zeros((1, num_classes))
    y_full[0,y] = 1
    return y_full

def relu_deriv(t):
    return (t >= 0).astype(float)


ALPHA = 0.01
NUM_EPOCHS = 100

loss = []

for ep in range(NUM_EPOCHS):
    for i in range(len(dataset)):

        x,y = dataset[i]
        
        # Forward

        t1 = x @ w1 + b1
        h1 = relu(t1)
        t2 = h1 @ w2 + b2
        z = softmax(t2)
        print(z)
        E = sparce_cross_entropy(z, y)

        # Backward

        y_full = to_full(y, OUT_DIM)

        dE_dt2 = z - y_full
        dE_dW2 = h1.T @ dE_dt2
        dE_db2 = dE_dt2
        dE_dh1 = dE_dt2 @ w2.T
        dE_dt1 = dE_dh1 * relu_deriv(t1)
        dE_dw1 = x.T @ dE_dt1
        dE_db1 = dE_dt1


        # Update
        w1 = w1 - ALPHA*dE_dw1
        b1 = b1 - ALPHA*dE_db1

        w2 = w2 - ALPHA*dE_dW2
        b2 = b2 - ALPHA*dE_db2
        
        loss.append(E)


def predict(x):
    t1 = x @ w1 + b1
    h1 = relu(t1)
    t2 = h1 @ w2 + b2
    z = softmax(t2)
    return z

def calc_acc():
    corr = 0
    for x,y in dataset:
        z = predict(x)
        y_pred = np.argmax(z)
        if y_pred == y:
            corr += 1
    acc = corr / len(dataset)
    return acc


acur = calc_acc()
print(acur)

import matplotlib.pyplot as plt

plt.plot(loss)
plt.show()

print(np.argmax(predict([0,0,0])))
print(np.argmax(predict([0,0,1])))
print(np.argmax(predict([0,1,0])))
print(np.argmax(predict([0,1,1])))
print(np.argmax(predict([1,0,0])))
print(np.argmax(predict([1,0,1])))
print(np.argmax(predict([1,1,0])))
print(np.argmax(predict([1,1,1])))
