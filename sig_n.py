import numpy as np

INPUT_DIM = 3
OUT_DIM = 2

L1_DIM = 100
L2_DIM = 100

dataset =   [(np.array([[0,0,0]]),0),
             (np.array([[0,0,1]]),1),
             (np.array([[0,1,0]]),1),
             (np.array([[0,1,1]]),0),
             (np.array([[1,0,0]]),1),
             (np.array([[1,0,1]]),0),
             (np.array([[1,1,0]]),0),
             (np.array([[1,1,1]]),1),
           
            ]

w1 = np.random.randn(INPUT_DIM, L1_DIM)
b1 = np.random.randn(L1_DIM)
w2 = np.random.randn(L1_DIM, L2_DIM)
b2 = np.random.randn(L2_DIM)
w3 = np.random.randn(L2_DIM, OUT_DIM)
b3 = np.random.randn(OUT_DIM)

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
NUM_EPOCHS = 10

loss = []

for ep in range(NUM_EPOCHS):
    for i in range(len(dataset)):
        # Forw

        x,y = dataset[i]

        i1 = x @ w1 + b1
        o1 = relu(i1)
        i2 = o1 @ w2 + b2
        o2 = relu(i2)
        i3 = o2 @ w3 + b3
        z = softmax(i3)
        E = sparce_cross_entropy(z, y)

        # Back

        y_full = to_full(y, OUT_DIM)

        dE_dt3 = z - y_full
        dE_dw3 = o2.T @ dE_dt3
        dE_db3 = dE_dt3
        dE_dh2 = dE_dt3 @ w3.T
        dE_dt2 = dE_dh2 * relu_deriv(i2)
        dE_dw2 = o1.T @ dE_dt2
        dE_db2 = dE_dt2
        dE_dh1 = dE_dt2 @ w2.T
        dE_dt1 = dE_dh1 * relu_deriv(i1)
        dE_dw1 = x.T @ dE_dt1
        dE_db1 = dE_dt1
        
        # Updat

        w1 = w1 - ALPHA*dE_dw1
        b1 = b1 - ALPHA*dE_db1

        w2 = w2 - ALPHA*dE_dw2
        b2 = b2 - ALPHA*dE_db2

        w3 = w3 - ALPHA*dE_dw3
        b3 = b3 - ALPHA*dE_db3
        
        loss.append(E)

def predict(x):
    i1 = x @ w1 + b1
    o1 = relu(i1)
    i2 = o1 @ w2 + b2
    o2 = relu(i2)
    i3 = o2 @ w3 + b3
    z = softmax(i3)
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