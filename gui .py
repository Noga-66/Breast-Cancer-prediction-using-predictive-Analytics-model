import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import numpy as np
from PIL import Image, ImageOps, ImageTk
import tkinter as tk
from tkinter import filedialog

# -----------------------------
# 1) تحميل MNIST وتهيئته
# -----------------------------
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train / 255.0
x_test  = x_test  / 255.0
x_train = x_train.reshape(-1,28,28,1)
x_test  = x_test.reshape(-1,28,28,1)

# -----------------------------
# 2) بناء موديل CNN
# -----------------------------
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# -----------------------------
# 3) تدريب الموديل
# -----------------------------
print("Training the model...")
model.fit(x_train, y_train, epochs=5, validation_split=0.1)

# -----------------------------
# 4) تقييم الموديل
# -----------------------------
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"Test Accuracy: {test_acc:.4f}")

# -----------------------------
# 5) حفظ الموديل
# -----------------------------
model.save("mnist_cnn_model.h5")
print("Model saved successfully!")

# -----------------------------
# 6) Preprocessing function
# -----------------------------
def preprocess_image(path):
    img = Image.open(path).convert('L')          # grayscale
    img = ImageOps.invert(img)                   # يقلب الألوان لو الخلفية بيضاء
    img = img.resize((28,28))
    img_array = np.array(img)/255.0
    img_array = img_array.reshape(1,28,28,1)    # batch + channel
    return img_array

# -----------------------------
# 7) GUI للتجربة
# -----------------------------
def open_image():
    path = filedialog.askopenfilename()
    if path:
        img_array = preprocess_image(path)
        pred = model.predict(img_array)
        digit = np.argmax(pred)

        # عرض الصورة
        img_display = Image.open(path).resize((100,100))
        img_display = ImageTk.PhotoImage(img_display)
        canvas.create_image(50,50,image=img_display)
        canvas.image = img_display

        label_result.config(text=f"Predicted Digit: {digit}")

root = tk.Tk()
root.title("MNIST Digit Recognizer")

btn = tk.Button(root, text="Open Image", command=open_image)
btn.pack()

canvas = tk.Canvas(root, width=100, height=100)
canvas.pack()

label_result = tk.Label(root, text="Predicted Digit: ")
label_result.pack()

root.mainloop()