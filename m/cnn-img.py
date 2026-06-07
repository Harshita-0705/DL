import cv2
import numpy as np

img = cv2.imread("image.jpg", 0)

kernel = np.ones((3,3)) / 9
stride = 2

h, w = img.shape
out = []

for i in range(0, h-2, stride):
    row = []
    for j in range(0, w-2, stride):
        val = np.sum(img[i:i+3, j:j+3] * kernel)
        row.append(val)
    out.append(row)

conv = np.array(out)

relu = np.maximum(0, conv)

pool = []
for i in range(0, relu.shape[0]-1, 2):
    row = []
    for j in range(0, relu.shape[1]-1, 2):
        row.append(np.max(relu[i:i+2, j:j+2]))
    pool.append(row)

pool = np.array(pool)

flat = pool.flatten()
w = np.random.rand(len(flat))
b = np.random.rand()

result = np.dot(flat, w) + b
print("Final Output:", result)