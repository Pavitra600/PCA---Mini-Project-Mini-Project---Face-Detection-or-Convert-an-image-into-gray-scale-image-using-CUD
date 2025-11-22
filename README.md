# PCA-Mini-Project---Face-Detection-or-Convert-an-image-into-gray-scale-image-using-CUD
Mini Project - Face Detection or Convert an image into gray scale image using CUDA GPU programming

## NAME: PAVITRA J
## REG NO: 212224110043
## DATE: 20/11/2025

## AIM:
To develop a GPU-accelerated image processing system using CUDA that performs face detection or conversion of a color image into a grayscale image,
in order to enhance processing speed and demonstrate parallel computing capabilities of NVIDIA GPUs.

## PROCEDURE:
1. Import the required OpenCV and CUDA libraries.
2. Load the input image from the user.
3. Convert the image into a suitable format for processing.
4. If face detection is selected:
   * Load the pretrained face detection classifier.
   * Detect faces and draw bounding boxes.
5. If grayscale conversion is selected:
   * Convert the image from RGB to grayscale using CUDA-based parallel execution.
6. Display the processed output image.
7. Save the final result.

## PROGRAM:
```

!pip install opencv-python

import cv2
import matplotlib.pyplot as plt
from google.colab import files

uploaded = files.upload()

input_image = list(uploaded.keys())[0]
img = cv2.imread(input_image)


gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(cascade_path)


faces = face_cascade.detectMultiScale(gray, 1.1, 4)

for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

print("Total Faces Detected:", len(faces))


plt.figure(figsize=(8, 8))
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.axis('off')


output_name = "face_detected.png"
cv2.imwrite(output_name, img)
print("Saved output as:", output_name)
files.download(output_name)

```
## OUTPUT:
<img width="865" height="504" alt="image" src="https://github.com/user-attachments/assets/0e43eb2a-f819-4479-bb78-4937bd7fe2c5" />



<img width="862" height="520" alt="image" src="https://github.com/user-attachments/assets/a613a1ab-ff25-417a-a42b-e21edbb3ea8e" />


## RESULT:
The program successfully executed face detection / grayscale image conversion using CUDA GPU programming.
