#python 3.7
#python -m pip install --upgrade pip
#python -m pip install  opencv-python
#python -m pip install  matplotlib
#python -m pip install  gtts
#python -m pip install  playsound
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
#from gtts import gTTS
#from playsound import playsound

def plotImg(img):
    if len(img.shape) == 2:
        plt.imshow(img, cmap='gray')
        plt.show()
    else:
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.show()

print("Enter Filename:")
fn=input()
source="Sample_Inputs/"+fn+".png"
img = cv2.imread(source)

cv2.imshow("Input image",img)
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
binary_img = cv2.adaptiveThreshold(gray_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 131, 15)
#plotImg(binary_img)

"""
im = binary_img
ret, thresh = cv2.threshold(im, 150, 255, cv2.THRESH_BINARY)
kernel = np.ones((5, 5), np.uint8)
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
cleaned = morphology.remove_small_objects(opening, min_size=62, connectivity=2)
#cv2.imshow("cleaned", cleaned)
binary_img=cleaned
"""

found=0
img1=cv2.imread(source)
# Convert it to HSV
folder='Bacterial blight'
for filename in os.listdir(folder):
    img2 = cv2.imread(os.path.join(folder,filename))
    if img2 is not None:
        img1_hsv = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)
        img2_hsv = cv2.cvtColor(img2, cv2.COLOR_BGR2HSV)

# Calculate the histogram and normalize it
    hist_img1 = cv2.calcHist([img1_hsv], [0,1], None, [180,256], [0,180,0,256])
    cv2.normalize(hist_img1, hist_img1, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX);
    hist_img2 = cv2.calcHist([img2_hsv], [0,1], None, [180,256], [0,180,0,256])
    cv2.normalize(hist_img2, hist_img2, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX);

# find the metric value
    metric_val = cv2.compareHist(hist_img1, hist_img2, cv2.HISTCMP_BHATTACHARYYA)
    if metric_val == 0.0:
        print("Bacterial blight")
        cv2.imshow("Bacterial blight",img1)
        found=found+1
#print("\n")
_, _, boxes, _ = cv2.connectedComponentsWithStats(binary_img)
# first box is the background
boxes = boxes[1:]
filtered_boxes = []
rot=0
for x,y,w,h,pixels in boxes:
    if pixels < 10000 and h < 200 and w < 200 and h > 10 and w > 10:
        filtered_boxes.append((x,y,w,h))
        rot+=1

for x,y,w,h in filtered_boxes:
    cv2.rectangle(img, (x,y), (x+w,y+h), (0,0,255),2)

if found==0:
    plotImg(img)

folder='Bacterial leaf streak'
for filename in os.listdir(folder):
    img2 = cv2.imread(os.path.join(folder,filename))
    if img2 is not None:
        img1_hsv = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)
        img2_hsv = cv2.cvtColor(img2, cv2.COLOR_BGR2HSV)


# Calculate the histogram and normalize it
    hist_img1 = cv2.calcHist([img1_hsv], [0,1], None, [180,256], [0,180,0,256])
    cv2.normalize(hist_img1, hist_img1, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX);
    hist_img2 = cv2.calcHist([img2_hsv], [0,1], None, [180,256], [0,180,0,256])
    cv2.normalize(hist_img2, hist_img2, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX);

# find the metric value
    metric_val = cv2.compareHist(hist_img1, hist_img2, cv2.HISTCMP_BHATTACHARYYA)
    if metric_val == 0.0:
        print("Bacterial leaf streak")
        cv2.imshow("Bacterial leaf streak",img)
        found=found+1
#print("\n")

folder='Blast'
for filename in os.listdir(folder):
    img2 = cv2.imread(os.path.join(folder,filename))
    if img2 is not None:
        img1_hsv = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)
        img2_hsv = cv2.cvtColor(img2, cv2.COLOR_BGR2HSV)
   

# Calculate the histogram and normalize it
    hist_img1 = cv2.calcHist([img1_hsv], [0,1], None, [180,256], [0,180,0,256])
    cv2.normalize(hist_img1, hist_img1, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX);
    hist_img2 = cv2.calcHist([img2_hsv], [0,1], None, [180,256], [0,180,0,256])
    cv2.normalize(hist_img2, hist_img2, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX);

# find the metric value
    metric_val = cv2.compareHist(hist_img1, hist_img2, cv2.HISTCMP_BHATTACHARYYA)
    if metric_val == 0.0:
        print("Blast")
        cv2.imshow("Blast",img)
        
        Image1 = cv2.imread('./blast/b1.png')
        Image1=cv2.resize(Image1, dsize=(100, 100))
        Image2 = cv2.imread('./blast/b2.png')
        Image2=cv2.resize(Image2, dsize=(100, 100))
        Image3 = cv2.imread('./blast/b3.png')
        Image3=cv2.resize(Image3, dsize=(100, 100))
        Image4 = cv2.imread('./blast/b4.png')
        Image4=cv2.resize(Image4, dsize=(100, 100))
        """
        Hori = np.concatenate((Image1, Image2,Image3,Image4), axis=1)
        #Verti = np.concatenate((Image3, Image4), axis=0)
        cv2.namedWindow("Resized_Window", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Resized_Window", 600,600)
        cv2.imshow('Resized_Window', Hori)
        #cv2.imshow('VERTICAL', Verti)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        """
        rows = 4
        columns =4

        fig = plt.figure(figsize=(8, 8))
        fig.add_subplot(rows, columns, 1)
        plt.imshow(Image1)
        plt.axis('off')
        plt.title("Blast: First")
        fig.add_subplot(rows, columns, 2)
        plt.imshow(Image2)
        plt.axis('off')
        plt.title("Second")
        fig.add_subplot(rows, columns, 3)
        plt.imshow(Image3)
        plt.axis('off')
        plt.title("Third")
        fig.add_subplot(rows, columns, 4)
        plt.imshow(Image4)
        plt.axis('off')
        plt.title("Fourth")
        plt.show()
        found=found+1
#print("\n")

folder='Brown spot'
for filename in os.listdir(folder):
    img2 = cv2.imread(os.path.join(folder,filename))
    if img2 is not None:
        img1_hsv = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)
        img2_hsv = cv2.cvtColor(img2, cv2.COLOR_BGR2HSV)


# Calculate the histogram and normalize it
    hist_img1 = cv2.calcHist([img1_hsv], [0,1], None, [180,256], [0,180,0,256])
    cv2.normalize(hist_img1, hist_img1, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX);
    hist_img2 = cv2.calcHist([img2_hsv], [0,1], None, [180,256], [0,180,0,256])
    cv2.normalize(hist_img2, hist_img2, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX);

# find the metric value
    metric_val = cv2.compareHist(hist_img1, hist_img2, cv2.HISTCMP_BHATTACHARYYA)
    if metric_val == 0.0:
        print("Brown spot")
        cv2.imshow("Brown spot",img)
        Image1 = cv2.imread('./brown spot/bs1.png')
        Image2 = cv2.imread('./brown spot/bs2.png')
        Image3 = cv2.imread('./brown spot/bs3.png')
        Image4 = cv2.imread('./brown spot/bs4.png')
        rows = 4
        columns =4

        fig = plt.figure(figsize=(8, 8))
        fig.add_subplot(rows, columns, 1)
        plt.imshow(Image1)
        plt.axis('off')
        plt.title("Brown spot: First")
        fig.add_subplot(rows, columns, 2)
        plt.imshow(Image2)
        plt.axis('off')
        plt.title("Second")
        fig.add_subplot(rows, columns, 3)
        plt.imshow(Image3)
        plt.axis('off')
        plt.title("Third")
        fig.add_subplot(rows, columns, 4)
        plt.imshow(Image4)
        plt.axis('off')
        plt.title("Fourth")
        plt.show()

        found=found+1
#print("\n")


folder='False smut'
for filename in os.listdir(folder):
    img2 = cv2.imread(os.path.join(folder,filename))
    if img2 is not None:
        img1_hsv = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)
        img2_hsv = cv2.cvtColor(img2, cv2.COLOR_BGR2HSV)
   

# Calculate the histogram and normalize it
    hist_img1 = cv2.calcHist([img1_hsv], [0,1], None, [180,256], [0,180,0,256])
    cv2.normalize(hist_img1, hist_img1, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX);
    hist_img2 = cv2.calcHist([img2_hsv], [0,1], None, [180,256], [0,180,0,256])
    cv2.normalize(hist_img2, hist_img2, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX);

# find the metric value
    metric_val = cv2.compareHist(hist_img1, hist_img2, cv2.HISTCMP_BHATTACHARYYA)
    if metric_val == 0.0:
        print("False smut")
		
        cv2.imshow("False smut",img)
                
        Image1 = cv2.imread('./False smut/fs1.png')
        Image2 = cv2.imread('./False smut/fs2.png')
        Image3 = cv2.imread('./False smut/fs3.png')
        Image4 = cv2.imread('./False smut/fs4.png')
        rows = 4
        columns =4

        fig = plt.figure(figsize=(8, 8))
        fig.add_subplot(rows, columns, 1)
        plt.imshow(Image1)
        plt.axis('off')
        plt.title("False smut: First")
        fig.add_subplot(rows, columns, 2)
        plt.imshow(Image2)
        plt.axis('off')
        plt.title("Second")
        fig.add_subplot(rows, columns, 3)
        plt.imshow(Image3)
        plt.axis('off')
        plt.title("Third")
        fig.add_subplot(rows, columns, 4)
        plt.imshow(Image4)
        plt.axis('off')
        plt.title("Fourth")
        plt.show()

        found=found+1
#print("\n")

folder='Sheath blight'
for filename in os.listdir(folder):
    img2 = cv2.imread(os.path.join(folder,filename))
    if img2 is not None:
        img1_hsv = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)
        img2_hsv = cv2.cvtColor(img2, cv2.COLOR_BGR2HSV)
   

# Calculate the histogram and normalize it
    hist_img1 = cv2.calcHist([img1_hsv], [0,1], None, [180,256], [0,180,0,256])
    cv2.normalize(hist_img1, hist_img1, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX);
    hist_img2 = cv2.calcHist([img2_hsv], [0,1], None, [180,256], [0,180,0,256])
    cv2.normalize(hist_img2, hist_img2, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX);

# find the metric value
    metric_val = cv2.compareHist(hist_img1, hist_img2, cv2.HISTCMP_BHATTACHARYYA)
    if metric_val == 0.0:
        print("Sheath blight")
        cv2.imshow("Sheath blight",img)
        Image1 = cv2.imread('./Sheath blight/sb1.png')
        Image2 = cv2.imread('./Sheath blight/sb2.png')
        Image3 = cv2.imread('./Sheath blight/sb3.png')
        Image4 = cv2.imread('./Sheath blight/sb4.png')
        rows = 4
        columns =4

        fig = plt.figure(figsize=(8, 8))
        fig.add_subplot(rows, columns, 1)
        plt.imshow(Image1)
        plt.axis('off')
        plt.title("Sheath blight: First")
        fig.add_subplot(rows, columns, 2)
        plt.imshow(Image2)
        plt.axis('off')
        plt.title("Second")
        fig.add_subplot(rows, columns, 3)
        plt.imshow(Image3)
        plt.axis('off')
        plt.title("Third")
        fig.add_subplot(rows, columns, 4)
        plt.imshow(Image4)
        plt.axis('off')
        plt.title("Fourth")
        plt.show()
        found=found+1
#print("\n")


if found==0:
    print('No Disease/Less Rot Area')
    if rot<=10:
        cv2.imshow("Blast",img)
    if rot>10 and rot<=20:
        cv2.imshow("Bacterial leaf streak",img)
    if rot>20:
        cv2.imshow("Bacterial blight",img)

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Set the path to your dataset folder
dataset_path = 'Dataset'

# Define parameters
batch_size = 32
image_size = (256, 256)  # Adjust the size based on your dataset

# Create an ImageDataGenerator with data augmentation
datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2  # Split the dataset into 80% training and 20% validation
)

# Load and preprocess the training set
train_generator = datagen.flow_from_directory(
    dataset_path,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',  # Assumes categorical labels
    subset='training'  # Set as training data
)

# Load and preprocess the validation set
validation_generator = datagen.flow_from_directory(
    dataset_path,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'  # Set as validation data
)

# Build the CNN model
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(image_size[0], image_size[1], 3)),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(6, activation='softmax')  # Assuming 6 classes, adjust accordingly
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
# Train the model
import matplotlib.pyplot as plt

# Train the model
epochs = 50  # You can adjust the number of epochs
history = model.fit(train_generator, epochs=epochs, validation_data=validation_generator)

# Print final accuracy
final_accuracy = history.history['accuracy'][-1] * 100
print(f"Final Training Accuracy: {final_accuracy:.2f}%")

# Plot training accuracy
plt.figure(figsize=(10, 5))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.title('Training Accuracy over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
