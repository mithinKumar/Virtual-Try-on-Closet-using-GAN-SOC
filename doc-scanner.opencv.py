#!/usr/bin/env python
# coding: utf-8

# In[2]:


#doc scanner using web cam of laptop


import cv2
import numpy as np



def preprocess_image(image):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # Use Canny edge detection
    edges = cv2.Canny(blurred, 50, 150)
    return edges
def find_contours(blurred):
    kernel = np.ones((5, 5), np.uint8)  # You can adjust the size of the kernel
    blurred = cv2.morphologyEx(blurred, cv2.MORPH_CLOSE, kernel)
    # Find contours
    contours, _ = cv2.findContours(blurred, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    # Sort contours by area in descending order
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    for contour in contours:
        # Approximate the contour
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.09 * peri, True)
        # If the contour has 4 points, we can assume it is a rectangle
        if len(approx) == 4:
            return approx
    return None
def perspective_warp(image, points):
    points = points.reshape(4, 2)
    # Order points: [top-left, top-right, bottom-right, bottom-left]
    rect = np.zeros((4, 2), dtype="float32")
    s = points.sum(axis=1)
    rect[0] = points[np.argmin(s)]
    rect[2] = points[np.argmax(s)]
    diff = np.diff(points, axis=1)
    rect[1] = points[np.argmin(diff)]
    rect[3] = points[np.argmax(diff)]
    
    # Get the width and height of the new image
    (tl, tr, br, bl) = rect
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    
    # Destination points for top-down view
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")
    
    # Get the perspective transform matrix and apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped
def main():
    cap = cv2.VideoCapture(0)
    document_detected = False  # Flag to track if a document is detected
    
    while not document_detected:
        ret, frame = cap.read()
        if not ret:
            break

        # Preprocess the image
        edges = preprocess_image(frame)
        
        # Find contours
        doc_contour = find_contours(edges)
        
        if doc_contour is not None:
            # Perform perspective warp
            warped = perspective_warp(frame, doc_contour)
            cv2.imshow('Warped Image', warped)
            break  # Set the flag to True to break the loop
        
        # Display the original frame
        cv2.imshow('Original Image', frame)
   

        # Exit on pressing 'q'
    while True:
        cv2.imshow('Warped Image', warped)
        cv2.imshow('Original Image', frame)
        cv2.imshow('Original Ima', edges)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    
    main()


# In[3]:


#This uses a mobile camera via ip camera app

import cv2
import numpy as np
import requests 
import imutils 

def preprocess_image(image):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # Use Canny edge detection
    edges = cv2.Canny(blurred, 50, 150)
    return edges
def find_contours(blurred):
    kernel = np.ones((5, 5), np.uint8)  # You can adjust the size of the kernel
    blurred = cv2.morphologyEx(blurred, cv2.MORPH_CLOSE, kernel)
    # Find contours
    contours, _ = cv2.findContours(blurred, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    # Sort contours by area in descending order
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    for contour in contours:
        # Approximate the contour
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
        # If the contour has 4 points, we can assume it is a rectangle
        if len(approx) == 4:
            return approx
    return None
def perspective_warp(image, points):
    points = points.reshape(4, 2)
    # Order points: [top-left, top-right, bottom-right, bottom-left]
    rect = np.zeros((4, 2), dtype="float32")
    s = points.sum(axis=1)
    rect[0] = points[np.argmin(s)]
    rect[2] = points[np.argmax(s)]
    diff = np.diff(points, axis=1)
    rect[1] = points[np.argmin(diff)]
    rect[3] = points[np.argmax(diff)]
    
    # Get the width and height of the new image
    (tl, tr, br, bl) = rect
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    
    # Destination points for top-down view
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")
    
    # Get the perspective transform matrix and apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped
def main():
    cap = cv2.VideoCapture(0)
    document_detected = False  # Flag to track if a document is detected


  
# Replace the below URL with your own. Make sure to add "/shot.jpg" at last. 
    url = "http://192.168.1.3:8080/shot.jpg"
  
# While loop to continuously fetching data from the Url 


    
    
    while not document_detected:
        img_resp = requests.get(url) 
        img_arr = np.array(bytearray(img_resp.content), dtype=np.uint8) 
        frame = cv2.imdecode(img_arr, -1) 
        frame = imutils.resize(frame, width=1000, height=1800) 
        

        # Preprocess the image
        edges = preprocess_image(frame)
        
        # Find contours
        doc_contour = find_contours(edges)
        
        if doc_contour is not None:
            # Perform perspective warp
            warped = perspective_warp(frame, doc_contour)
            cv2.imshow('Warped Image', warped)
            break  # Set the flag to True to break the loop
        
        # Display the original frame
        cv2.imshow('Original Image', frame)
   

        # Exit on pressing 'q'
    while True:
        cv2.imshow('Warped Image', warped)
        cv2.imshow('Original Image', frame)
        cv2.imshow('Original Ima', edges)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    
    main()


# In[ ]:





# In[ ]:




