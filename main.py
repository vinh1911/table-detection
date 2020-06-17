import numpy as np
import cv2
import pandas as pd
import utils
import json

def ocr(img):
    qrDecoder = cv2.QRCodeDetector()

    # Detect and decode the qrcode
    data,bbox,_ = qrDecoder.detectAndDecode(img)
    if len(data)>0:
        user = data.split(',')
        user_id = user[0]
        user_name = user[1]
        user_birthdate = user[2]
    else:
        user_id = 'QRCode not found'
        user_name = user_id
        user_birthdate = user_id
    # Convert resized RGB image to grayscale
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Apply gaussianblur
    img_blur = cv2.GaussianBlur(img_gray, (5, 5), 0)
    # Threshold the image
    img_th = cv2.adaptiveThreshold(img_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 199, 5)
    if bbox is not None:
        x, y, w, h = cv2.boundingRect(bbox)
        img_th = cv2.rectangle(img_th,(x-10,y-10),(x+w+10,y+h+10), (255,255,255), -1)
    #inverting the image 
    img_bin = 255-img_th

    # countcol(width) of kernel as 100th of total width
    kernel_len = np.array(img).shape[1]//100
    # Defining a vertical kernel to detect all vertical lines of image 
    ver_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_len))
    # Defining a horizontal kernel to detect all horizontal lines of image
    hor_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_len, 1))
    # A kernel of 2x2
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))

    #Use vertical kernel to detect and save the vertical lines
    image_1 = cv2.erode(img_bin, ver_kernel, iterations=3)
    vertical_lines = cv2.dilate(image_1, ver_kernel, iterations=3)

    #Use horizontal kernel to detect and save the horizontal lines
    image_2 = cv2.erode(img_bin, hor_kernel, iterations=3)
    horizontal_lines = cv2.dilate(image_2, hor_kernel, iterations=3)

    # Combine horizontal and vertical lines in a new third image, with both having same weight.
    img_vh = cv2.addWeighted(vertical_lines, 0.5, horizontal_lines, 0.5, 0.0)
    #Eroding and thesholding the image
    img_vh = cv2.erode(~img_vh, kernel, iterations=2)
    thresh, img_vh = cv2.threshold(img_vh,128,255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    bitxor = cv2.bitwise_xor(img_gray,img_vh)

    # Detect contours for following box detection
    contours, hierarchy = cv2.findContours(img_vh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Sort all the contours by top to bottom.
    contours, boundingBoxes = utils.sort_contours(contours, method="top-to-bottom")

    #Creating a list of heights for all detected boxes
    heights = [boundingBoxes[i][3] for i in range(len(boundingBoxes))]

    #Get mean of heights
    mean = np.mean(heights)

    #Create list box to store all boxes in  
    box = []

    # Get position (x,y), width and height for every cells
    # The value in this case is to to neglect bounding box which might be no cells (e.g. the table it self)
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if (160<w<1000 and 80<h<500): # this could vary depending on the size of the image
            # image = cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
            box.append([x,y,w,h])

    #Creating two lists to define row and column in which cell is located
    row=[]
    column=[]
    j=0

    #Sorting the boxes to their respective row and column
    for i in range(len(box)):    
            
        if(i==0):
            column.append(box[i])
            previous=box[i]    
        
        else:
            if(box[i][1]<=previous[1]+mean/2):
                column.append(box[i])
                previous=box[i]            
                
                if(i==len(box)-1):
                    row.append(column)        
                
            else:
                row.append(column)
                column=[]
                previous = box[i]
                column.append(box[i])
                
    # print(column)
    # print(row)

    #calculating maximum number of cells
    countcol = 0
    for i in range(len(row)):
        countcol = len(row[i])
        if countcol > countcol:
            countcol = countcol

    #Retrieving the center of each column
    center = [int(row[i][j][0]+row[i][j][2]/2) for j in range(len(row[i])) if row[0]]

    center=np.array(center)
    center.sort()

    #Regarding the distance to the columns center, the boxes are arranged in respective order

    finalboxes = []
    for i in range(len(row)):
        lis=[]
        for k in range(countcol):
            lis.append([])
        for j in range(len(row[i])):
            diff = abs(center-(row[i][j][0]+row[i][j][2]/4))
            minimum = min(diff)
            indexing = list(diff).index(minimum)
            lis[indexing].append(row[i][j])
        finalboxes.append(lis)


    #from every single image-based cell/box the strings are extracted via pytesseract and stored in a list
    outer=[]
    for i in range(1,len(finalboxes)):
        for j in range(len(finalboxes[i])):
            if(len(finalboxes[i][j])==0):
                outer.append(' ')
            else:
                for k in range(len(finalboxes[i][j])):
                    y,x,w,h = finalboxes[i][j][k][0],finalboxes[i][j][k][1], finalboxes[i][j][k][2],finalboxes[i][j][k][3]
                    padding = 5 # croping a little bit inside to avoid the borders
                    finalimg = img_th[x+padding:x+h-padding, y+padding:y+w-padding]
                    border = cv2.copyMakeBorder(finalimg,2,2,2,2, cv2.BORDER_CONSTANT,value=[255,255])
                    resizing = cv2.resize(border, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
                    if (i != 0):
                        out = utils.predict_number(resizing)
                outer.append(out) 

    #Creating a dataframe of the generated OCR list
    arr = np.array(outer)
    tag = ['day','month','year','beforeBreakfast','afterBreakfast','beforeLunch','afterLunch','beforeDinner','afterDinner']
    df = pd.DataFrame(arr.reshape(len(row)-1, countcol),columns = tag)
    df['date']= df['day']+'/'+ df['month']+'/'+df['year']
    df.insert(0,'count',df.index)
    d = df.to_json(orient='records')
    user_data= json.loads(d)
    return user_id,user_name,user_birthdate,user_data

# if __name__ == '__main__':
#     print(ocr(source))