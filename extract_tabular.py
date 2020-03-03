#Loading all required libraries 
from cv2 import cv2
import numpy as np 
import pandas as pd
import pytesseract
import statistics

# config and variables
source = 'input/example1.jpg' # <---------- change input here
pytesseract.pytesseract.tesseract_cmd = r'C:\Users\USER\AppData\Local\Tesseract-OCR\tesseract.exe'

# function to sort contours by its x-axis (top to bottom)
def sort_contours(cnts, method="left-to-right"):
	# initialize the reverse flag and sort index
	reverse = False
	i = 0
 
	# handle if we need to sort in reverse
	if method == "right-to-left" or method == "bottom-to-top":
		reverse = True
 
	# handle if we are sorting against the y-coordinate rather than the x-coordinate of the bounding box
	if method == "top-to-bottom" or method == "bottom-to-top":
		i = 1
 
	# construct the list of bounding boxes and sort them from top to bottom
	boundingBoxes = [cv2.boundingRect(c) for c in cnts]
	(cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes), key=lambda b:b[1][i], reverse=reverse))
 
	# return the list of sorted contours and bounding boxes
	return (cnts, boundingBoxes)

def main(input):
    # loading image form directory
    img_original = cv2.imread(input,cv2.IMREAD_GRAYSCALE)
    img_original.shape

    # for adding border to an image
    cv2.resize(img_original, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    img_initial= cv2.copyMakeBorder(img_original,50,50,50,50,cv2.BORDER_CONSTANT,value=[255,255])
    cv2.imwrite('output/steps/init.png',img_initial)

    # Thresholding the image
    img_original = cv2.GaussianBlur(img_original,(5,5),0)
    (thresh, img_thresholded) = cv2.threshold(img_initial, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    cv2.resize(img_thresholded, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    cv2.imwrite('output/steps/1.png',img_thresholded)

    # to flip image pixel values
    img_thresholded_inverted = 255-img_thresholded
    cv2.resize(img_thresholded_inverted, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    cv2.imwrite('output/steps/2.png',img_thresholded_inverted)

    # initialize kernels for table boundaries detections
    if(img_thresholded_inverted.shape[0]<1000):
        ver = np.array([[1],
                [1],
                [1],
                [1],
                [1],
                [1],
                [1]])
        hor = np.array([[1,1,1,1,1,1]])
        
    else:
        ver = np.array([[1],
                [1],
                [1],
                [1],
                [1],
                [1],
                [1],
                [1],
                [1],
                [1],
                [1],
                [1],
                [1],
                [1],
                [1],
                [1],
                [1],
                [1],
                [1]])
        hor = np.array([[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]])

    # to detect vertical lines of table borders
    img_ver = cv2.erode(img_thresholded_inverted, ver, iterations=5)
    ver_lines_img = cv2.dilate(img_ver, ver, iterations=5)
    cv2.resize(ver_lines_img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    cv2.imwrite('output/steps/3.png',ver_lines_img)

    # to detect horizontal lines of table borders
    img_hor = cv2.erode(img_thresholded_inverted, hor, iterations=5)
    hor_lines_img = cv2.dilate(img_hor, hor, iterations=5)
    cv2.resize(hor_lines_img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    cv2.imwrite('output/steps/4.png',hor_lines_img)


    # adding horizontal and vertical lines
    
    ## old adding code
    # hor_ver = cv2.add(hor_lines_img, ver_lines_img)
    # cv2.resize(hor_ver, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    # cv2.imwrite('output/steps/5.png',hor_ver)
    # hor_ver = 255-hor_ver
    # cv2.resize(hor_ver, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    # cv2.imwrite('output/steps/6.png',hor_ver)
    ## 

    # Weighting parameters, this will decide the quantity of an image to be added to make a new image.
    alpha = 0.5
    beta = 1.0 - alpha
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    # This function helps to add two image with specific weight parameter to get a third image as summation of two image.
    hor_ver = cv2.addWeighted(ver_lines_img, alpha, hor_lines_img, beta, 0.0)
    hor_ver = cv2.erode(~hor_ver, kernel, iterations=2)
    (thresh, hor_ver) = cv2.threshold(hor_ver, 128,255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    cv2.imwrite('output/steps/6.png',hor_ver)

    # subtracting table borders from image
    borders_subtracted = cv2.subtract(img_thresholded_inverted,hor_ver)
    cv2.resize(borders_subtracted, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    cv2.imwrite('output/steps/7.png',borders_subtracted)
    borders_subtracted = 255 - borders_subtracted 
    cv2.resize(borders_subtracted, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    cv2.imwrite('output/steps/8.png',borders_subtracted)


    #Doing xor operation for erasing table boundaries
    img_borders_removed_inverted = cv2.bitwise_xor(img_thresholded,borders_subtracted)
    cv2.resize(img_borders_removed_inverted, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    cv2.imwrite('output/steps/9.png',img_borders_removed_inverted)
    img_borders_removed = cv2.bitwise_not(img_borders_removed_inverted)
    cv2.resize(img_borders_removed, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    cv2.imwrite('output/steps/10.png',img_borders_removed)
    img_borders_removed_1=img_borders_removed.copy()
    cv2.resize(img_borders_removed_1, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

    #kernel initialization
    ver1 = np.array([[1,1],
                [1,1],
                [1,1],
                [1,1],
                [1,1],
                [1,1],
                [1,1],
                [1,1],
                [1,1]])
    hor1 = np.array([[1,1,1,1,1,1,1,1,1,1],
                [1,1,1,1,1,1,1,1,1,1]])

    #morphological operation
    temp1 = cv2.erode(img_borders_removed_1, ver1, iterations=1)
    ver_lines_img1 = cv2.dilate(temp1, ver1, iterations=1)
    cv2.resize(ver_lines_img1, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    cv2.imwrite('output/steps/11.png',ver_lines_img1)

    temp2 = cv2.erode(img_borders_removed_1, hor1, iterations=1)
    hor_lines_img2 = cv2.dilate(temp2, hor1, iterations=1)
    cv2.resize(hor_lines_img2, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    cv2.imwrite('output/steps/12.png',hor_lines_img2)

    # doing or operation for detecting only text part and removing rest all
    hor_ver = cv2.add(hor_lines_img2,ver_lines_img1)
    cv2.resize(hor_ver, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    cv2.imwrite('output/steps/13.png',hor_ver)

    dim1 = (hor_ver.shape[1],hor_ver.shape[0])
    dim = (hor_ver.shape[1]*2,hor_ver.shape[0]*2)

    # resizing image to its double size to increase the text size
    resized = cv2.resize(hor_ver, dim, interpolation = cv2.INTER_AREA)

    #bitwise not operation for fliping the pixel values so as to apply morphological operation such as dilation and erode
    want = cv2.bitwise_not(resized)
    cv2.resize(want, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    cv2.imwrite('output/steps/14.png',want)


    if(want.shape[0]<1000):
        kernel1 = np.array([[1,1,1]])
        kernel2 = np.array([[1],
                            [1],
                            [1]])
    else:
        kernel1 = np.array([[1,1,1,1,1,1]])
        kernel2 = np.array([[1],
                            [1],
                            [1],
                            [1],
                            [1],
                            [1]])

    img_borders_removed_1 = cv2.dilate(want,kernel1,iterations=27) # iterations need to be change depending on the picture
    img_borders_removed_1 = cv2.dilate(img_borders_removed_1,kernel2,iterations=10) # iterations need to be change depending on the picture
    cv2.resize(img_borders_removed_1, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
   
    # getting image back to its original size
    resized1 = cv2.resize(img_borders_removed_1, dim1, interpolation = cv2.INTER_AREA)
    cv2.imwrite('output/steps/15.png',resized1)

    # Find contours for image, which will detect all the boxes
    contours1, hierarchy1 = cv2.findContours(resized1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    #sorting contours by calling fuction
    (cnts, boundingBoxes) = sort_contours(contours1, method="top-to-bottom")

    #storing value of all bouding box height
    heightlist=[]
    for i in range(len(boundingBoxes)):
        heightlist.append(boundingBoxes[i][3])
    #sorting height values
    heightlist.sort()
    sportion = int(.5*len(heightlist))
    eportion = int(0.05*len(heightlist))
    #taking 50% to 95% values of heights and calculate their mean 
    #this will neglect small bounding box which are basically noise 
    try:
        medianheight = statistics.mean(heightlist[-sportion:-eportion])
    except:
        medianheight = statistics.mean(heightlist[-sportion:-2])
    #keeping bounding box which are having height more then 70% of the mean height and deleting all those value where ratio of width to height is less then 0.9
    box =[]
    img_temp = img_borders_removed.copy()
    for i in range(len(cnts)):    
        cnt = cnts[i]
        x,y,w,h = cv2.boundingRect(cnt)
        if(h>=.5*medianheight and w/h > 0.9):
            image = cv2.rectangle(img_temp,(x+4,y-2),(x+w-5,y+h),(0,255,0),1)
            box.append([x,y,w,h])
        # to show image
    cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    cv2.imwrite('output/steps/final.png',image)

    #rearranging all the bounding boxes horizontal wise where every box fall on same horizontal line 
    main=[]
    j=0
    l=[]
    for i in range(len(box)):    
        if(i==0):
            l.append(box[i])
            last=box[i]
        else:
            if(box[i][1]<=last[1]+medianheight/2):
                l.append(box[i])
                last=box[i]
                if(i==len(box)-1):
                    main.append(l)
            else:
                # print(l)            
                main.append(l)
                l=[]
                last = box[i]
                l.append(box[i])
    #calculating maximum number of box in a particular row
    maxsize=0
    for i in range(len(main)):
        l=len(main[i])
        if(maxsize<=l):
            maxsize=l
    ylist=[]
    for i in range(len(boundingBoxes)):
        ylist.append(boundingBoxes[i][0])
    ymax = max(ylist)
    ymin = min(ylist)
    ymaxwidth=0
    for i in range(len(boundingBoxes)):
        if(boundingBoxes[i][0]==ymax):
            ymaxwidth=boundingBoxes[i][2]
    TotWidth = ymax+ymaxwidth-ymin
    width = []
    widthsum=0
    for i in range(len(main)):
        for j in range(len(main[i])):
            widthsum = main[i][j][2]+widthsum
        
        # print(" Row ",i,"total width",widthsum)
        width.append(widthsum)
        widthsum=0
    #removing all the lines which are not the part of the table
    main1=[]
    flag=0
    for i in range(len(main)):
        if(i==0):
            if(width[i]>=(.8*TotWidth) and len(main[i])==1 or width[i]>=(.8*TotWidth) and width[i+1]>=(.8*TotWidth) or len(main[i])==1):
                flag = 1
        else:
            if(len(main[i])==1 and width[i-1]>=.8*TotWidth):
                flag=1
            
            elif(width[i]>=(.8*TotWidth) and len(main[i])==1):
                flag=1
                
            elif(len(main[i-1])==1 and len(main[i])==1 and (width[i]>=(.7*TotWidth) or width[i-1]>=(.8*TotWidth))):
                flag=1
        
            
        if(flag==1):
            pass
        else:
            main1.append(main[i])
        
        flag=0
    maxsize1=0
    for i in range(len(main1)):
        l=len(main1[i])
        if(maxsize1<=l):
            maxsize1=l
    #calculating the values of the mid points of the columns 
    midpoint=[]
    for i in range(len(main1)):
        if(len(main1[i])==maxsize1):
            # print(main1[i])
            for j in range(maxsize1):
                midpoint.append(int(main1[i][j][0]+main1[i][j][2]/2))
            break
    midpoint=np.array(midpoint)
    midpoint.sort()
    final = [[]*maxsize1]*len(main1)

    #sorting the boxes left to right
    for i in range(len(main1)):
        for j in range(len(main1[i])):
            min_idx = j        
            for k in range(j+1,len(main1[i])):
                if(main1[i][min_idx][0]>main1[i][k][0]):
                    min_idx = k
            
            main1[i][j], main1[i][min_idx] = main1[i][min_idx],main1[i][j]#sorting the boxes left to right
    for i in range(len(main1)):
        for j in range(len(main1[i])):
            min_idx = j        
            for k in range(j+1,len(main1[i])):
                if(main1[i][min_idx][0]>main1[i][k][0]):
                    min_idx = k
            
            main1[i][j], main1[i][min_idx] = main1[i][min_idx],main1[i][j]

    #storing the boxes in their respective columns based upon their distances from mid points  
    finallist = []
    for i in range(len(main1)):
        lis=[ [] for k in range(maxsize1)]
        for j in range(len(main1[i])):
            # diff=np.zeros[maxsize]
            diff = abs(midpoint-(main1[i][j][0]+main1[i][j][2]/4))
            minvalue = min(diff)
            ind = list(diff).index(minvalue)
            # print(minvalue)
            lis[ind].append(main1[i][j])
        # print('----------------------------------------------')
        finallist.append(lis)

    #extration of the text from the box using pytesseract and storing the values in their respective row and column
    todump=[]
    count=1
    for i in range(len(finallist)):
        for j in range(len(finallist[i])):
            to_out=''
            if(len(finallist[i][j])==0):
                print('-')
                todump.append(' ')
            
            else:
                for k in range(len(finallist[i][j])):                
                    y,x,w,h = finallist[i][j][k][0],finallist[i][j][k][1],finallist[i][j][k][2],finallist[i][j][k][3]

                    roi = img_borders_removed[x:x+h, y+2:y+w] #change which image the to be cropped here
                    roi1= cv2.copyMakeBorder(roi,5,5,5,5,cv2.BORDER_CONSTANT,value=[255,255])
                    img = cv2.resize(roi1, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
                    kernel = np.ones((2, 1), np.uint8)
                    img = cv2.dilate(img, kernel, iterations=1)
                    img = cv2.erode(img, kernel, iterations=2)
                    img = cv2.dilate(img, kernel, iterations=1)
                    # output cropped img
                    # cv2.imwrite('output/cropped/'+str(count)+'.png',img)
                    count+=1
                    out = pytesseract.image_to_string(img)
                    if(len(out)==0):
                        out = pytesseract.image_to_string(img,config='--psm 10 --oem 1')
                
                    to_out = to_out +" "+out
                    
                print(to_out)
                    
                todump.append(to_out)
                # # Debug stuff
                # cv2.imshow('image',img)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()
        print("--------------------------------------------------")

    #creating numpy array
    npdump = np.array(todump)

    #creating dataframe of the array 
    dataframe = pd.DataFrame(npdump.reshape(len(main1),maxsize1))
    data = dataframe.style.set_properties(**{'text-align': 'left'})

    #storing value in excel and csv format
    data.to_excel("output/output.xlsx", index = False, header= False)

main(source)