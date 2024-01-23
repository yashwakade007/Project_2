import cv2 
import numpy as np 

video=cv2.VideoCapture('video.mp4')

count_line_position=550
#initializing algorithm
min_width_reactangle=80
min_height_reactangle=80
offset=6
counter=0

algo=cv2.createBackgroundSubtractorMOG2()

def center_handle(x,y,w,h):
    x1=int(w/2)
    y1=int(h/2)
    cx=(x+x1)
    cy=(y+y1)
    return cx,cy
detect=[]


while True:
    ret,frame1=video.read()
    grey=cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
    blur=cv2.GaussianBlur(grey,(3,3),5)
#Applying algorithm on each frame
    image_sub=algo.apply(blur)
    dilat=cv2.dilate(image_sub,np.ones((5,5)))
    kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    dilatada=cv2.morphologyEx(dilat,cv2.MORPH_CLOSE,kernel)
    dilatada=cv2.morphologyEx(dilatada,cv2.MORPH_CLOSE,kernel)
    countershape,h=cv2.findContours(dilatada,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
   
    cv2.line(frame1,(25,count_line_position),(1200,count_line_position),(255,127,111),3)

    for (i,c) in enumerate(countershape):
        (x,y,w,h)=cv2.boundingRect(c)    
        validate_counter=(w>=min_width_reactangle)and(h>=min_height_reactangle)
        if not validate_counter:
            continue
        cv2.rectangle(frame1,(x,y),(x+w,y+h),(255,122,0),2)
        center=center_handle(x,y,w,h)
        detect.append(center)
        cv2.circle(frame1,center,5,(0,0,255),-1)

        for (x,y) in detect:
            if y<(count_line_position+offset) and y>(count_line_position-offset):
                counter+=1
                cv2.line(frame1,(25,count_line_position),(1200,count_line_position),(0,0,155),3)
                detect.remove((x,y))
                print("VEHICLE COUNTER:"+str(counter))
    cv2.putText(frame1,"VEHICLE COUNTER"+str(counter),(450,70),cv2.FONT_HERSHEY_SIMPLEX,2,(0,0,255),5)

    # cv2.imshow("Detector",dilatada)

    cv2.imshow('Original',frame1)

    if cv2.waitKey(120) & 0xff==ord('q'):
        break

cv2.destroyAllWindows()
video.release()