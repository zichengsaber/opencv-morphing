import cv2 as cv 
import numpy as np
import os  

if __name__ =="__main__":
    I=cv.imread("frames/0.png")
    H,W,_=I.shape
    frames_path="./frames"
    frame_names=sorted(os.listdir(frames_path),key=lambda x:int(x.split(".")[0]))
    fourcc = cv.VideoWriter_fourcc(*'XVID')
    fps=2 # 
    print(frame_names)
    video = cv.VideoWriter("./videos/output.avi", cv.VideoWriter_fourcc('I', '4', '2', '0'), fps,(W,H))
    for idx in range(len(frame_names)):
        img_file=os.path.join(frames_path,frame_names[idx])
        img=cv.imread(img_file)
        video.write(img)
    