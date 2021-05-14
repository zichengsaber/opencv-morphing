import numpy as np
import cv2 as cv
import dlib 
import imutils
from scipy.spatial import Delaunay
import argparse

def shape_to_numpy(shape,dtype=np.int):
    coords=np.zeros((68,2),dtype=dtype)
    for i in range(68):
        coords[i,0]=shape.part(i).x
        coords[i,1]=shape.part(i).y 
    return coords
# 可视化获取特征提取点
def visualize(img,name,img_points,tr):
    for (x,y) in img_points:
        cv.circle(img,(x,y),2,(0,255,0),-1)
    
    for (a,b,c) in tr:
        pts=np.array([img_points[a],img_points[b],img_points[c]])
        cv.polylines(img,[pts],True,(0,0,255),1)

    cv.imwrite(name,img)


# 在68个点的基础上添加边界点
def generate_point(img,pre):
    H,W,_=img.shape
    detector=dlib.get_frontal_face_detector()
    landmark_predictor=dlib.shape_predictor(pre) 
    # 检测人脸
    rects=detector(img,1)
    for (i,rect) in enumerate(rects):
        shape=landmark_predictor(img,rect)
        shape=shape_to_numpy(shape)
        edges=np.array([
            [0,0],[W-1,H-1],[W-1,0],[0,H-1],
            [W-1,H//4],[0,H//4],[W//3,0],[W//3,H-1],
            [W-1,H//2],[0,H//2],[2*W//3,0],[2*W//3,H-1],
            [W-1,3*H//4],[0,3*H//4],
        ],dtype=np.int)
        shape=np.vstack([edges,shape])

        return shape


# 对三角区域进行affine
def affineTriangle(img1,img2,tr1,tr2):
    # 计算包裹该三角形的最小长方形
    r1=cv.boundingRect(tr1)
    r2=cv.boundingRect(tr2)
    # 我们要将affine应用与此处方块
    # 需要更改三角形的坐标
    tri1Cropped = []
    tri2Cropped = []
    for i in range(3):
        tri1Cropped.append([tr1[i][0]-r1[0],tr1[i][1]-r1[1]])
        tri2Cropped.append([tr2[i][0]-r2[0],tr2[i][1]-r2[1]])

    img1Cropped =img1[r1[1]:r1[1]+r1[3],r1[0]:r1[0]+r1[2]]
    # 获得Affine Matrix
    AffineMat=cv.getAffineTransform(np.float32(tri1Cropped),np.float32(tri2Cropped))
    # Apply the Affine Matrix
    img2Cropped=cv.warpAffine(img1Cropped,AffineMat,(r2[2], r2[3]),None, flags=cv.INTER_LINEAR, borderMode=cv.BORDER_REFLECT_101)

    # Get mask 过滤出三角形
    mask=np.zeros((r2[3],r2[2],3),dtype=np.float32)
    cv.fillConvexPoly(mask, np.int32(tri2Cropped), (1.0, 1.0, 1.0), 16, 0);

    img2Cropped = img2Cropped * mask

    img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] = img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] * ( (1.0, 1.0, 1.0) - mask )
    
    img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] = img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] + img2Cropped

def morphimg(img_girl,img_boy,step,factor):
    img1_points=generate_point(img_girl,args.pretrained)
    img2_points=generate_point(img_boy,args.pretrained)
    img3_points=np.round(factor*img1_points+(1-factor)*img2_points) # 获取C的顶点坐标 平均图像
    img3_points=img3_points.astype(np.int)
    # step1 计算三角形我们仅仅生成一种剖分
    tr=Delaunay(img2_points).simplices 
    # 可视化剖分结果
    """
    visualize(img_girl,"test/lmg.png",img1_points,tr)
    visualize(img_boy,"test/zzc.png",img2_points,tr)
    """
    # 主要的问题在于数目是否匹配
    # step2 计算仿射变换
    # 1.获取三角形矩阵
    tr1_set=[] 
    for i,j,k in tr:
        a,b,c=img1_points[i],img1_points[j],img1_points[k]
        tr1_set.append(np.array([a,b,c]))
    tr1_set=np.array(tr1_set)
    
    tr2_set=[]
    for i,j,k in tr:
        a,b,c=img2_points[i],img2_points[j],img2_points[k]
        tr2_set.append(np.array([a,b,c]))
    tr2_set=np.array(tr2_set)

    tr3_set=[]
    for i,j,k in tr:
        a,b,c=img3_points[i],img3_points[j],img3_points[k]
        tr3_set.append(np.array([a,b,c]))
    tr3_set=np.array(tr3_set)
    # 2. 计算变换矩阵
    # 生成平均图像 
    img1_mean=np.zeros(img_girl.shape,dtype=np.uint8) 
    for i in range(len(tr1_set)):
        affineTriangle(img_girl,img1_mean,tr1_set[i],tr3_set[i]) 
    
    img2_mean=np.zeros(img_boy.shape,dtype=np.uint8)
    for i in range(len(tr2_set)):
        affineTriangle(img_boy,img2_mean,tr2_set[i],tr3_set[i])
    
    # cv.imwrite("lmg_mean.png",img1_mean)
    # cv.imwrite("zzc_mean.png",img2_mean)
    new_img =np.uint8(np.round(factor*img1_mean+(1-factor)*img2_mean))

    cv.imwrite("./frames/{}.png".format(step),new_img)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-p","--pretrained",required=True,
        help="Path to facial landmark predictor")
    args=parser.parse_args()


    # 读入图片
    img_girl=cv.imread("./img/lmg.jpg")
    img_boy=cv.imread("./img/zzc.jpg")
    
    # 图像放缩
    img_girl=imutils.resize(img_girl,width=720)
    img_boy=imutils.resize(img_boy,width=720)

    for i in range(20):
        morphimg(img_girl,img_boy,i,i/20)




    
    
    

    
    
    