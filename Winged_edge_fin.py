#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 12 14:51:39 2019

@author: williamzimmerman
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  9 11:18:25 2019

@author: williamzimmerman
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

https://www.scratchapixel.com/lessons/3d-basic-rendering/ray-tracing-rendering-a-triangle/barycentric-coordinates 
(ignore below parallelogram)
Triangle area using determinants 
https://people.richland.edu/james/lecture/m116/matrices/applications.html

Created on Tue Jul  9 12:35:18 2019

@author: williamzimmerman

1. Extract ORB features from RGB image
2. Get 2D Delaunay Triangulation (DT) of ORB features
3. Plug in 3D points (using depth map/image) instead of 2D points and 
make ply file with triangles in 3D.

4. Created "winged edge" representation of mesh. See 
https://www.cs.stevens.edu/~mordohai/classes/cs532_f15/cs532f15_Week12.pdf 
starting on slide 25. (This is also on google drive in case my page 
doesn't last for very long.)
The most important slides are 28 and 29 and then the following ones that 
show how to build the representation.

For some text see 
https://www.cs.umd.edu/class/spring2012/cmsc754/Lects/cmsc754-lects.pdf, 
lecture 22; or http://www.sccg.sk/~samuelcik/dgs/half_edge.pdf
If we can implement this data structure, it's easy to find neighboring 
triangles etc.

5. For each half-edge and its twin, retrieve the two triangles and check 
if they should be flipped. (If the half-edge has no twin, it's external 
and it cannot be flipped.) If a flip is performed, delete half edge and 
its twin and insert new diagonal and its twin. Vertices remain 
unchanged, but faces may have to be modified.

6. The test for flipping should be based on the depth of interior points 
of the triangles, but let's complete step 5 first. You can randomly flip 
for now.

====

I made an editable scratch folder inside 
https://drive.google.com/drive/u/1/folders/1sEQjHPUvL91-LwUx16f1_NUB26pUpKMr
to share files. You should have received invitations from google drive.

The data structure will not be easy. Email me as needed.

Attachments area

"""
from skimage import data
from skimage.color import rgb2gray
from skimage.feature import match_descriptors, ORB, plot_matches
from skimage.measure import ransac
from skimage.transform import FundamentalMatrixTransform
import argparse
import sys
import os
from PIL import Image
import math
import glob
from pyquaternion import Quaternion


import cv2
import matplotlib.pyplot as plt
#import openmesh as om
import numpy as np
import scipy
from scipy import interpolate

from scipy.spatial import Delaunay
#import meshlabxml as xml
import random
from scipy.spatial.transform import Rotation 

img = cv2.imread('/Users/williamzimmerman/Downloads/rgbd_dataset_freiburg1_xyz/rgb/1305031113.311452.png', 0)
img_path='/Users/williamzimmerman/Downloads/rgbd_dataset_freiburg1_xyz/rgb/1305031113.311452.png'
img_depth='/Users/williamzimmerman/Downloads/rgbd_dataset_freiburg1_xyz/depth/1305031125.386792.png'
img_pr= cv2.imread('/Users/williamzimmerman/Downloads/rgbd_dataset_freiburg1_xyz/rgb/1305031113.343252.png',0)
img_pr_path='/Users/williamzimmerman/Downloads/rgbd_dataset_freiburg1_xyz/rgb/1305031113.343252.png',0
img_pr_depth='/Users/williamzimmerman/Downloads/rgbd_dataset_freiburg1_xyz/depth/1305031111.639405.png'
img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(img)
plt.show()
width1, height1, c =img.shape

u1= width1/2
v1=height1/2
width2, height2 = img_pr.shape
u2=width2/2
v2=height2/2

imgcopy=img
num_points=25

orb = cv2.ORB_create(num_points)

kp = orb.detect(img, None)

kp,des = orb.compute(img, kp)
print(np.shape(kp))
img2 = cv2.drawKeypoints(img,kp,None,color=(0,255,0), flags=0)

plt.imshow(img2),plt.show()

size=np.shape(img)
print(size)
rect=(0,0,size[0],size[1])

subdiv=cv2.Subdiv2D(rect)
pts=[]

for i in range(0,num_points):

    x=kp[i].pt[0]
    y=kp[i].pt[1]
    subdiv.insert((y,x))
    pts.append((y,x))

def in_rect(r, point):
    if point[0] < rect[0] :
        return False
    elif point[1] < rect[1] :
        return False
    elif point[0] > rect[2] :
        return False
    elif point[1] > rect[3] :
        return False
    return True
    
depth_points=[]   
    
def delaunay_gen(img,subdiv,color,rect):
    del_triangles=subdiv.getTriangleList()
    for t in del_triangles:
        pt1 = (t[0], t[1])
        pt2 = (t[2], t[3])
        pt3 = (t[4], t[5])
       
        if in_rect(rect, pt1) and in_rect(rect, pt2) and in_rect(rect, pt3) :
         
            cv2.line(img, pt1, pt2, color, 1)
            cv2.line(img, pt2, pt3, color, 1)
            cv2.line(img, pt3, pt1, color, 1)

        

delaunay_gen(img, subdiv, (255,0,0),rect)

#cv2.imshow("image", img)

plt.imshow(img)
plt.show()

focalLength = 525.0
centerX = 319.5
centerY = 239.5
scalingFactor = 5000.0



pts=np.array(pts)
print(pts)
tri=Delaunay(pts)
triangles=tri.simplices
print(triangles)





def counter_clockwise(tris):
    global pts
    for i in range(len(tris)):
        tri=tris[i]
        print(tri)
        
        
        a=tri[0]
        b=tri[1]
        c=tri[2]
        at=a
        bt=b
        ct=c
        a=pts[a]
        b=pts[b]
        c=pts[c]
        a=(a[0],a[1],0)
        b=(b[0],b[1],0)
        c=(c[0],c[1],0)
        ab=(a,b)
        ac=(a,c)
        
        cross=np.cross(ab,ac)
        print(cross[1][2])
        if(cross[1][2]<0):
            tris[i]=[bt,at,ct]
            cross=np.cross(ab,ac)
            print("cross")
            print(tris[i])
    return tris



depth_pts=[]



for i in range(len(pts)):

    pts[i][0]=int(pts[i][0])
    pts[i][1]=int(pts[i][1])
    
pts_color=[] 
prime_3d=[]
pyr_pts=[]
pts_prime=[]
ptcld_pr=[]

'''
a_mat=[]
f_mat=0
a_mat=np.matrix([u1*u2 , v1*v2 , v1 , u1*v2 , v1*v2 , v2 , u1 , u2 ])
print(a_mat)
f_mat=scipy.linalg.null_space(a_mat[0])
#f_mat=np.reshape(f_mat,(3,3))
print(f_mat)
print(np.matmul(a_mat,f_mat))
print("ns")
stop

for i in range(len(f_mat)):
    text.write(str(f_mat[i]))
 '''
text=open("P-mat.txt","w")
text2=open("P-mat2.txt","w")


def generate_pointcloud(rgb_file,depth_file,ply_file,extrinsics):
    """
    Generate a colored point cloud in PLY format from a color and a depth image.
    
    Input:
    rgb_file -- filename of color image
    depth_file -- filename of depth image
    ply_file -- filename of ply file
    extrinsics -- extrinsic parameters of the camera
    
    """
    global pts
    global depth_pts
    rgb = Image.open(rgb_file)
    depth = Image.open(depth_file)

    if rgb.size != depth.size:
        raise Exception("Color and depth image do not have the same resolution.")
    if rgb.mode != "RGB":
        raise Exception("Color image is not in RGB format")
    if depth.mode != "I":
        raise Exception("Depth image is not in intensity format")
    print(extrinsics)
    # Compute camera motion matrix
    q = Quaternion(float(extrinsics[7]), float(extrinsics[4]), float(extrinsics[5]), float(extrinsics[6]))
    R = q.rotation_matrix #np.concatenate((np.matrix(q.rotation_matrix), axis=0))
    t = np.matrix([[float(extrinsics[1])], [float(extrinsics[2])], [float(extrinsics[3])]])
    
    q2 = Quaternion(float(extrinsics2[7]), float(extrinsics2[4]), float(extrinsics2[5]), float(extrinsics2[6]))
    R2 = q.rotation_matrix #np.concatenate((np.matrix(q.rotation_matrix), axis=0))
    t2 = np.matrix([[float(extrinsics2[1])], [float(extrinsics2[2])], [float(extrinsics2[3])]])
    #motion = np.concatenate((R,t), axis=1)
    #extrinsic_dot_product=np.dot(t, R)
    #points = []
   # file = open(ply_file,"w")
    
    p4=np.matmul(-R,t)
    
    P_mat0=np.concatenate((R,t),axis=1)
    print(P_mat0)
    P_mat0=np.concatenate((P_mat0,np.matrix([0,0,0,1])))
    ptxt=str(P_mat0)
    ptxt=ptxt.replace('[',' ')
    ptxt=ptxt.replace(']',' ')
    text.write(ptxt)
    
    p42=np.matmul(-R2,t2)
    
    P_mat02=np.concatenate((R2,t2),axis=1)
    print(R2)
    print("r")
    print(P_mat02)
    P_mat02=np.concatenate((P_mat02,np.matrix([0,0,0,1])))
    ptxt2=str(P_mat02)
    ptxt2=ptxt2.replace('[',' ')
    ptxt2=ptxt2.replace(']',' ')
    text2.write(ptxt2)
  
    
    k=np.matrix([focalLength, 0, centerX,0, focalLength, centerY,0,0,1])
    k=np.reshape(k,(3,3))
    textk=open("k-mat.txt","w")
    ktxt=str(k)
    ktxt=ktxt.replace('[',' ')
    ktxt=ktxt.replace(']',' ')
    textk.write(ktxt)
   
    text.close()
    text2.close()
    textk.close()
    
    
    for i in range(len(pts)):
        pt_ind=pts[i]
        print(pt_ind)
        print(np.shape(depth))
        Z = depth.getpixel((pt_ind[1],pt_ind[0])) / scalingFactor
        
        
 
        X = (pts[i][0] - centerX) * Z / focalLength
        Y = (pts[i][1] - centerY) * Z / focalLength
        print(focalLength)

        depth_pts.append([X,Y,Z])
        xy=[pts[i][0],pts[i][1]]
      

        x=int(xy[0])
        y=int(xy[1])
    
        pts_color.append([img[x,y]])
        re=np.reshape([X,Y,Z],(3,1))
        
        three_d_prime=np.matmul(R,re)#fix move r on the left of column vector
        #make XYZ a column
        three_d_prime=np.add(three_d_prime,t)
        prime_3d.append(three_d_prime)
        col=img[x,y]
        print(prime_3d)
        
        prime_3d2=prime_3d[0]
    
        #pyr pts has  indexes with sub index
        pts_prime=(prime_3d2)
        print(pts_prime)
        ptcld_pr.append([pts_prime[0],pts_prime[1],pts_prime[2]])
   
    
           # PVec = np.matmul(motion,np.array([X, Y, Z, 1]))
            # PVec = [[X, Y, Z, 1]]
           # P = np.asarray(PVec)[0]
            # points.append("%f %f %f %d %d %d 0\n"%(X,Y,Z,color[0],color[1],color[2]))
           # points.append("%f %f %f %d %d %d 0\n" % (P[0], P[1], P[2], color[0], color[1], color[2]))
            #file.write("%f %f %f %d %d %d\n" % (P[0], P[1], P[2], color[0], color[1], color[2]))
    #file.close()
    #writePLY(points)
    
 
#    file = open(ply_file,"w")
#    file.write('''ply
#        format ascii 1.0
#        element vertex %d
#        property float x
#        property float y
#        property float z
#        element face num_tri
#        property list uchar int vertex_index 
#        end_header
#        %s
#        '''%(len(points),"".join(points)))
#    
#    file.close()
    
    
    
    
    
    
    
    
    
def gen_plyfile(triangles,pts_func,filenameply,filenametxt):

    
    file = open(filenameply,"w")
    print('{}\n', format(triangles))
    file.write('''ply
        format ascii 1.0
        element vertex %d
        property float x
        property float y
        property float z
        property uchar red
        property uchar green 
        property uchar blue
        element face %d
        property list uchar int vertex_index 
        end_header
        '''%(len(pts_func),len(triangles)))
    for i in range(len(pts_func)):
        rgb=img[int(pts_func[i][0]),int(pts_func[i][1])]
        r=rgb[0]
        g=rgb[1]
        b=rgb[2]
        rgb_list=list([r,g,b])
        file.write('{0:.2f} {1:.2f} {2:.2f} {3} {4} {5}\n'.format(pts_func[i][0], pts_func[i][1],pts_func[i][2],r,g,b))
    for i in range(len(triangles)):
        file.write('3 {} {} {} \n'.format((triangles[i][0]),(triangles[i][1]), (triangles[i][2])))
        
    file.close()
    
    file = open(filenametxt,"w")
    print('{}\n', format(triangles))
    file.write('''ply
        format ascii 1.0
        element vertex %d
        property float x
        property float y
        property float z
        property uchar red 
        property uchar green
        property uchar blue
        element face %d
        property list uchar int vertex_index 
        end_header
        '''%(len(pts_func),len(triangles)))
    for i in range(len(pts_func)):
        rgb=img[int(pts_func[i][0]),int(pts_func[i][1])]
        r=rgb[0]
        g=rgb[1]
        b=rgb[2]
        rgb_list=list([r,g,b])
        file.write('{0:.2f} {1:.2f} {2:.2f} {3} {4} {5}\n'.format(pts_func[i][0], pts_func[i][1],pts_func[i][2],r,g,b))
        rgb_list.clear()
    for i in range(len(triangles)):
        file.write('3 {} {} {} \n'.format((triangles[i][0]), (triangles[i][1]), (triangles[i][2])))
    file.close()



plt.imshow(img)
plt.show()  
#










rgbPaths = glob.glob("/Users/williamzimmerman/Downloads/rgbd_dataset_freiburg1_xyz/rgb/*.png")
depthPaths = glob.glob("/Users/williamzimmerman/Downloads/rgbd_dataset_freiburg1_xyz/depth/*.png")
gtPath = "/Users/williamzimmerman/Downloads/rgbd_dataset_freiburg1_xyz/groundtruth.txt"

lines = [line.rstrip('\n') for line in open(gtPath)]

rgbPaths.sort()
depthPaths.sort()


    # parser.add_argument('rgb_file', help='input color image (format: png)')
    # parser.add_argument('depth_file', help='input depth image (format: png)')
    # parser.add_argument('ply_file', help='output PLY file (format: ply)')
    # args = parser.parse_args()

    #generate_pointcloud(args.rgb_file,args.depth_file,args.ply_file)

extrinsics = lines[3].split(" ")
extrinsics2= lines[4].split(" ")
     
        # PLY
generate_pointcloud(rgbPaths[0],depthPaths[0],"ply/out_" + str(0) + '.ply', extrinsics)
        # .txt
        #if(i < 10):
        #  prefix = "pointclouds/out_00"
        #elif(i < 100):
        #  prefix = "pointclouds/out_0"
        #generate_pointcloud(rgbPaths[i],depthPaths[i],prefix + str(i) + '.txt', extrinsics)

print(depth_pts)
print("depth")
gen_plyfile(triangles,depth_pts, "orig_file.ply", "orig_file.txt")

wing_vert_table=[]
winged_edge_table=[]
wing_face_table=[]

triangles=counter_clockwise(triangles)

def gen_datastructures(pts,triangles):
    edge_table=[]
    face_table=[]
    wing_vert_table.clear()
    winged_edge_table.clear()
    wing_face_table.clear()
    
    for i in range(len(pts)):
        print(pts[i])
        if(i!=len(pts)-1):
            edge=(pts[i],pts[i+1])
        else:
            edge=(pts[i], pts[0])
       
        edge_table.append(edge)
    
    '''
    edge_table[i]=(edge,prev,next_edge)
    face_table[i]=(leftface,rightface)
    '''
    
    
    #winged edge sarts here 
    
    
    edge_table=[]
    face_table=[]
    face_table_2=[]
    
    for i in range(len(triangles)):
        verticies=triangles[i]
        edge=[verticies[0],verticies[1]]
        next_edge=[verticies[1],verticies[2]]
        prev=[verticies[0], verticies[2]]
        data=[edge,next_edge,prev]
        edge_table.append(data)
    print(edge_table)
    face_table=[] 
    first_vert_table=[]
    second_vert_table=[]
    match_table=[]
    match_vert_table=[]
    match_table_2=[]
    match_table_3=[]
    trifaces=[]
    
    
    
    for i in range(len(edge_table)):
        
        edge=edge_table[i]#get 3 edges
        edge1=edge[0]
        vert1_1=edge1[0]#get all the verticies of the triangle
        vert1_2=edge1[1]
        edge2=edge[1]
        vert2_1=edge2[0]
        vert2_2=edge2[1]
        edge3=edge[2]
    
        vert3_1=edge3[0]
        vert3_2=edge3[1]
        verts=[vert1_1,vert1_2,vert2_1, vert2_2,vert3_1,vert3_2]#store in array
        
        wing_face_table.append((edge,edge1))
        
        
        wing_vert_table.append((vert1_1, edge1))
        wing_vert_table.append((vert2_1, edge2))
        wing_vert_table.append((vert3_2, edge3))
        
        depth=[]
        
        for i in range(len(triangles)):
            if((vert1_1 in triangles[i]) and (vert1_2 in triangles[i])):
                trifaces.append(triangles[i])
                
                if((vert2_2 in triangles[i])==False):
                  
                    secface=triangles[i]       
        depth.append(depth_pts[vert1_1][2])
        depth.append(depth_pts[vert1_2][2]) 
        
        if(len(trifaces)==2):
            for t in secface:
                if((t!=vert1_1) and (t!=vert1_2)):#if find third vert that isn't shared by 2 triangles
                    edgesec1=[t,vert1_1]#create other two edges
                    edgesec2=[t,vert1_2]
            
            winged_edge_table.append([edge1,trifaces[0],trifaces[1], edge2, edge3, edgesec1, edgesec2,depth])
            
            
            #winged_edge_table.append([edge1[::-1],trifaces[0],trifaces[1], edge3, edge2, edgesec2, edgesec1,depth[::-1]])
        elif(len(trifaces)==1):
            
             winged_edge_table.append([edge1,trifaces[0],edge2, edge3,depth])
             #winged_edge_table.append([edge1[::-1],trifaces[0],edge3, edge2,depth])
        trifaces=[]
        depth=[]
        
        
        
        for i in range(len(triangles)):
            if((vert2_1 in triangles[i]) and (vert2_2 in triangles[i])):
                trifaces.append(triangles[i])
                if((vert1_1 in triangles[i])==False):
                    secface=triangles[i]
                
        depth.append(depth_pts[vert2_1][2])
        depth.append(depth_pts[vert2_2][2])
        if(len(trifaces)==2):
           # secface=trifaces[1]
            for t in secface:
                if((t!=vert2_1) and (t!=vert2_2)):
                    edgesec1=[t,vert2_1]
                    edgesec2=[t,vert2_2]
            winged_edge_table.append([edge2,trifaces[0],trifaces[1], edge1, edge3, edgesec1, edgesec2,depth])
            
           
            
            #winged_edge_table.append([edge2[::-1],trifaces[0],trifaces[1], edge3, edge2, edgesec2, edgesec1,depth])

        elif(len(trifaces)==1):
             winged_edge_table.append([edge2,trifaces[0],edge1, edge3,depth])
             #winged_edge_table.append([edge2[::-1],trifaces[0],edge3, edge2,depth])
        trifaces=[]
        depth=[]
        
        
        depth.append(depth_pts[vert3_1][2])
        depth.append(depth_pts[vert3_2][2])
        for i in range(len(triangles)):
            if((vert3_1 in triangles[i]) and (vert3_2 in triangles[i])):
                trifaces.append(triangles[i])
                if((vert1_2 in triangles[i])==False):
                    secface=triangles[i]
                    
        if(len(trifaces)==2):
           # secface=trifaces[1]
            for t in secface:
                if((t!=vert3_1) and (t!=vert3_2)):
                    edgesec1=[t,vert3_1]
                    edgesec2=[t,vert3_2]
            winged_edge_table.append([edge3,trifaces[0],trifaces[1], edge1, edge2, edgesec1, edgesec2,depth])
            #winged_edge_table.append([edge3[::-1],trifaces[0],trifaces[1], edge1, edge2, edgesec1, edgesec2,depth])

        elif(len(trifaces)==1):
             winged_edge_table.append([edge3,trifaces[0],edge1, edge2,depth])
            # winged_edge_table.append([edge1[::-1],trifaces[0],edge3, edge2,depth])
        trifaces=[]
        depth=[]
        
    
  
    
    


gen_datastructures(pts,triangles)
print('x')


        

ply_vert=[]
ply_pts=[]
ply_ptsfunc=[]

def pts_4_ply(pts_ent):
    for i in range(len(wing_vert_table)):
     
        index=wing_vert_table[i]
        index=index[0]
        
        ply_pts.append(index)
        
    
    
    i=0
    pts_=[]
    while(i<len(pts_ent)):
        pts_.append((pts_ent[i],pts_ent[i+1]))
        i=i+2
    i=0
    while(i<len(ply_pts)):
        ply_ptsfunc.append((ply_pts[i],ply_pts[i+1],ply_pts[i+2]))
        
        i=i+3
    return pts_


#gen_plyfile(triangles,depth_pts,"test_fileply.ply","test_filetxt.txt")

#print(ply_vert)
ab=[]
ac=[]




split_var=random.randint(0, len(winged_edge_table))

def split_face(edgenum):

    data=winged_edge_table[edgenum]
    edge=data[0]
    
    rightface=data[1]
    
    leftface=data[2]
    for a in rightface:
        if((a in edge)!=True):
            rvert=a
    for a in  leftface:
        if((a in edge)!=True):
            lvert=a
    
    
    #vert_or=vert             
    pt_lvert=pts[lvert]
    pt_rvert=pts[rvert]
                 
    vert1=pts[edge[0]]
    vert2=pts[edge[1]]
    
    x_new_vert=vert1[0] + vert2[0]
    x_new_vert=x_new_vert/2
    y_new_vert=vert1[1]+vert2[1]
    y_new_vert=y_new_vert/2
    new_vertpts=((x_new_vert),(y_new_vert))
    
    print("len")
    

    pts_new=np.append(pts,new_vertpts)

    #new_vert=np.where(pts_new==new_vertpts)
    new_vert=num_points
    
    new_edge_lvert=(lvert,new_vert)
    
    new_edge_rvert=(rvert,new_vert)
    
    new_edge_midpt=(edge[0],new_vert)
    
    new_edge_midpt2=(edge[1],new_vert)
    
    runon_opp_edge=(lvert, edge[0])
    lunon_opp_edge=(lvert, edge[1])
    rdnon_opp_edge=(rvert, edge[0])
    ldnon_opp_edge=(rvert, edge[1])
    '''
    triangles.append((lvert,edge[0],new_vert))
    triangles.append((lvert,edge[1],new_vert))
    triangles.append((rvert,edge[0],new_vert))
    triangles.append((rvert,edge[1],new_vert))
    '''
    func_tri=[]
    for i in (range(len(triangles))):
        func_tri.append(triangles[i])

    func_tri.append((lvert,edge[0],new_vert))
    func_tri.append((lvert,edge[1],new_vert))
    func_tri.append((rvert,edge[0],new_vert))
    func_tri.append((rvert,edge[1],new_vert))
    
    '''
    triangles1=np.append(triangles,(lvert,edge[0],new_vert))
    triangles2=np.append(triangles1,(lvert,edge,[1],new_vert))
    triangles3=np.append(triangles2,(rvert,edge[0],new_vert))
    triangles4=np.append(triangles3,(rvert,edge[1],new_vert))
    '''
    
  
    

    return func_tri, pts_new


    



    
    
split_var=random.randint(0, len(winged_edge_table))

#trianglesnew, pts = split_face(split_var)   
triangles1=[]

i=0
'''
while(i<len(trianglesnew)-3):
    print(i)
    print(len(trianglesnew))
    triangles1.append((trianglesnew[i],trianglesnew[i+1],trianglesnew[i+2]))
    i+=1
'''  







#pts=pts_4_ply(pts)






#trianglesnew=counter_clockwise(trianglesnew) 
#gen_plyfile(trianglesnew,pts,"test_halffileply.ply","test_halffiletxt.txt")
print(winged_edge_table)
def flip_edge(edgenum):
        global pts
        global winged_edge_table
        global triangles
        
        data=winged_edge_table[edgenum]
        edge=data[0]
        print('edge:' + str(edge))
        vert=edge[0]
        vert2 = edge[1]
        rightface=data[1]
        if(len(data)!=8):
            return
        leftface=data[2]
        left_trave_prev=data[3]
        left_trav_succ=data[4]
        right_trave_prev=data[5]
        right_trav_succ=data[6]
        non_opp_vert_r=[]
        non_opp_vert_l=[]
    
        for num in rightface:
    
    
            if(vert == num or vert2 == num):
                non_opp_vert_r.append(num)
    
    
    
            else:
                opp_vert_r = num
        for num in leftface:
            if(vert == num or vert2 == num):
                non_opp_vert_l.append(num)
    
    
    
            else:
                opp_vert_l = num
        #Create other edge
        length = len(winged_edge_table)
        new_edge_1 = [opp_vert_l, opp_vert_r]
        new_edge_2 = [opp_vert_r, opp_vert_l]
        print('new'+str(new_edge_1))
        #delete other edges
        re_edges = [new_edge_1, new_edge_2]
        ind_del = []
        
        ed = data
        
        if(len(ed)!=8):
            return
        if edge in ed or [vert2, vert] in ed:
            if(ed[0]==[vert,vert2]):
                ed[0] = new_edge_1
                ed[1] = (opp_vert_l, opp_vert_r, vert)
                ed[2] = (opp_vert_l, opp_vert_r, vert2)
                ed[3] = [vert, opp_vert_l]
                ed[4] = [opp_vert_r, vert]
                ed[5] = [vert2, opp_vert_l]
                ed[6] = [opp_vert_r, vert2]
                winged_edge_table[edgenum]=ed
                
                
            elif(ed[0]==[vert2, vert]):
                ed[0] = new_edge_2
                ed[1] = (opp_vert_r, opp_vert_l, vert)
                ed[2] = (opp_vert_r, opp_vert_l, vert2)
                ed[3] = [vert, opp_vert_r]
                ed[4] = [opp_vert_l, vert]
                ed[5] = [vert2, opp_vert_r]
                ed[6] = [opp_vert_l, vert2]
                
                winged_edge_table[edgenum]=ed
        
        for i in range(length):
            
            if(len(winged_edge_table[i])==8):
                
                checkf1=ed[1]
                checkf2=ed[2]
                
                if(rightface[0]==checkf1[0] and rightface[1]==checkf1[1] and rightface[2]==checkf1[2]):
                    if vert in winged_edge_table[i][2]:
                        winged_edge_table[i][1]=ed[1]
                    else:
                        winged_edge_table[i][1]=ed[2]
                elif(rightface[0]==checkf2[0] and rightface[1]==checkf2[1] and rightface[2]==checkf2[2]):
                    if vert in winged_edge_table[i][1]:
                        winged_edge_table[i][2]=ed[1]
                    else:
                        winged_edge_table[i][2]=ed[2]
                    
                    
      
                if(leftface[0]==checkf1[0] and leftface[1]==checkf1[1] and leftface[2]==checkf1[2]):
                    if vert in winged_edge_table[i][2]:
                        winged_edge_table[i][1]=ed[1]
                    else:
                        winged_edge_table[i][1]=ed[2]
                elif(leftface[0]==checkf2[0] and leftface[1]==checkf2[1] and leftface[2]==checkf2[2]):
                    if vert in winged_edge_table[i][1]:
                        winged_edge_table[i][2]=ed[1]
                    else:
                        winged_edge_table[i][2]=ed[2]
        
        index=[]
        for i in range(len(triangles)):
             
            
             if((vert in triangles[i]) and (vert2 in triangles[i])):
                 index.append(i)
                 print(triangles[i])
        if(len(index)==2):
            triangles[index[0]]=(vert,opp_vert_l,opp_vert_r)
            triangles[index[1]]=(vert2,opp_vert_l,opp_vert_r)
        
            

def test_flip(rightface, leftface, vert, vert2):
        non_opp_vert_r=[]
        non_opp_vert_l=[]
    
        non_opp_vert_r=[]
        non_opp_vert_l=[]
        for num in rightface:
    
    
            if(vert == num or vert2 == num):
                non_opp_vert_r.append(num)
    
    
    
            else:
                opp_vert_r = num
        for num in leftface:
            if(vert == num or vert2 == num):
                non_opp_vert_l.append(num)
    
    
    
            else:
                opp_vert_l = num
                
        face1 = (opp_vert_l, opp_vert_r, vert)
        face2 = (opp_vert_l, opp_vert_r, vert2)
        
        trav_edge1=[vert, opp_vert_l]
        
        trav_edge2=[vert, opp_vert_r]
        
        trav_edge3=[vert2, opp_vert_l]
        
        trav_edge4=[vert2, opp_vert_r]
        

        
#        angle1=find_angle(trav_edge2,trav_edge1)
#        angle2=find_angle(trav_edge3,trav_edge2)
#        angle3=find_angle(trav_edge4,trav_edge3)
#        angle4=find_angle(trav_edge1,trav_edge4)
        
        return face1, face2#, angle1, angle2, angle3, angle4
            
            

def find_angle(A,B):
    global pts
    point1=pts[A[0]]
    point2=pts[A[1]]
    bpoint1=pts[B[0]]
    bpoint2=pts[B[1]]
    Avec=[point1[0]-point2[0],point1[1]-point2[1]]
    Bvec=[bpoint1[0]-bpoint2[0],bpoint1[1]-bpoint2[1]]  
    #a_length=math.sqrt((point1[0]-point2[0])**2+(point1[1]-point2[1])**2)
    a_length=np.linalg.norm(Avec)
    #b_length=math.sqrt((bpoint1[0]-bpoint2[0])**2+(bpoint1[1]-bpoint2[1])**2)
    b_length=np.linalg.norm(Bvec)    
    normmul=a_length*b_length    
    dot=np.dot(Avec,Bvec)   
    dot_over_norm=dot/normmul   #may need to be fixed

    if(abs(dot_over_norm)>1):
        dot_over_norm=int(dot_over_norm)
    print(dot_over_norm)
    theta=math.acos(dot_over_norm)
    theta=math.degrees(theta)
  
    return theta 
    
centroid_depth=[]
tweaked={}
def centroid(face):

    x = int((depth_pts[face[0]][0]+depth_pts[face[1]][0]+depth_pts[face[2]][0])/3)
    y = int((depth_pts[face[0]][1]+depth_pts[face[1]][1]+depth_pts[face[2]][1])/3)
    z = ((depth_pts[face[0]][2]+depth_pts[face[1]][2]+depth_pts[face[2]][2])/3)
    return (x,y,z)

for i in range(len(winged_edge_table)):
    edge=winged_edge_table[i][0]  

    if(len(winged_edge_table[i])==8):

        alt_face1, alt_face2 =test_flip(winged_edge_table[i][1],winged_edge_table[i][2],edge[0],edge[1])
        cent=centroid(winged_edge_table[i][1])
        alt_cent1=centroid(alt_face1)
        alt_cent2=centroid(alt_face2)
        cent2=centroid(winged_edge_table[i][2])
    
        depth_file=Image.open(depthPaths[0])
        depth_dif1=abs(cent[2]-((depth_file.getpixel((cent[1],cent[0])))/scalingFactor))
        depth_dif2=abs(cent2[2]-((depth_file.getpixel((cent2[1],cent2[0])))/scalingFactor))
        
        alt_depth_dif1=abs(alt_cent1[2]-((depth_file.getpixel((alt_cent1[1],alt_cent1[0])))/scalingFactor))
        alt_depth_dif2=abs(alt_cent2[2]-((depth_file.getpixel((alt_cent2[1],alt_cent2[0])))/scalingFactor))
        
        sum1=depth_dif1+depth_dif2
        
        sum2=alt_depth_dif1+alt_depth_dif2
        
        
        ang_sum1=find_angle((alt_face1[0],alt_face1[1]),(alt_face1[0],alt_face1[2])) + find_angle((alt_face1[0],alt_face1[2]),(alt_face1[2],alt_face1[1])) + find_angle((alt_face1[1],alt_face1[2]),(alt_face1[0],alt_face1[1]))
        ang_sum2=find_angle((alt_face2[0],alt_face2[1]),(alt_face2[0],alt_face2[2])) + find_angle((alt_face2[0],alt_face2[2]),(alt_face2[2],alt_face2[1])) + find_angle((alt_face2[1],alt_face2[2]),(alt_face2[0],alt_face2[1]))
        
        if(sum2<sum1):
           
            if(ang_sum1==180 and ang_sum2==180):
                
                if((tuple(winged_edge_table[i][0]) in tweaked)!=True):
                    print("flip edge: "+str(i))
                    flip_edge(i)
                    tweaked.update({tuple(winged_edge_table[i][3]):True})
                    tweaked.update({tuple(winged_edge_table[i][4]):True})
                    tweaked.update({tuple(winged_edge_table[i][5]):True})
                    tweaked.update({tuple(winged_edge_table[i][6]):True})
                    
def normaltest():
    for i in range(len(triangles)):
        temp_var=triangles[i]
        vertex_a=pts[temp_var[0]]
        vertex_b=pts[temp_var[1]]
        vertex_c=pts[temp_var[2]]
        normal=np.cross(np.matrix(vertex_a,vertex_b),np.cross(vertex_a,vertex_c))
        z=[0,0,1]
        dot=np.dot(normal,z)
        denom=1*abs(math.sqrt(normal[0]**2+normal[1]**2))
        theta=math.arccos(dot/denom)
        if(theta<5):
            triangles[i].clear()#get rid of this triangle
            print("true")
    return theta  

def aspect_ratio_test():
    for i in range(len(triangles)):
        temp_var=triangles[i]
        vertex_a=pts[temp_var[0]]
        vertex_b=pts[temp_var[1]]
        vertex_c=pts[temp_var[2]]
        numer=(vertex_a*vertex_b*vertex_c)
        s=(1/2)*(vertex_a+vertex_b+vertex_c)
        denom=8*(s-vertex_a)*(s-vertex_b)*(s-vertex_c)
        aspect_ratio=numer/denom
        if (aspect_ratio<.15 or aspect_ratio>1.5):
            triangles[i].clear()#get rid of this triangle
            print("true")
                            
                    
gen_plyfile(triangles,depth_pts,"totallynew_fileply.ply","totallynew_filetxt.txt")    
    

def make_pyr_ply(pyr_pts):
	fh = open('cam.ply', 'w')
	fh.write('ply\n')
	fh.write('format ascii 1.0\n')
	fh.write('comment Right-Handed System\n')
	fh.write('element vertex 6\n')
	fh.write('property float x\n')
	fh.write('property float y\n')
	fh.write('property float z\n')
	fh.write('property uchar red\n')
	fh.write('property uchar green\n')
	fh.write('property uchar blue\n')
	fh.write('element edge 10\n')
	fh.write('property int vertex1\n')
	fh.write('property int vertex2\n')
	fh.write('end_header\n')
	fh.write('{:.10f}'.format( pyr_pts[0][0][0] ) + ' ' + '{:.10f}'.format( pyr_pts[0][1][0] ) + ' ' + '{:.10f}'.format( pyr_pts[0][2][0] ) + ' 255 128 0\n')#fix to add colors
	fh.write('{:.10f}'.format( pyr_pts[1][0][0] ) + ' ' + '{:.10f}'.format( pyr_pts[1][1][0] ) + ' ' + '{:.10f}'.format( pyr_pts[1][2][0] ) + ' 255 128 0\n')
	fh.write('{:.10f}'.format( pyr_pts[2][0][0] ) + ' ' + '{:.10f}'.format( pyr_pts[2][1][0] ) + ' ' + '{:.10f}'.format( pyr_pts[2][2][0] ) + ' 255 128 0\n')
	fh.write('{:.10f}'.format( pyr_pts[3][0][0] ) + ' ' + '{:.10f}'.format( pyr_pts[3][1][0] ) + ' ' + '{:.10f}'.format( pyr_pts[3][2][0] ) + ' 255 128 0\n')
	fh.write('{:.10f}'.format( pyr_pts[4][0][0] ) + ' ' + '{:.10f}'.format( pyr_pts[4][1][0] ) + ' ' + '{:.10f}'.format( pyr_pts[4][2][0] ) + ' 255 128 0\n')
	fh.write('{:.10f}'.format( pyr_pts[5][0][0] ) + ' ' + '{:.10f}'.format( pyr_pts[5][1][0] ) + ' ' + '{:.10f}'.format( pyr_pts[5][2][0] ) + ' 255 128 0\n')
	fh.write('0 1\n')
	fh.write('0 2\n')
	fh.write('0 3\n')
	fh.write('0 4\n')
	fh.write('1 2\n')
	fh.write('2 3\n')
	fh.write('3 4\n')
	fh.write('4 1\n')
	fh.write('1 5\n')
	fh.write('5 2\n')
	fh.close()
	return

def make_unity_pyr_ply(pyr_pts):
	fh = open('cam_unity.ply', 'w')
	fh.write('ply\n')
	fh.write('format ascii 1.0\n')
	fh.write('comment Left-Handed System\n')
	fh.write('element vertex 6\n')
	fh.write('property float x\n')
	fh.write('property float y\n')
	fh.write('property float z\n')
	fh.write('property uchar red\n')
	fh.write('property uchar green\n')
	fh.write('property uchar blue\n')
	fh.write('element edge 10\n')
	fh.write('property int vertex1\n')
	fh.write('property int vertex2\n')
	fh.write('end_header\n')
	fh.write('{:.10f}'.format( pyr_pts[0][0][0] ) + ' ' + '{:.10f}'.format( pyr_pts[0][1][0] ) + ' ' + '{:.10f}'.format( pyr_pts[0][2][0] ) + ' 0 128 255\n')
	fh.write('{:.10f}'.format( pyr_pts[1][0][0] ) + ' ' + '{:.10f}'.format( pyr_pts[1][1][0] ) + ' ' + '{:.10f}'.format( pyr_pts[1][2][0] ) + ' 0 128 255\n')
	fh.write('{:.10f}'.format( pyr_pts[2][0][0] ) + ' ' + '{:.10f}'.format( pyr_pts[2][1][0] ) + ' ' + '{:.10f}'.format( pyr_pts[2][2][0] ) + ' 0 128 255\n')
	fh.write('{:.10f}'.format( pyr_pts[3][0][0] ) + ' ' + '{:.10f}'.format( pyr_pts[3][1][0] ) + ' ' + '{:.10f}'.format( pyr_pts[3][2][0] ) + ' 0 128 255\n')
	fh.write('{:.10f}'.format( pyr_pts[4][0][0] ) + ' ' + '{:.10f}'.format( pyr_pts[4][1][0] ) + ' ' + '{:.10f}'.format( pyr_pts[4][2][0] ) + ' 0 128 255\n')
	fh.write('{:.10f}'.format( pyr_pts[5][0][0] ) + ' ' + '{:.10f}'.format( pyr_pts[5][1][0] ) + ' ' + '{:.10f}'.format( pyr_pts[5][2][0] ) + ' 0 128 255\n')
	fh.write('0 1\n')
	fh.write('0 2\n')
	fh.write('0 3\n')
	fh.write('0 4\n')
	fh.write('1 2\n')
	fh.write('2 3\n')
	fh.write('3 4\n')
	fh.write('4 1\n')
	fh.write('1 5\n')
	fh.write('5 2\n')
	fh.close()
	return

#make_pyr_ply(pyr_pts)
    
    