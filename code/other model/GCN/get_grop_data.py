from PIL import Image as I
import numpy as np
import random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import glob,os
from tqdm import tqdm

def give_num(num=30,l=5,m=6):
    a=np.array([x for x in range(num)])#构建列表
    a=a.reshape([l,m])#改变形状（7*4）
    a=np.transpose(a,[1,0])#转置（4*7）
    a=a.reshape([-1,])#改变形状（1*28）
    return a
    
def Generator_matrix(size=30,l=5,m=6):
    num_list=give_num(size,l,m)
    M=np.zeros((size,size))#构建28*28的全零矩阵
    for i in range(len(M)):
        num=num_list[i]
        M[i][num]=1#把28*28的全零矩阵上每一行的num列表列附1
    return M

def read_image(path,size=30):
    img_1=I.open(path).convert("L").resize([size,size])
    img_1=np.array(img_1)
    return img_1

#def show(ret):
#    plt.imshow(ret)
#    plt.show()
#    plt.close()
    
def do_one(ret,A):
    ret=np.dot(A,ret)
    ret=np.dot(A,np.transpose(ret,[1,0]))
    return ret

#def do_one_reverse(ret,T):
#    T_=np.linalg.inv(T)   #矩阵求逆
#    ret=np.transpose(np.dot(T_,ret),[1,0])
#    ret=np.dot(T_,ret)
#    return ret
def save(ar,path_):
    ar=ar.astype("uint8")
    #print(ar)
    img=I.fromarray(ar)
    img.save(path_)

def makepath(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path

def pos_matrix(size,B):
    edge_matrix = np.zeros((size*size,size*size))
    b=np.array([x for x in range(size*size)])
    b_raw=b.reshape([size,size])
    
    b_ret=np.dot(B,b_raw)
    b_ret=np.dot(B,np.transpose(b_raw,[1,0]))
    lst_1 = []
    for i,j in zip(b_raw,b_ret):
        for ii,jj in zip(i,j):
            edge_matrix[ii][int(jj)]=1
            lst_1.append([ii,int(jj)])
    return edge_matrix,lst_1

import numpy as np

def build_graph():
    #feature_matrix = image.flatten().reshape(-1, 1)  # 784x1

    adj_matrix = np.zeros((784, 784), dtype=int)
    rows, cols = image.shape

    for row in range(rows):
        for col in range(cols):
            index = row * cols + col
            if row > 0:    # 
                adj_matrix[index, index - 28] = 1
            if row < rows - 1:  # 
                adj_matrix[index, index + 28] = 1
            if col > 0:    # 
                adj_matrix[index, index - 1] = 1
            if col < cols - 1:  # 
                adj_matrix[index, index + 1] = 1

    return adj_matrix

if __name__=="__main__":
    size=28
    l=7
    m=4
    split="/"
    T=Generator_matrix(size,l,m)#把28*28的全零矩阵上每一行的num列表列附1
    edge_matrix_size,edge_matrix_2=pos_matrix(size,T)
    #adj_matrix = build_graph()
    
    #raw_path="minist"
    raw_path = glob.glob('../2_model/data/minist_raw/*')
    for i in tqdm(raw_path):
        #to_path="mnist7*7/"
        file_lst=glob.glob(i+split+"*")
        data_name = i.split('/')[-1]
        lst_feature = []
        lst_label = []
        for path_ in tqdm(file_lst):
            #to_path = makepath('../2_model/mnist7*7/%s/'%(str(path_.split('/')[1])))
            name = path_.split('/')[-1]
            name = path_.split(name)[0]
            label = path_.split('/')[-1].split('_')[0]
            
            raw=read_image(path_,size=size)#28.28
            change=do_one(raw,T)#28.28
            
            feature=raw.flatten().reshape(-1, 1)
            
            lst_feature.append(feature)
            lst_label.append(int(label))
            #to_save=path_.replace(name,to_path)
            #save(change,to_save)
        lst_edge = np.array([edge_matrix_2])
        lst_edge = np.transpose(lst_edge,[0,2,1])
        lst_feature = np.array(lst_feature)
        print(lst_feature.shape)
        lst_label = np.array(lst_label)
        #np.save('data/%s_edge.npy'%(data_name), lst_edge)
        np.save('data_4/%s_feature.npy'%(data_name), lst_feature)
        np.save('data_4/%s_label.npy'%(data_name), lst_label)
        #data = np.load('data.npy')