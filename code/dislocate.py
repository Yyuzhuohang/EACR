from PIL import Image as I
import numpy as np
import random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import glob,os
from tqdm import tqdm

def give_num(num=30,l=5,m=6):
    a=np.array([x for x in range(num)])
    a=a.reshape([l,m])
    a=np.transpose(a,[1,0])
    a=a.reshape([-1,])
    return a
    
def Generator_matrix(size=30,l=5,m=6):
    num_list=give_num(size,l,m)
    M=np.zeros((size,size))
    for i in range(len(M)):
        num=num_list[i]
        M[i][num]=1
    return M

def read_image(path,size=30):
    img_1=I.open(path).convert("L").resize([size,size])
    img_1=np.array(img_1)
    return img_1

def show(ret):
    plt.imshow(ret)
    plt.show()
    plt.close()
    
def do_one(ret,A):
    ret=np.dot(A,ret)
    ret=np.dot(A,np.transpose(ret,[1,0]))
    return ret
def do_one_reverse(ret,T):
    T_=np.linalg.inv(T) 
    ret=np.transpose(np.dot(T_,ret),[1,0])
    ret=np.dot(T_,ret)
    return ret
def save(ar,path_):
    ar=ar.astype("uint8")
    #print(ar)
    img=I.fromarray(ar)
    img.save(path_)

def makepath(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path

if __name__=="__main__":
    size=32
    l=8
    m=4
    split="/"
    T=Generator_matrix(size,l,m)
    raw_path = glob.glob('../data/CIFAR10/TRAIN/AIRPLANE')
    for i in tqdm(raw_path):
        file_lst=glob.glob(i+split+"*")
        for path_ in file_lst:
            to_path = makepath('../data/dis_result/%s/'%(str(path_.split('/')[4])))
            name = path_.split('/')[-1]
            name = path_.split(name)[0]
            raw=read_image(path_,size=size)
            change=do_one(raw,T)
            to_save=path_.replace(name,to_path)
            save(change,to_save)