import PIL.Image as I
import numpy as np
def get_dis(loc,loc2):
    x,y=loc
    x2,y2=loc2
    dis=((x-x2)**2+(y-y2)**2)**0.5
    return dis

def generate_matrix(n):
    index2loc={}
    for i in range(n*n):
        x=int(i/n)
        y=i-x*n
        index2loc[i]=[x,y]

    new_matrix=np.zeros([n*n,n*n])
    for i,loc in index2loc.items():
        for j, loc2 in index2loc.items():
            dis=get_dis(loc,loc2)
            if dis<=4:
                new_matrix[i,j]=255
    return new_matrix
if __name__=="__main__":
    n=28
    matrix=generate_matrix(n)
    np.save('data_4/train_edge.npy', [matrix])
    np.save('data_4/test_edge.npy', [matrix])
    #image=I.fromarray(matrix)
    #image.show()
    #image.save("image.png")
    #print(generate_matrix(5))