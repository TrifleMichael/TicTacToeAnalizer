from PIL import Image
import random as rd
import numpy as np


def generate_x(n):
    offset = max(1, n//5)
    x = [-1]*n*n
    for i in range(n):
        if offset <= i <= n-1-offset:
            x[i*n+i] = 1
            x[(n-1-i)*n+i] = 1
    return x


def generate_o(n):
    o = [-1]*n*n
    offset = max(1, n//5)
    for i in range(offset, n-offset):
        o[i*n+offset] = 1
        o[i*n+n-offset-1] = 1
        o[offset*n+i] = 1
        o[n*n-(offset+1)*n+i] = 1
    return o


def generate_blank(n):
    return [-1]*n*n


def brake_sign(m, k):
    for _ in range(k):
        x = rd.randint(0, n-1)*n+rd.randint(0, n-1)
        m[x] = -m[x]


def construct_board(n, temp=(('x','o','_'),('_','x','o'),('_','o','x'))):
    board = [-1]*(3*n+2)*(3*n+2)
    for i in range(3*n+2):
        board[(3*n+2)*i+n] = 1
        board[(3*n+2)*i+2*n+1] = 1
        board[(3*n+2)*n+i] = 1
        board[(3*n+2)*(2*n+1) + i] = 1
    for i in range(3):
        for j in range(3):
            if temp[i][j] == 'x':
                field = generate_x(n)
            elif temp[i][j] == 'o':
                field = generate_o(n)
            else:
                field = generate_blank(n)
            brake_sign(field, n//2)
            for x in range(n):
                for y in range(n):
                    board[(n*i+i+x)*(3*n+2)+n*j+j+y] = field[x*n+y]
    return board


def save_image(filename, image):
    flat_image = list(np.reshape(image, image.size))
    height, width = image.shape
    image_out = Image.new("RGB", (width, height))
    image_out.putdata(flat_image)
    image_out.save(filename)


n = 27
b = construct_board(n)

save_image("./x.png", np.array(list(map(lambda x: ((1+x)//2*255, (1+x)//2*255, (1+x)//2*255), generate_x(n))), dtype="i,i,i").astype(object).reshape((n, n)))
save_image("./o.png", np.array(list(map(lambda x: ((1+x)//2*255, (1+x)//2*255, (1+x)//2*255), generate_o(n))), dtype="i,i,i").astype(object).reshape((n, n)))
save_image("./blank.png", np.array(list(map(lambda x: ((1+x)//2*255, (1+x)//2*255, (1+x)//2*255), generate_blank(n))), dtype="i,i,i").astype(object).reshape((n, n)))
save_image("./board_test.png", np.array(list(map(lambda x: ((1+x)//2*255, (1+x)//2*255, (1+x)//2*255), b)), dtype="i,i,i").astype(object).reshape((3*n+2, 3*n+2)))