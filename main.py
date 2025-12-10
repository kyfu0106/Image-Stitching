import cv2
import numpy as np
import random
import math
import sys

# read the image file & output the color & gray image
def read_img(path):
    # opencv read image in BGR color space
    img = cv2.imread(path)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img, img_gray

# the dtype of img must be "uint8" to avoid the error of SIFT detector
def img_to_gray(img):
    if img.dtype != "uint8":
        print("The input image dtype is not uint8 , image type is : ",img.dtype)
        return
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img_gray

# create a window to show the image
# It will show all the windows after you call im_show()
# Remember to call im_show() in the end of main
def creat_im_window(window_name,img):
    cv2.imshow(window_name,img)

# show the all window you call before im_show()
# and press any key to close all windows
def im_show():
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def ratioTest_2knn(kp_list, des_list):
    good_matches = []
    for i in range(len(des_list)-1):
        matches = []
        for x in range(len(des_list[i])):
            Inf = float('inf')
            min1, min2 = Inf, Inf 
            min1_kp, min2_kp = None, None
            for y in range(len(des_list[i+1])):
                dist = np.linalg.norm(des_list[i][x] - des_list[i+1][y])
                if dist < min1:
                    min2 = min1
                    min2_kp = min1_kp
                    min1 = dist
                    min1_kp = kp_list[i+1][y]
                elif min1 < dist < min2:
                    min2 = dist
                    min2_kp = kp_list[i+1][y]
            matches.append([(min1_kp, min1), (min2_kp, min2)])

        # apply ratio test
        for n in range(len(matches)):
            d1 = matches[n][0][1]
            d2 = matches[n][1][1]
            if d1 < 0.75 * d2:
                good_matches.append([kp_list[i][n], matches[n][0][0]])
    #print('knn is done.') 
    return good_matches

def homography(good_matches):
    samples = random.sample(good_matches, 4)
    A = np.zeros((8,9))
    for i in range(4):
        A[i*2,0] = samples[i][0].pt[0]
        A[i*2,1] = samples[i][0].pt[1]
        A[i*2,2] = 1
        A[i*2,6] = -1 * samples[i][0].pt[0] * samples[i][1].pt[0]
        A[i*2,7] = -1 * samples[i][0].pt[1] * samples[i][1].pt[0]
        A[i*2,8] = -1 * samples[i][1].pt[0] 
    for i in range(4):
        A[i*2+1,3] = samples[i][0].pt[0]
        A[i*2+1,4] = samples[i][0].pt[1]
        A[i*2+1,5] = 1
        A[i*2+1,6] = -1 * samples[i][0].pt[0] * samples[i][1].pt[1]
        A[i*2+1,7] = -1 * samples[i][0].pt[1] * samples[i][1].pt[1]
        A[i*2+1,8] = -1 * samples[i][1].pt[1]

    U, S, VT = np.linalg.svd(A)
    H = np.reshape(VT[-1], (3, 3))
    H = (1 / H.item(8)) * H
    return H

def RANSAC(good_matches):
    maxS = []
    bestH = None
    for _ in range(1500):
        H = homography(good_matches)
        S = []
        for pairs in good_matches:
            p1 = pairs[0]
            p2 = pairs[1]
            p2_estimate = H @ np.array([[p1.pt[0]], [p1.pt[1]], [1]])
            p2_estimate = (1 / p2_estimate[2]) * p2_estimate
  
            dist = np.linalg.norm(np.array((p2.pt[0], p2.pt[1])) - np.array((p2_estimate[0][0], p2_estimate[1][0])))
            if dist < 5:
                S.append(pairs)
        if len(S) > len(maxS):
            maxS = S
            bestH = H

    #print('RANSAC is done.')         
    return bestH, maxS

def image_stitching(img1, img2, H):
    h1,w1 = img1.shape[:2]
    h2,w2 = img2.shape[:2]

    pts1 = np.float32([[0,0],[0,h1],[w1,h1],[w1,0]]).reshape(-1,1,2)
    pts2 = np.float32([[0,0],[0,h2],[w2,h2],[w2,0]]).reshape(-1,1,2)
    pts1_ = cv2.perspectiveTransform(pts1, H)
    pts = np.concatenate((pts1_, pts2), axis=0)

    [xmin, ymin] = np.int32(pts.min(axis=0).ravel() - 0.5)
    [xmax, ymax] = np.int32(pts.max(axis=0).ravel() + 0.5)

    t = [-xmin,-ymin]
    Ht = np.array([[1,0,t[0]],[0,1,t[1]],[0,0,1]]) # translate
    A = Ht @ H

    size = (xmax-xmin, ymax-ymin)
    result = cv2.warpPerspective(src=img1, M=A, dsize=size)
    result[t[1]:h2+t[1],t[0]:w2+t[0]] = img2

    return result
    

if __name__ == '__main__':
    # the example of image window
    # creat_im_window("Result",img)
    # im_show()

    # you can use this function to store the result
    # cv2.imwrite("result.jpg",img)
    image_pairs =  [['./test/m1.jpg', './test/m2.jpg', './test/m3.jpg', './test/m4.jpg'],
                    ['./test/m2.jpg', './test/m3.jpg', './test/m4.jpg', './test/m5.jpg'],
                    ['./test/m3.jpg', './test/m4.jpg', './test/m5.jpg', './test/m6.jpg'],
                    ['./test/m4.jpg', './test/m5.jpg', './test/m6.jpg', './test/m7.jpg'],
                    ['./test/m5.jpg', './test/m6.jpg', './test/m7.jpg', './test/m8.jpg'],
                    ['./test/m6.jpg', './test/m7.jpg', './test/m8.jpg', './test/m9.jpg'],
                    ['./test/m7.jpg', './test/m8.jpg', './test/m9.jpg', './test/m10.jpg']]
    
    index = 1
    images = []
    for image_pair in image_pairs:
        print("stitching m"+str(index)+" to m"+str(index+3)+'...')
        img1, img1_gray = read_img(image_pair[0])
        for i in range(1,len(image_pair)):
            kp_list = []
            des_list = []

            img2, img2_gray = read_img(image_pair[i])
            sift = cv2.SIFT_create()
            kp1, des1 = sift.detectAndCompute(img1_gray, None)
            kp2, des2 = sift.detectAndCompute(img2_gray, None)

            kp_list.append(kp1)
            kp_list.append(kp2)
            des_list.append(des1)
            des_list.append(des2)
           
            good_matches = ratioTest_2knn(kp_list, des_list)
            bestH, maxS = RANSAC(good_matches)
            result = image_stitching(img1, img2, bestH)
            img1, img1_gray = result, img_to_gray(result)

        
        #cv2.imwrite("result_m"+str(index)+"_m"+str(index+3)+".jpg",img1)
        print("stitching m"+str(index)+" to m"+str(index+3)+' is done.')
        creat_im_window("result_m"+str(index)+"_m"+str(index+3),img1)
        im_show()
        index += 1

        