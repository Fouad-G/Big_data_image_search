import numpy as np
import cv2
from recommender.features.color import extract_hsv_histogram,chi_square_distance,color_similarity

def test_histogram_shape():
    img = np.zeros((100,100,3),dtype=np.uint8)
    hist= extract_hsv_histogram(img)
    assert hist.ndim== 1
    assert hist.sum()>0


#gleiche distanzen müssen 0 haben
def test_chi_square_zero_distance():
    hist =np.ones(100)
    dist=chi_square_distance(hist,hist)
    assert dist == 0.0
    

def test_color_similarty_identity():
    img= np.zeros((64,64,3),dtype=np.uint8)
    dist= color_similarity(img,img)
    assert dist ==0.0