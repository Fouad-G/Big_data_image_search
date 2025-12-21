import cv2
import numpy as np


def extract_hsv_histogram(
    image: np.ndarray,
    h_bins: int = 50,
    s_bins: int = 60,
) -> np.ndarray:
    
    # h hue , farbton(r,b,g)
    #s saturation , sättigung , (Blss vs kräftig
    #v value = helligkeit  , wir nehmen nur  hs weil helligkeit stört
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    hist = cv2.calcHist(
        [hsv],
        channels=[0, 1],# also h und s
        mask=None,
        histSize=[h_bins, s_bins],
        ranges=[0, 180, 0, 256],
        )
    
    # damit alle bilder gleich groß sind normaliseren wir die größe

    cv2.normalize(hist, hist)
    return hist.flatten()

#distanz zwischen histgramme berechnen
def chi_square_distance(hist_a:np.ndarray,hist_b:np.ndarray,eps:float=1e-10,)->float:
        diff =hist_a-hist_b
        sum=hist_a+hist_b+eps
        return 0.5* np.sum((diff*diff)/sum)

def color_similarity(image_a: np.ndarray,image_b: np.ndarray,) -> float:
  #ähnlichkeit zwischen 2 bilder
    hist_a = extract_hsv_histogram(image_a)
    hist_b = extract_hsv_histogram(image_b)
    return chi_square_distance(hist_a, hist_b)
