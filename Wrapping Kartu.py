import cv2
import numpy as np


image = cv2.imread('kartumiring.jpg')

image2 = image.copy

        image = DrawLine(image, pp1, pp2)
        image = DrawLine(image, pp2, pp4)
        image = DrawLine(image, pp1, pp3)
        image = DrawLine(image, pp3, pp4)
        
    
        imo = TransformasiCitra(image2,pp1,pp2,pp3.pp4)
        lk.append(imo)



return lk

def DeteksiKartu(ims, L):
    for im in L :
        SAD = sum(abs(ims - im) )
        lSAD.append(SAD)
    lSAD = np.array (lSAD)
    return lSAD








L = EkstrakKartu(image)
for idx, im in enumerate(L):
    cv2.imshow(f'{idx}', im)