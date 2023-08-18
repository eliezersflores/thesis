import cv2
import numpy as np

def shad_atten(Iin_rgb, mask, saturate_opt, normalize_opt):

    """
    
    This function was implemented by Eliezer Soares Flores
    (https://scholar.google.com.br/citations?user=3jNjADcAAAAJ&hl=pt-BR&oi=ao).
    If you have any question about it, feel free to contact via email
    (eliezerflores@unipampa.edu.br or eliezersflores@gmail.com).
    
    This is a Python implementation of the "shading attenuation method",
    as described in the paper 
    "Segmentation of melanocytic skin lesions using feature learning
    and dictionaries" (ESWA).
    
    If you use this function, please cite the aforementioned paper.
    
    INPUTS 
        
        Iin_rgb: input image in the RGB colorspace. 
        
        mask: binary mask, where 1 indicates the positions to model
        the quadric function.
        
        saturate_opt: option to saturate the values in the output V
        channel (i.e., the third channel of the HSV colorspace) to 
        be in the range between 0 and 1.
            
        normalize_opt: option to normalize the average intensity of
        the output V channel to be the same than in the input V channel. 
            
    OUTPUTS

        Iout_rgb: output image, with the shading effects attenuated, 
        in the RGB colorspace. 
        
        Vin: input V channel.
        
        Vout: output V channel. 
            
        Z: adjusted quadric function. 

    USAGE EXAMPLE

        nr = im.shape[0]
        nc = im.shape[1]
    
        k = np.int(0.2*min(nr,nc))
        mask = np.zeros((nr, nc), dtype=int)
        mask[0:k, 0:k] = 1
        mask[0:k, nc-k:nc] = 1
        mask[nr-k:nr, 0:k] = 1
        mask[nr-k:nr, nc-k:nc] = 1
        
        imout, Vin, Vout, Z = shad_atten(im, mask, 1, 1)
    
    """
        
    nr = Iin_rgb.shape[0]
    nc = Iin_rgb.shape[1]
    
    Iin_hsv = cv2.cvtColor(Iin_rgb, cv2.COLOR_RGB2HSV)
    Hin = Iin_hsv[:,:,0]
    Sin = Iin_hsv[:,:,1]
    Vin = Iin_hsv[:,:,2]
    
    [r, c] = np.nonzero(mask == 1)
    vin = Vin[r, c]
    
    D = np.column_stack((r**2, c**2, r*c, r, c, np.ones((len(vin),)))) # design matrix
    p = np.dot(np.linalg.pinv(D), vin) # optimal parameters
    
    [R, C] = np.meshgrid(range(nr), range(nc), indexing='ij')
    
    Z = p[0]*R**2 + p[1]*C**2 + p[2]*R*C + p[3]*R + p[4]*C + p[5] + 1e-7
    
    Vout = Vin/Z
            
    if saturate_opt == 1:   
        Vout[Vout < 0] = 0
        Vout[Vout > 1] = 1
    
    if normalize_opt == 1:   
        Vout = np.uint8((Vin.mean()/Vout.mean())*Vout)
    else:
        Vout = np.uint8(255*Vout)    
        
    Iout_hsv = np.dstack((Hin, Sin, Vout))
    Iout_rgb = cv2.cvtColor(Iout_hsv, cv2.COLOR_HSV2RGB)
    
    Z[Z > 255] = 255
    Z = np.uint8(Z)    
    
    return Iout_rgb, Vin, Vout, Z