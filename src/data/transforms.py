from functools import reduce
import numpy as np

class RandomFlip:
    def __init__(self, prob):
        self.prob = prob
    def __call__(self, img):
        if np.random.random() > self.prob:
            if np.random.random() > self.prob:
                return np.flip(img, 0)
            else:
                return(np.flip(img, 1))
        return img

class Gaussian:
    def __init__(self, prob, mean=0.0, var=0.1):
        self.prob = prob
        self.mean = mean
        self.var = var
    def __call__(self, img):
        if np.random.random() > self.prob:
            sigma = self.var**0.5
            gauss = np.random.normal(self.mean,sigma,(img.shape))
            gauss = gauss.reshape(img.shape)
            noisy = img + gauss
            return noisy
        return img

def composite_function(*func): 
      
    def compose(f, g): 
        return lambda x : f(g(x)) 
              
    return reduce(compose, func, lambda x : x) 