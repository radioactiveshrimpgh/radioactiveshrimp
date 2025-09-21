import secrets
import numpy as np
# import matplotlib.pyplot as plt
import math as m

def uniform(a:float = 0.0, b:float = 1.0)->float:
    """ 
    Generate cryptographically Secure uniform sample

    Args:
        a (float): lower bound value (default 0.0)
        b (float): upper bound value (default 1.0)

    Returns:
        float representing uniform sample
    
    Example:
        >>> uniform()
        0.3546927444895387
        >>> uniform()
        0.21098189931876055
    """
    # 53 random bits gives 53-bit precision double
    u = secrets.randbits(53)/(1<<53) # in [0,1)
    return a+(b-a)*u

def exponentialdist(lamb):
    """ 
    Generate exponentially distributed sample

    Args:
        lamb (float or int >0): lambda value of exponential distrobution

    Returns:
        x: float representing the exponentially distributed sample

    Raises:
        ValueError: if lambda is not greater than 0
    
    Example:
        >>> uexponentialdist(1)
        0.6839328417240588
        >>> uexponentialdist(1)
        2.901353101723147
        >>> exponentialdist(.5)
        2.451134119865936
    """
    if lamb<=0:
        raise ValueError('Lambda must be greater than 0')
    
    y = uniform()
    x = -(1/lamb)*np.log(y)
    return x


def poissondist(lamb):
    """ 
    Generate poisson distributed sample

    Args:
        lamb (int or float > 0): lambda value of poisson distribution

    Returns:
        i: int representing poisson distributed sample
    
    Example:
        >>> poissondist(1)
        1
        >>> poissondist(1)
        3
        >>> poissondist(10)
        9
    """
    if lamb<=0:
        raise ValueError('Lambda must be greater than 0')
    
    elamb = np.exp(-lamb)
    i = 0
    y = uniform()
    prob = elamb
    while y>prob:
        i = i+1
        factorial = m.factorial(i)
        power = lamb**i
        # print(y,i,elamb, power, factorial)
        prob = prob + (power/factorial)*elamb

    return i


# # draw histogram of 100,000 randomly distributed samples using exponential distribution
# samplexs = []
# for _ in range(100000):
#     samplexs.append(exponentialdist(1))

# plt.figure()
# hist = plt.hist(samplexs, bins='auto', orientation='horizontal', histtype='bar')
# plt.xlabel('count')
# plt.ylabel('X')
# plt.show()

# # draw histogram of 100,000 randomly distributed samples using exponential distribution
# samplexs = []
# for _ in range(100000):
#     samplexs.append(poissondist(10))

# plt.figure()
# hist = plt.hist(samplexs, bins='auto', histtype='bar')
# plt.xlabel('X')
# plt.ylabel('count')
# plt.show()