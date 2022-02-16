import numpy as np


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))  # Here we manually define sigmoid function


def pre_process_img(img):
    """
    Here we define a function to simplify the image fed by Pong
    :param img: the raw image
    :return: the flatten array of the simplified image
    """
    img = img[35:185]
    img = img[::2, ::2, 0]
    img[img == 144] = 0
    img[img == 109] = 0
    img[img != 0] = 1
    return img.astype(np.float).ravel()


def discount_rewards(r, gamma):
    """
    Here we define a function to compute the discount rewards by the raw ones
    :param r: raw reward
    :param gamma: discount factor
    :return: discounted rewards
    """
    discounted_r = np.zeros_like(r)
    sum_up = 0

    for t in reversed(range(0, r.size)):
        if r[t] != 0:
            sum_up = 0
        sum_up = sum_up * gamma + r[t]
        discounted_r[t] = sum_up
    return discounted_r
