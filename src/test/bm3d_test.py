import time
import numpy as np
import cv2 as cv
from bm3d import bm3d


sigma = 25


def add_noise(image):
    noise = np.random.normal(0, sigma, image.shape)
    noisy_image = image + noise
    return noisy_image


if __name__ == '__main__':
    gt = cv.cvtColor(cv.imread("/home/nju-student/mkh/datasets/LLIE/LOLv1/eval15/high/1.png"), cv.COLOR_BGR2RGB)
    noisy_img = add_noise(gt).astype(np.uint8)

    start_time = time.time()
    final_img = bm3d(noisy_img, sigma_psd=sigma)
    print("Time: ", time.time() - start_time)

    final_img = np.clip(final_img, 0, 255).astype(np.uint8)
    result = np.hstack((gt, noisy_img, final_img))
    cv.imwrite("./bm3d.png", cv.cvtColor(result, cv.COLOR_RGB2BGR))

