import numpy as np
import torch
import cv2

def ssim(img1, img2):
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())
    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def calculate_ssim_function(img1, img2):
    # [0,1]
    # ssim is the only metric extremely sensitive to gray being compared to b/w
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[0] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1[i], img2[i]))
            return np.array(ssims).mean()
        elif img1.shape[0] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')

def calculate_ssim(images1, images2):
    assert images1.shape == images2.shape

    ssim_results = []

    for image_num in range(images1.shape[0]):
        # image tensor [channel, h, w]
        img1 = images1[image_num].cpu().numpy()
        img2 = images2[image_num].cpu().numpy()

        # calculate per-image SSIM
        ssim_results.append(calculate_ssim_function(img1, img2))

    ssim_results = np.array(ssim_results)
    ssim_val = {}
    ssim_std = {}

    ssim_val[0] = np.mean(ssim_results)
    ssim_std[0] = np.std(ssim_results)

    result = {
        "value": ssim_val,
        "value_std": ssim_std,
        "tensor_setting": images1.shape,
        "tensor_setting_name": "batch, channel, height, width",
    }

    return result

# test code / using example

def main():
    NUMBER_OF_IMAGES = 8
    CHANNEL = 3
    SIZE = 64
    images1 = torch.zeros(NUMBER_OF_IMAGES, CHANNEL, SIZE, SIZE, requires_grad=False)
    images2 = torch.zeros(NUMBER_OF_IMAGES, CHANNEL, SIZE, SIZE, requires_grad=False)

    import json
    result = calculate_ssim(images1, images2)
    print(json.dumps(result, indent=4))

if __name__ == "__main__":
    main()
