import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy import signal
from PIL import Image

nearImage = cv2.imread("./Q4_01_near.jpg", cv2.IMREAD_UNCHANGED)
farImage = cv2.imread("./Q4_02_far.jpg", cv2.IMREAD_UNCHANGED)
height, width, channels = nearImage.shape
farImage = cv2.resize(farImage, dsize=(width , height), interpolation=cv2.INTER_CUBIC)

near_fft = np.zeros((height, width, channels), 'complex')
near_shifted_image = np.zeros((height, width, channels), 'complex')
near_amplitude_image = np.zeros((height, width, channels))
near_log_amplitude_image = np.zeros((height, width, channels))
near_real_log_amplitude_image = np.zeros((height, width, channels))

near_fft[..., 0] = np.fft.fft2(nearImage[..., 0])
near_fft[..., 1] = np.fft.fft2(nearImage[..., 1])
near_fft[..., 2] = np.fft.fft2(nearImage[..., 2])

near_shifted_image[..., 0] = np.fft.fftshift(near_fft[..., 0])
near_shifted_image[..., 1] = np.fft.fftshift(near_fft[..., 1])
near_shifted_image[..., 2] = np.fft.fftshift(near_fft[..., 2])

near_amplitude_image[..., 0] = np.abs(near_shifted_image[..., 0])
near_amplitude_image[..., 1] = np.abs(near_shifted_image[..., 1])
near_amplitude_image[..., 2] = np.abs(near_shifted_image[..., 2])

near_log_amplitude_image[..., 0] = np.log(near_amplitude_image[..., 0])
near_log_amplitude_image[..., 1] = np.log(near_amplitude_image[..., 1])
near_log_amplitude_image[..., 2] = np.log(near_amplitude_image[..., 2])

near_real_log_amplitude_image[..., 0] = np.real(near_log_amplitude_image[..., 0])
near_real_log_amplitude_image[..., 1] = np.real(near_log_amplitude_image[..., 1])
near_real_log_amplitude_image[..., 2] = np.real(near_log_amplitude_image[..., 2])

near_real_log_amplitude_image = near_real_log_amplitude_image / near_real_log_amplitude_image.max() * 255
cv2.imwrite("‫‪Q4_05_dft_near.jpg‬‬",near_real_log_amplitude_image)


far_fft = np.zeros((height, width, channels), 'complex')
far_shifted_image = np.zeros((height, width, channels), 'complex')
far_amplitude_image = np.zeros((height, width, channels))
far_log_amplitude_image = np.zeros((height, width, channels))
far_real_log_amplitude_image = np.zeros((height, width, channels))

far_fft[..., 0] = np.fft.fft2(farImage[..., 0])
far_fft[..., 1] = np.fft.fft2(farImage[..., 1])
far_fft[..., 2] = np.fft.fft2(farImage[..., 2])

far_shifted_image[..., 0] = np.fft.fftshift(far_fft[..., 0])
far_shifted_image[..., 1] = np.fft.fftshift(far_fft[..., 1])
far_shifted_image[..., 2] = np.fft.fftshift(far_fft[..., 2])

far_amplitude_image[..., 0] = np.abs(far_shifted_image[..., 0])
far_amplitude_image[..., 1] = np.abs(far_shifted_image[..., 1])
far_amplitude_image[..., 2] = np.abs(far_shifted_image[..., 2])

far_log_amplitude_image[..., 0] = np.log(far_amplitude_image[..., 0])
far_log_amplitude_image[..., 1] = np.log(far_amplitude_image[..., 1])
far_log_amplitude_image[..., 2] = np.log(far_amplitude_image[..., 2])

far_real_log_amplitude_image[..., 0] = np.real(far_log_amplitude_image[..., 0])
far_real_log_amplitude_image[..., 1] = np.real(far_log_amplitude_image[..., 1])
far_real_log_amplitude_image[..., 2] = np.real(far_log_amplitude_image[..., 2])

far_real_log_amplitude_image = far_real_log_amplitude_image / far_real_log_amplitude_image.max() * 255
cv2.imwrite("‫‪Q4_06_dft_far.jpg‬‬",far_real_log_amplitude_image)


def lowpass_filter(D0):
    mat = np.zeros((height, width))
    for i in range(-int(height / 2), int(height / 2)):
        for j in range(-int(width / 2), int(width / 2)):
            D = math.sqrt(j ** 2 + i ** 2)
            mat[i + int(height / 2), j + (width // 2)] = math.exp(-(D ** 2) / (2 * (D0 ** 2)))
    return mat


r = 18
s = 15

lpf = lowpass_filter(s)
lpf2 = lpf / lpf.max() * 255
plt.imshow(lpf)
plt.show()
cv2.imwrite("Q4_08_lowpass_" + str(s) + ".jpg", lpf2)

hpf = 1 - lowpass_filter(r)
hpf2 = hpf / hpf.max() * 255
plt.imshow(hpf)
plt.show()
cv2.imwrite('‫‪Q4_07_highpass_‬'+ str(r) + '.jpg', hpf2)


def cutoff(D0):
    filter = np.zeros((height, width))
    for i in range(-int(height / 2), int(height / 2)):
        for j in range(-int(width / 2), int(width / 2)):
            if math.sqrt(i ** 2 + j ** 2) < D0:
                filter[i + int(height / 2), j + (width // 2)] = 1
    return filter


lp_cutoff = np.multiply(cutoff(18), lpf)
lpc = lp_cutoff / lp_cutoff.max() * 255
cv2.imwrite("Q4_10_lowpass_cutoff.jpg", lpc)

hp_cutoff = np.multiply(1 - cutoff(18) , hpf)
hpc = hp_cutoff / hp_cutoff.max() * 255
cv2.imwrite("Q4_9_highpass_cutoff.jpg" , hpc)


near_con = np.zeros((height, width, channels), 'complex')
near_con[..., 0] = np.multiply(near_shifted_image[..., 0], hp_cutoff)
near_con[..., 1] = np.multiply(near_shifted_image[..., 1], hp_cutoff)
near_con[..., 2] = np.multiply(near_shifted_image[..., 2], hp_cutoff)

near_shifted = np.zeros((height, width, channels), 'complex')
near_shifted[..., 0] = np.fft.ifftshift(near_con[..., 0])
near_shifted[..., 1] = np.fft.ifftshift(near_con[..., 1])
near_shifted[..., 2] = np.fft.ifftshift(near_con[..., 2])

near_im = np.zeros((height, width, channels), 'complex')
near_im[..., 0] = np.fft.ifft2(near_shifted[..., 0])
near_im[..., 1] = np.fft.ifft2(near_shifted[..., 1])
near_im[..., 2] = np.fft.ifft2(near_shifted[..., 2])
near_im = np.real(near_im)
near_im = near_im / near_im.max() * 255
cv2.imwrite("Q4_11_highpassed.jpg‬‬" ,  near_im)


far_con = np.zeros((height, width, channels), 'complex')
far_con[..., 0] = np.multiply(far_shifted_image[..., 0], lp_cutoff)
far_con[..., 1] = np.multiply(far_shifted_image[..., 1], lp_cutoff)
far_con[..., 2] = np.multiply(far_shifted_image[..., 2], lp_cutoff)

far_shifted = np.zeros((height, width, channels), 'complex')
far_shifted[..., 0] = np.fft.ifftshift(far_con[..., 0])
far_shifted[..., 1] = np.fft.ifftshift(far_con[..., 1])
far_shifted[..., 2] = np.fft.ifftshift(far_con[..., 2])

far_im = np.zeros((height, width, channels), 'complex')
far_im[..., 0] = np.fft.ifft2(far_shifted[..., 0])
far_im[..., 1] = np.fft.ifft2(far_shifted[..., 1])
far_im[..., 2] = np.fft.ifft2(far_shifted[..., 2])
far_im = np.real(far_im)
cv2.imwrite("Q4_12_lowpassed.jpg‬‬" ,  far_im)


final = np.zeros((height , width , channels) , 'complex')
final[... , 0] = (3*near_con[... , 0] + 2 * far_con[... , 0]) / 5
final[... , 1] = (3*near_con[... , 1] + 2 * far_con[... , 1]) / 5
final[... , 2] = (3*near_con[... , 2] + 2 * far_con[... , 2]) / 5

final_shifted = np.zeros((height , width , channels) , 'complex')
final_shifted[... , 0] = np.fft.ifftshift(final[... , 0])
final_shifted[... , 1] = np.fft.ifftshift(final[... , 1])
final_shifted[... , 2] = np.fft.ifftshift(final[... , 2])

final_im = np.zeros((height , width , channels))
final_im[... , 0] = np.fft.ifft2(final_shifted[... , 0])
final_im[... , 1] = np.fft.ifft2(final_shifted[... , 1])
final_im[... , 2] = np.fft.ifft2(final_shifted[... , 2])


final_im[... , 0] = np.real(final_im[... , 0])
final_im[... , 1] = np.real(final_im[... , 1])
final_im[... , 2] = np.real(final_im[... , 2])

cv2.imwrite("Q4_13_hybrid_frequency.jpg", np.array(final , 'uint8'))

cv2.imwrite("Q4_14_hybrid_near.jpg.jpg",final_im)

hybrid_far = cv2.resize(final_im,(int(width / 5), int(height / 5)), interpolation=cv2.INTER_AREA)
cv2.imwrite("Q4_15_hybrid_far.jpg", hybrid_far)