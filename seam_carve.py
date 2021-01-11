import numpy as np
import cv2
import matplotlib.pyplot as plt
from numba import jit


def get_gradient_magnitude(img):
	"""
	:param img: The image to get gradient of
	:return: image gradient
	"""
	img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	gx = cv2.Sobel(img, cv2.CV_8U, 1, 0)
	gy = cv2.Sobel(img, cv2.CV_8U, 0, 1)
	grad = cv2.add(gx, gy)
	return grad


@jit
def get_opt_seam(img_grad):
	"""
	Using a bottom up approach to build
	smallest sum gradient paths (seam).

	Let gms(r,c) represent the gradient magnitude
	at row r and column c.

	Let f(r,c) represent the smallest sum of
	gradients starting from row r and col c going
	down column by column to the bottom of the image.

	f(r, c) = img_grad(r,c) +
				min {
					f(r+1, c-1), if 0 <= c-1 < # cols
					f(r+1, c), if 0 <= c < # cols
					f(r+1, c+1), if 0 <= c+1 < # cols
				}

	:param img_grad: gradient magnitudes
	:return: the optimal seam to remove
	"""
	rows, cols = img_grad.shape
	f = np.zeros((rows, cols))
	paths = np.zeros((rows, cols))
	opt_seam = []

	# copy last row of gradient magnitudes since
	# there is now beneath the last row
	f[-1] = img_grad[-1]

	# build seam matrix f based on recurrence
	for r in np.arange(rows - 2, -1, -1):
		for c in np.arange(cols):
			idx_start = max(0, c - 1)
			idx_end = min(c + 1, cols)
			idx_min_grad = np.argmin(f[r+1, idx_start:idx_end+1])
			if c == 0:
				paths[r, c] = c + idx_min_grad
				f[r, c] = img_grad[r, c] + f[r+1, (c + idx_min_grad)]
			else:
				paths[r, c] = c + idx_min_grad - 1
				f[r, c] = img_grad[r, c] + f[r+1, (c + idx_min_grad) - 1]

	# backtrack to get opt seam
	idx_curr_seam = np.argmin(f[0])
	for r in range(rows):
		opt_seam.append((r, int(idx_curr_seam)))
		idx_curr_seam = paths[r, int(idx_curr_seam)]

	return opt_seam


@jit
def remove_seam(img, seam):
	"""
	Remove seam from image.
	:param img: Original image to remove seam from
	:param seam: Seam to remove
	:return: return image with seam removed
	"""
	rows, cols = img.shape[:2]
	mask_img = np.ones_like(img, bool)
	seam_img = img.copy()
	for i in seam:
		seam_img[i[0], i[1]] = [57, 255, 20]
		mask_img[i[0], i[1]] = False
	res_img = img[mask_img].reshape(rows, cols-1, 3)
	return res_img


@jit
def seam_carve(img, dim):
    """
    Perform seam carving operation for the desired image.
    :param img: Original image
    :param dim: Desired dimensions (rows, cols)
    :return: Seam carved image with desired dimensions.
    """
    rows, cols = img.shape[:2]
    desired_rows, desired_cols = dim

    # remove cols
    while cols > desired_cols:
        img_grad = get_gradient_magnitude(img)
        seam = get_opt_seam(img_grad)
        img = remove_seam(img, seam)
        cols = img.shape[1]

    # remove rows
    while rows > desired_rows:
        img = cv2.transpose(img)
        img_grad = get_gradient_magnitude(img)
        seam = get_opt_seam(img_grad)
        img = remove_seam(img, seam)
        rows = img.shape[1]
        img = cv2.transpose(img)

    return img


def show_plots(images, titles):
	axes = []
	fig = plt.figure(figsize=(10, 60))
	fig_title = titles[-1]

	fig.suptitle(fig_title, fontsize=16)
	for img in range(len(images)):
		axes.append(fig.add_subplot(2, 2, img + 1))
		subplot_title = titles[img]
		plt.imshow(images[img])
		axes[-1].set_title(subplot_title)
	plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=0.5)
	plt.show()


if __name__ == "__main__":

	# -- EX1 --
	ex1 = cv2.cvtColor(cv2.imread("ex1.jpg"), cv2.COLOR_BGR2RGB)
	carved_ex1 = seam_carve(ex1, (968, 957))
	cropped_ex1 = ex1[0:969, 0:958]
	resized_ex1 = cv2.resize(ex1, (957, 968))

	show_plots([ex1,
				carved_ex1,
				cropped_ex1,
				resized_ex1],
			   ["Original Image - Ex1",
				"Seam Carve - 968 Rows x 957 Columns",
				"Ex1 - Cropped",
				"Ex1 - Scaled",
				"Example 1 Figure Plots"])