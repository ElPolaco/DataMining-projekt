#%%
import numpy as np

from matplotlib import pyplot as plt

from skimage import data, io
from skimage.util import img_as_float
from skimage.feature import (corner_harris, corner_subpix, corner_peaks,
                             plot_matches)
from skimage.transform import warp, AffineTransform
from skimage.exposure import rescale_intensity
from skimage.color import rgb2gray
from skimage.measure import ransac

img=(img_as_float(io.imread("5.jpg")))
#KANAŁ R
img_orig_r=np.zeros(list(img.shape))
# print(img_orig.shape)
# img_orig[...,0]= (img[...,0])
# img_orig[...,1]= (img[...,1])
# img_orig[...,2]= (img[...,2])
img_orig_r[...,0]=img[...,0]
gradient_r, gradient_c = (np.mgrid[0:img_orig_r.shape[0],
                                   0:img_orig_r.shape[1]]
                          / float(img_orig_r.shape[0]))
img_orig_r[...,1]=gradient_r#.repeat(3).reshape(img.shape)
img_orig_r[...,2]=gradient_c#.repeat(3).reshape(img.shape)
img_orig_r = rescale_intensity(img_orig_r)
img_orig_r_gray = rgb2gray(img_orig_r)

#KANAŁ G

img_orig_g=np.zeros(list(img.shape))
img_orig_g[...,0]=img[...,1]
gradient_r, gradient_c = (np.mgrid[0:img_orig_g.shape[0],
                                   0:img_orig_g.shape[1]]
                          / float(img_orig_g.shape[0]))
img_orig_g[...,1]=gradient_r
img_orig_g[...,2]=gradient_c
img_orig_g = rescale_intensity(img_orig_g)
img_orig_g_gray = rgb2gray(img_orig_g)

#KANAŁ B

img_orig_b=np.zeros(list(img.shape))
img_orig_b[...,0]=img[...,2]
gradient_r, gradient_c = (np.mgrid[0:img_orig_b.shape[0],
                                   0:img_orig_b.shape[1]]
                          / float(img_orig_b.shape[0]))
img_orig_b[...,1]=gradient_r
img_orig_b[...,2]=gradient_c
img_orig_b = rescale_intensity(img_orig_b)
img_orig_b_gray = rgb2gray(img_orig_b)

#%%
imgg= (img_as_float(io.imread("6.jpg")))
#KANAŁ R
img_warped_r=np.zeros(list(imgg.shape))
img_warped_r[...,0]=imgg[...,0]
gradient_r, gradient_c = (np.mgrid[0:img_warped_r.shape[0],
                                   0:img_warped_r.shape[1]]
                          / float(img_warped_r.shape[0]))
img_warped_r[...,1]=gradient_r
img_warped_r[...,2]=gradient_c
img_warped_r = rescale_intensity(img_warped_r)
img_warped_r_gray = rgb2gray(img_warped_r)

#KANAŁ G

img_warped_g=np.zeros(list(img.shape))
img_warped_g[...,0]=imgg[...,1]
gradient_r, gradient_c = (np.mgrid[0:img_warped_g.shape[0],
                                   0:img_warped_g.shape[1]]
                          / float(img_warped_g.shape[0]))
img_warped_g[...,1]=gradient_r
img_warped_g[...,2]=gradient_c
img_warped_g = rescale_intensity(img_warped_g)
img_warped_g_gray = rgb2gray(img_warped_g)

#KANAŁ B

img_warped_b=np.zeros(list(img.shape))
img_warped_b[...,0]=imgg[...,2]
gradient_r, gradient_c = (np.mgrid[0:img_warped_b.shape[0],
                                   0:img_warped_b.shape[1]]
                          / float(img_warped_b.shape[0]))
img_warped_b[...,1]=gradient_r
img_warped_b[...,2]=gradient_c
img_warped_b = rescale_intensity(img_warped_b)
img_warped_b_gray = rgb2gray(img_warped_b)
# img_warped=np.zeros(list(imgg.shape)+[3])
# print(img_warped.shape)
# # img_warped[...,0]=(imgg[...,0])
# # img_warped[...,1]=(imgg[...,1])
# # img_warped[...,2]=(imgg[...,2])
# img_warped[...,0]=imgg
# gradient_r, gradient_c = (np.mgrid[0:img_warped.shape[0],
#                                    0:img_warped.shape[1]]
#                           / float(img_warped.shape[0]))
# img_warped[...,1]=gradient_r.repeat(3).reshape(imgg.shape)
# img_warped[...,2]=gradient_c.repeat(3).reshape(imgg.shape)
# img_warped = rescale_intensity(img_warped)
# img_warped_gray = rgb2gray(rgb2gray(img_warped))

def gaussian_weights(window_ext, sigma=1):
    y, x = np.mgrid[-window_ext:window_ext+1, -window_ext:window_ext+1]
    g = np.zeros(y.shape, dtype=np.double)
    g[:] = np.exp(-0.5 * (x**2 / sigma**2 + y**2 / sigma**2))
    g /= 2 * np.pi * sigma * sigma
    return g


def match_corner(coord,coords_warped,coords_warped_subpix,img_orig,img_warped,window_ext=0):
    r, c = np.round(coord).astype(np.intp)
    window_orig = img_orig[r-window_ext:r+window_ext+1,c-window_ext:c+window_ext+1, :]

    # weight pixels depending on distance to center pixel
    weights = gaussian_weights(window_ext, 3)
    weights = np.dstack((weights, weights, weights))

    # compute sum of squared differences to all corners in warped image
    SSDs = []
    for cr, cc in coords_warped:
        window_warped = img_warped[cr-window_ext:cr+window_ext+1, cc-window_ext:cc+window_ext+1, :]
        SSD = np.sum(weights * (window_orig - window_warped)**2)
        SSDs.append(SSD)

    # use corner with minimum SSD as correspondence
    min_idx = np.argmin(SSDs)
    return coords_warped_subpix[min_idx]
# coords_orig = np.concatenate((corner_peaks(corner_harris((img_orig_r_gray)), threshold_rel=0.001, min_distance=5),
                            #   corner_peaks(corner_harris(img_orig_g_gray), threshold_rel=0.001, min_distance=5),
                            #   corner_peaks(corner_harris(img_orig_b_gray), threshold_rel=0.001, min_distance=5)))
# coords_warped =np.concatenate((corner_peaks(corner_harris(img_warped_r_gray),threshold_rel=0.001, min_distance=5),
                            #    corner_peaks(corner_harris(img_warped_g_gray),threshold_rel=0.001, min_distance=5),
                            #    corner_peaks(corner_harris(img_warped_b_gray),threshold_rel=0.001, min_distance=5)))

coords_orig_r = corner_peaks(corner_harris((img_orig_r_gray)), threshold_rel=0.001, min_distance=5)
coords_orig_g = corner_peaks(corner_harris(img_orig_g_gray), threshold_rel=0.001, min_distance=5)
coords_orig_b = corner_peaks(corner_harris(img_orig_b_gray), threshold_rel=0.001, min_distance=5)
coords_warped_r = corner_peaks(corner_harris(img_warped_r_gray),threshold_rel=0.001, min_distance=5)
coords_warped_g = corner_peaks(corner_harris(img_warped_g_gray),threshold_rel=0.001, min_distance=5)
coords_warped_b = corner_peaks(corner_harris(img_warped_b_gray),threshold_rel=0.001, min_distance=5)
# coords_orig_subpix = np.concatenate((corner_subpix(img_orig_r_gray, coords_orig, window_size=9),
#                                      corner_subpix(img_orig_g_gray, coords_orig, window_size=9),
#                                      corner_subpix(img_orig_b_gray, coords_orig, window_size=9)))
# coords_warped_subpix = np.concatenate((corner_subpix(img_warped_r_gray, coords_warped,window_size=9),
#                                        corner_subpix(img_warped_g_gray, coords_warped,window_size=9),
#                                        corner_subpix(img_warped_b_gray, coords_warped,window_size=9)))


coords_orig_subpix_r = corner_subpix(img_orig_r_gray, coords_orig_r, window_size=9)
coords_orig_subpix_g = corner_subpix(img_orig_g_gray, coords_orig_g, window_size=9)
coords_orig_subpix_b = corner_subpix(img_orig_b_gray, coords_orig_b, window_size=9)
coords_warped_subpix_r = corner_subpix(img_warped_r_gray, coords_warped_r,window_size=9)
coords_warped_subpix_g = corner_subpix(img_warped_g_gray, coords_warped_g,window_size=9)
coords_warped_subpix_b = corner_subpix(img_warped_b_gray, coords_warped_b,window_size=9)

src_r= []
dst_r= []
for coord in coords_orig_subpix_r:
    src_r.append(coord)
    dst_r.append(match_corner(coord,coords_warped=coords_warped_r,coords_warped_subpix=coords_warped_subpix_r,img_orig=img_orig_r,img_warped=img_warped_r))

src_g=[]
dst_g=[]
for coord in coords_orig_subpix_g:
    src_g.append(coord)
    dst_g.append(match_corner(coord,coords_warped=coords_warped_g,coords_warped_subpix=coords_warped_subpix_g,img_orig=img_orig_g,img_warped=img_warped_g))

src_b=[]
dst_b=[]
for coord in coords_orig_subpix_b:
    src_b.append(coord)
    dst_b.append(match_corner(coord,coords_warped=coords_warped_b,coords_warped_subpix=coords_warped_subpix_b,img_orig=img_orig_b,img_warped=img_warped_b))
src_r = np.array(src_r)
dst_r = np.array(dst_r)
src_b = np.array(src_b)
dst_b = np.array(dst_b)

src_g = np.array(src_g)
dst_g = np.array(dst_g)


# model_robust, inliers = ransac(((src_r, dst_r),(src_g,dst_g),(src_b,dst_b)), AffineTransform, min_samples=3,
#                                residual_threshold=2, max_trials=100)
# outliers = inliers == False
# coords_orig = corner_peaks(corner_harris(img_orig_r_gray), threshold_rel=0.001,
#                            min_distance=5)
# coords_warped = corner_peaks(corner_harris(img_warped_r_gray),
#                              threshold_rel=0.001, min_distance=5)

# # determine sub-pixel corner position
# coords_orig_subpix = corner_subpix(img_orig_r_gray, coords_orig, window_size=9)
# coords_warped_subpix = corner_subpix(img_warped_r_gray, coords_warped,
#                                      window_size=9)


#find correspondences using simple weighted sum of squared differences
# def match_corner(coord, window_ext=0):
#     r, c = np.round(coord).astype(np.intp)
#     window_orig = img[r-window_ext:r+window_ext+1,
#                            c-window_ext:c+window_ext+1, :]

#     # weight pixels depending on distance to center pixel
#     weights = gaussian_weights(window_ext, 3)
#     weights = np.dstack((weights, weights, weights))

#     # compute sum of squared differences to all corners in warped image
#     SSDs = []
#     for cr, cc in coords_warped:
#         window_warped = imgg[cr-window_ext:cr+window_ext+1,
#                                    cc-window_ext:cc+window_ext+1, :]
#         SSD = np.sum(weights * (window_orig - window_warped)**2)
#         SSDs.append(SSD)

#     # use corner with minimum SSD as correspondence
#     min_idx = np.argmin(SSDs)
#     return coords_warped_subpix[min_idx]

# src = []
# dst = []
# for coord in coords_orig_subpix:
#     src.append(coord)
#     dst.append(match_corner(coord))
# src = np.array(src)
# dst = np.array(dst)


# estimate affine transform model using all coordinates
model_r = AffineTransform()
model_r.estimate(src_r, dst_r)

model_g = AffineTransform()
model_g.estimate(src_g, dst_g)

model_b = AffineTransform()
model_b.estimate(src_b, dst_b)

# robustly estimate affine transform model with RANSAC
model_robust_r, inliers_r = ransac((src_r , dst_r ), AffineTransform, min_samples=1,
                               residual_threshold=2, max_trials=100)

model_robust_g, inliers_g = ransac((src_g , dst_g ), AffineTransform, min_samples=1,
                               residual_threshold=2, max_trials=100)

model_robust_b, inliers_b = ransac((src_b , dst_b ), AffineTransform, min_samples=1,
                               residual_threshold=2, max_trials=100)
fig, ax = plt.subplots(nrows=3, ncols=1)

plt.gray()

inlier_idxs = np.nonzero(inliers_r)[0]
plot_matches(ax[0], img, imgg, src_r, dst_r,
             np.column_stack((inlier_idxs, inlier_idxs)), matches_color='b')
ax[0].axis('off')
ax[0].set_title('R Channel')

inlier_idxs = np.nonzero(inliers_g)[0]
plot_matches(ax[1], img, imgg, src_g, dst_g,
             np.column_stack((inlier_idxs, inlier_idxs)), matches_color='r')
ax[1].axis('off')
ax[1].set_title('G Channel')

inlier_idxs = np.nonzero(inliers_b)[0]
plot_matches(ax[2], img, imgg, src_b, dst_b,
             np.column_stack((inlier_idxs, inlier_idxs)), matches_color='g')
ax[2].axis('off')
ax[2].set_title('B Channel')


plt.show()
