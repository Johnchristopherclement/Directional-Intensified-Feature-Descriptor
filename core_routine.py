import cv2
import numpy as np
from skimage.feature import hog


def get_grad(img):
    custom_filter1 = np.array([[0, -1, 0], [0, 0, 0], [0, 0, 0], [0, 1, 0]])
    custom_filter2 = np.array([[0, 0, 0], [-1, 0, 0], [1, 0, 0], [0, 0, 0]])
    custom_filter3 = np.array([[0, 0, 0], [0, 0, -1], [0, 0, 1], [0, 0, 0]])

    custom_filter4 = np.array([[0, 0, 0], [-1, 0, 1], [0, 0, 0], [0, 0, 0]])
    custom_filter5 = np.array([[0, 0, 0], [0, 0, 0], [-1, 0, 1], [0, 0, 0]])

    custom_filter6 = np.array([[0, 0, 0], [-1, 0, 0], [0, 0, 1], [0, 0, 0]])
    custom_filter7 = np.array([[0, 0, 0], [0, 0, -1], [1, 0, 0], [0, 0, 0]])

    filter_one = cv2.filter2D(img, cv2.CV_64F, custom_filter1)
    filter_two = cv2.filter2D(img, cv2.CV_64F, custom_filter2)
    filter_th = cv2.filter2D(img, cv2.CV_64F, custom_filter3)
    filter_fr = cv2.filter2D(img, cv2.CV_64F, custom_filter4)
    filter_fv = cv2.filter2D(img, cv2.CV_64F, custom_filter5)
    filter_sx = cv2.filter2D(img, cv2.CV_64F, custom_filter6)
    filter_sv = cv2.filter2D(img, cv2.CV_64F, custom_filter7)

    Gy = (filter_one ** 2 + filter_two ** 2 + filter_th ** 2) ** (0.5)
    Gx = (filter_fr ** 2 + filter_fv ** 2) ** (0.5)
    Gd = (filter_sx ** 2 + filter_sv ** 2) ** (0.5)

    weights = (Gx ** 2 + Gy ** 2 + Gd ** 2) ** 0.5

    ##theta = np.arctan2(Gy, Gx) * 180 / np.pi
    theta = np.arctan2(Gd, Gy) * 180 / np.pi + np.arctan2(Gd, Gx) * 180 / np.pi

    return weights, theta

def normalizeBlock(block,method,eps=1e-5):
    '''Normalize hog blocks, for hog()'''
    if method=='l1':
        norm=np.sum(np.abs(block))+eps
        result=block/norm
    elif method=='l2':
        norm=np.sqrt(np.sum(block**2)+eps**2)
        result=block/norm
    elif method=='l1-sqrt':
        norm=np.sum(np.abs(block))+eps
        result=np.sqrt(block/norm)
    return result
def getLine(beta,x,y):#pixels from image
    '''Get a line of pixels given slope and a point'''
    if abs(beta)>=1:
        xhat=y/beta
        result=np.where((x>=xhat) & (x<=xhat+1),1,0)
    else:
        yhat=x*beta
        result=np.where((y>=yhat) & (y<=yhat+1),1,0)
    return result

def compute_hog(img,weights, theta, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(3, 3), block_norm='l1',
                visualise=False, transform_sqrt=False, feature_vector=True, verbose=True):
    ndim = np.ndim(img)
    if ndim != 2 and ndim != 3:
        raise Exception("<img> needs to be 2D or 3D.")

    if not isinstance(orientations, (int)) or orientations <= 0:
        raise Exception("<orientations> needs to be an positive int.")
    if not block_norm in ['l1', 'l2', 'l1-sqrt']:
        raise Exception("<block_norm> needs to be one of ['l1','l2','l1-sqrt'].")

    # -----------apply power law compression-----------
    if transform_sqrt:
        img = np.sqrt(img)

    if ndim == 2:
        height, width = img.shape
    else:
        height, width, channels = img.shape
    cy, cx = pixels_per_cell
    by, bx = cells_per_block

    # Get gradient orientations in [0,180]
    #theta,weights=get_grad(img)

    # -----Compute histogram of gradients in cells-----
    ny = height // cy
    nx = width // cx

    # ----------------Loop though cells----------------
    hists = []
    for ii in range(ny):
        histsii = []
        for jj in range(nx):
            thetaij = theta[ii * cy:(ii + 1) * cy, jj * cx:(jj + 1) * cx]
            weightsij = weights[ii * cy:(ii + 1) * cy, jj * cx:(jj + 1) * cx]
            histij, binedges = np.histogram(thetaij, bins=orientations,
                                            range=(0, 180), weights=weightsij)
            histij = histij / cy / cx
            histsii.append(histij)
        hists.append(histsii)
    hists = np.array(hists)

    # --------Create visualization of gradients--------
    if visualise:

        # NOTE: the choice of gradient angles would introduce some differences
        # to the resultant plot.
        # thetas=0.5*(binedges[:-1]+binedges[1:])
        thetas = binedges[:-1]
        thetas = np.tan((thetas + 90.) / 180. * np.pi)
        hog_image = np.zeros([height, width])
        #img = np.zeros([height, width])#line1removed
        xcell = np.arange(cx)
        ycell = np.arange(cy)
        xcell, ycell = np.meshgrid(xcell - xcell.mean(), ycell - ycell.mean())

        for ii in range(ny):
            for jj in range(nx):
                for kk, tkk in enumerate(thetas):
                    linekk = getLine(tkk, xcell, ycell) * hists[ii, jj, kk]
                    hog_image[ii * cy:(ii + 1) * cy, jj * cx:(jj + 1) * cx] += linekk
                    #img[ii * cy:(ii + 1) * cy, jj * cx:(jj + 1) * cx] += linekk#line2removed

    # --------------Normalize over blocks--------------
    feature = []
    for ii in range(ny - (by - 1)):
        fsii = []
        for jj in range(nx - (bx - 1)):
            blockij = hists[ii:ii + by, jj:jj + bx, :]
            blockij = normalizeBlock(blockij, block_norm)
            fsii.append(blockij)
        feature.append(fsii)

    feature = np.array(feature)

    if feature_vector:
        feature = feature.flatten()

    if visualise:
        return feature, hog_image
    else:
        return feature



