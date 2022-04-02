import copy
import glob
import numpy as np
import cv2

# 加载参数
def load_stereo_coefficients(path):
    """ Loads stereo matrix coefficients. """
    # FILE_STORAGE_READ
    cv_file = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)

    # note we also have to specify the type to retrieve other wise we only get a
    # FileNode object back instead of a matrix
    K1 = cv_file.getNode("K1").mat()
    D1 = cv_file.getNode("D1").mat()
    K2 = cv_file.getNode("K2").mat()
    D2 = cv_file.getNode("D2").mat()
    R = cv_file.getNode("R").mat()
    T = cv_file.getNode("T").mat()
    E = cv_file.getNode("E").mat()
    F = cv_file.getNode("F").mat()
    R1 = cv_file.getNode("R1").mat()
    R2 = cv_file.getNode("R2").mat()
    P1 = cv_file.getNode("P1").mat()
    P2 = cv_file.getNode("P2").mat()
    Q = cv_file.getNode("Q").mat()

    cv_file.release()
    return [K1, D1, K2, D2, R, T, E, F, R1, R2, P1, P2, Q]


# 旋转+裁剪图像
def rotate(image_path,isNeedRotate=True,scale=None,shape=None):

    image_file = image_path.replace("\\", "/", 10)
    image = cv2.imread(image_file)
    if isNeedRotate:
        image = np.rot90(image,3)
    if scale is not None:
        h, w, _ = image.shape
        s_x, e_x, s_y, e_y = int((scale[0] - 1) * w / (2 * scale[0])), int(
            (scale[0] + 1) * w / (2 * scale[0])), int(
            (scale[1] - 1) * h / (2 * scale[1])), int((scale[1] + 1) * h / (2 * scale[1]))
        cropped_img1 = image[s_y:e_y, s_x:e_x, :]
        cropped_img1 = cv2.resize(cropped_img1, dsize=shape)
        image = cropped_img1
    return image

def disparity_map(imgL, imgR):
    window_size = 5 # wsize default 3; 5; 7 for SGBM reduced size image; 15 for SGBM full size image (1300px and above); 5 Works nicely
    # print(imgR.shape)
    left_matcher = cv2.StereoSGBM_create(
        minDisparity= 0,
        numDisparities= 6 * 16,  # max_disp has to be dividable by 16 f. E. HH 192, 256
        blockSize=window_size,
        P1= 16 * 3 * window_size ,
        # wsize default 3; 5; 7 for SGBM reduced size image; 15 for SGBM full size image (1300px and above); 5 Works nicely
        P2= 48 * 3 * window_size,
        disp12MaxDiff= 1 ,
        uniquenessRatio= 10,
        speckleWindowSize= 96 ,
        speckleRange= 50,
        preFilterCap= 31,
        mode=cv2.STEREO_SGBM_MODE_HH,

    )
    right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)
    # FILTER Parameters
    lmbda = 80000
    sigma = 1.2
    visual_multiplier = 6

    wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=left_matcher)
    wls_filter.setLambda(lmbda)

    wls_filter.setSigmaColor(sigma)
    displ = left_matcher.compute(imgL, imgR)  # .astype(np.float32)/16
    dispr = right_matcher.compute(imgR, imgL)  # .astype(np.float32)/16
    displ = np.int16(displ)
    dispr = np.int16(dispr)
    filteredImg = wls_filter.filter(displ, imgL, None, dispr) # important to put "imgL" here!!!

    final_filteredImg = cv2.normalize(src=filteredImg, dst=filteredImg, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX)
    final_filteredImg = np.uint8(final_filteredImg)


    return displ,dispr,filteredImg,final_filteredImg

def get_depth(calibration_file,leftFrame,rightFrame):
    # 加载立体标定参数
    K1, D1, K2, D2, R, T, E, F, R1, R2, P1, P2, Q = load_stereo_coefficients(calibration_file)  # Get cams params

    height, width, channel = leftFrame.shape  # We will use the shape for remap
    # # Undistortion and Rectification part!

    leftMapX, leftMapY = cv2.initUndistortRectifyMap(K1, D1, R1, P1, (width, height), cv2.CV_32FC1)
    left_rectified = cv2.remap(leftFrame, leftMapX, leftMapY, cv2.INTER_LINEAR, cv2.BORDER_CONSTANT)

    rightMapX, rightMapY = cv2.initUndistortRectifyMap(K2, D2, R2, P2, (width, height), cv2.CV_32FC1)
    right_rectified = cv2.remap(rightFrame, rightMapX, rightMapY, cv2.INTER_LINEAR, cv2.BORDER_CONSTANT)

    gray_left = cv2.cvtColor(left_rectified, cv2.COLOR_BGR2GRAY)
    gray_right = cv2.cvtColor(right_rectified, cv2.COLOR_BGR2GRAY)

    #求解视差图
    disp_l,disp_r,filteredImg,final_filteredImg = disparity_map(gray_left, gray_right)  # Get the disparity map

    # 由视差图 -> 重投影 -> 深度图
    depth= cv2.reprojectImageTo3D(disp_l, Q, handleMissingValues= True)
    depth = depth * 16
    z = depth[:,:,2]



    disp = disp_l.astype(np.float32)/ 16.0
    disop_8U = cv2.normalize(disp,disp,alpha=0,beta=255,norm_type=cv2.NORM_MINMAX,dtype=cv2.CV_8U)
    color_disp_l=cv2.applyColorMap(disop_8U,cv2.COLORMAP_JET)


    # 突出显示6m以内的深度图
    z[ z >= 5 ]= 5
    vis_z = np.zeros(z.shape,dtype='uint8')
    cv2.normalize(np.abs(z),vis_z,0,255,cv2.NORM_MINMAX,dtype=cv2.CV_8U)
    vis_z = np.abs(255-vis_z)
    color_depth = cv2.applyColorMap(vis_z,cv2.COLORMAP_TURBO)

    return leftFrame, left_rectified, color_depth, z, leftMapX, leftMapY


if __name__ == '__main__':
    shape = [480,640]
    resize_scale = [1.0834863 ,1.0900041]

    imgs_path= "data/action"
    
    param_path = 'huawei_mate40pro_fixed_3.0m.yml'

    image_pathes_r = glob.glob(f"{imgs_path}/right*.png")
    image_pathes_l = glob.glob(f"{imgs_path}/left*.png")


    for i in range(0,len(image_pathes_l)):
        left_image_path = image_pathes_l[i]
        right_image_path = image_pathes_r[i]
        left_img = rotate(left_image_path,scale=resize_scale,shape = shape)
        right_img = rotate(right_image_path)

        get_depth(param_path, left_img, right_img)
