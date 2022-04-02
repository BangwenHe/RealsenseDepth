from builtins import all
import numpy as np
import cv2
import matplotlib.pyplot as plt

joints_name = ('Head_top', 'Thorax', 'R_Shoulder', 'R_Elbow', 'R_Wrist', 'L_Shoulder', 'L_Elbow', 'L_Wrist', 'R_Hip', 'R_Knee', 'R_Ankle', 'L_Hip', 'L_Knee', 'L_Ankle', 'Pelvis', 'Spine', 'Head', 'R_Hand', 'L_Hand', 'R_Toe', 'L_Toe')
joint_num = len(joints_name) # 21
SKELETON = ( (0, 16), (16, 1), (1, 15), (15, 14), (14, 8), (14, 11), (8, 9), (9, 10), (10, 19), (11, 12), (12, 13), (13, 20), (1, 2), (2, 3), (3, 4), (4, 17), (1, 5), (5, 6), (6, 7), (7, 18) )

cmap = plt.get_cmap('rainbow')
colors = [cmap(i) for i in np.linspace(0, 1, joint_num + 2)]
colors_cv = [(c[2] * 255, c[1] * 255, c[0] * 255) for c in colors]
colors_plt = [np.array((c[0], c[1], c[2])) for c in colors]

fig = plt.figure(figsize=(8, 6), dpi=200)
ax = fig.add_subplot(111, projection='3d')

def crop_image(image, bbox):

    xmin = int(bbox[0])
    ymin = int(bbox[1])
    xmax = int(bbox[2])
    ymax = int(bbox[3])

    input_image = image[ymin:ymax,xmin:xmax]

    return input_image

def pixel2cam(pixel_coord, f, c):
    # use original `pixel2cam` function with accurate focals and principals
    x = (pixel_coord[:, 0] - c[0]) / f[0] * pixel_coord[:, 2]
    y = (pixel_coord[:, 1] - c[1]) / f[1] * pixel_coord[:, 2]
    z = pixel_coord[:, 2]
    cam_coord = np.concatenate((x[:,None], y[:,None], z[:,None]),1)
    return cam_coord

def draw_skeleton(img, keypoints, scores, kp_thres = 0.02):
    vis_img = img.copy()

    for i, segment in enumerate(SKELETON):
        point1_id = segment[0]
        point2_id = segment[1]

        point1 = (int(keypoints[point1_id, 0]), int(keypoints[point1_id, 1]))
        point2 = (int(keypoints[point2_id, 0]), int(keypoints[point2_id, 1]))

        vis_img = cv2.line(vis_img, point1, point2,colors_cv[i], thickness=3, lineType=cv2.LINE_AA)
        cv2.circle(vis_img, point1, radius=5, color=colors_cv[i], thickness=-1, lineType=cv2.LINE_AA)
        cv2.circle(vis_img, point2, radius=5, color=colors_cv[i], thickness=-1, lineType=cv2.LINE_AA)

        ''' In case confidence is used 
        if scores[point1_id] > kp_thres and scores[point2_id] > kp_thres:
            img = cv2.line(img, point1, point2,colors[i], thickness=3, lineType=cv2.LINE_AA)

        if scores[point1_id] > kp_thres:
            cv2.circle(img, point1, radius=5, color=colors[i], thickness=-1, lineType=cv2.LINE_AA)

        if scores[point2_id] > kp_thres:   
            cv2.circle(img, point2, radius=5, color=colors[i], thickness=-1, lineType=cv2.LINE_AA)

        '''

    return vis_img

def draw_heatmap(img, img_heatmap):

    norm_heatmap = (255*((img_heatmap-np.min(img_heatmap))/(np.max(img_heatmap) - np.min(img_heatmap))))
    color_heatmap = cv2.applyColorMap(cv2.convertScaleAbs(norm_heatmap,1), cv2.COLORMAP_MAGMA)
    return cv2.addWeighted(img, 0.4, color_heatmap, 0.6, 0)

def vis_3d_multiple_skeleton(kpt_3d, kpt_3d_vis, skeleton=SKELETON, filename=None, 
                            model_type="hg", output_dir="output_viewpoints", all_angle=False):
    ax.cla()
    for l in range(len(skeleton)):
        i1 = skeleton[l][0]
        i2 = skeleton[l][1]

        person_num = kpt_3d.shape[0]
        for n in range(person_num):
            x = np.array([kpt_3d[n,i1,0], kpt_3d[n,i2,0]])
            y = np.array([kpt_3d[n,i1,1], kpt_3d[n,i2,1]])
            z = np.array([kpt_3d[n,i1,2], kpt_3d[n,i2,2]])

            if kpt_3d_vis[n,i1,0] > 0 and kpt_3d_vis[n,i2,0] > 0:
                ax.plot(x, z, -y, color=colors_plt[l], linewidth=4)
            if kpt_3d_vis[n,i1,0] > 0:
                ax.scatter(kpt_3d[n,i1,0], kpt_3d[n,i1,2], -kpt_3d[n,i1,1], color=colors_plt[l], marker='o', s=40)
            if kpt_3d_vis[n,i2,0] > 0:
                ax.scatter(kpt_3d[n,i2,0], kpt_3d[n,i2,2], -kpt_3d[n,i2,1], color=colors_plt[l], marker='o', s=40)

            if kpt_3d[n,i1,2] == 5:
                ax.scatter(kpt_3d[n,i1,0], kpt_3d[n,i1,2], -kpt_3d[n,i1,1], color=((0, 0, 0)), marker='d', s=40)
    #  Hide grid lines
    # ax.w_zaxis.line.set_lw(0.)
    # ax.xaxis.pane.fill = False
    # ax.xaxis.pane.set_edgecolor('white')
    # ax.yaxis.pane.fill = False
    # ax.yaxis.pane.set_edgecolor('white')
    # ax.zaxis.pane.fill = False
    # ax.zaxis.pane.set_edgecolor('white')
    # ax.grid(False)
    # ax.set_xticks([])
    # ax.set_yticks([])
    # ax.set_zticks([])
    # ax.view_init(elev=7, azim=-74)
    # plt.tight_layout()

    # https://stackoverflow.com/questions/29988241/python-hide-ticks-but-show-tick-labels
    # plt.setp(ax.get_xticklabels(), visible=False)
    # plt.setp(ax.get_yticklabels(), visible=False)
    # plt.setp(ax.get_zticklabels(), visible=False)

    ax.set_xlabel("$x$")
    ax.set_ylabel("$z$")
    ax.set_zlabel("$y$")
    # ax.set_ylim(500, 2500)
    # ax.set_zlim(-400, 500)

    model_type = model_type.lower()
    assert model_type in ("mhp", "gt", "hg", "tp")

    from tqdm import tqdm
    # if model_type == "mhp":
    #     depth_scale_ratio = 1
    #     min_depth = kpt_3d[:, :, 2].min()
    #     max_depth = kpt_3d[:, :, 2].max()
    #     mean_depth = (min_depth + max_depth) / 2
    #     diff_depth = max_depth - min_depth
    #     ax.set_ylim(mean_depth - depth_scale_ratio*diff_depth, mean_depth + depth_scale_ratio*diff_depth)

    # elif model_type == "hg":
    #     depth_scale_ratio = 1
    #     min_depth = kpt_3d[:, :, 2].min()
    #     max_depth = kpt_3d[:, :, 2].max()
    #     mean_depth = (min_depth + max_depth) / 2
    #     diff_depth = max_depth - min_depth
    #     ax.set_ylim(mean_depth - depth_scale_ratio*diff_depth, mean_depth + depth_scale_ratio*diff_depth)

    # elif model_type == "tp":
    depth_scale_ratio = 1
    min_depth = kpt_3d[:, :, 2].min()
    max_depth = kpt_3d[:, :, 2].max()
    mean_depth = (min_depth + max_depth) / 2
    diff_depth = max_depth - min_depth
    ax.set_ylim(mean_depth - depth_scale_ratio*diff_depth, mean_depth + depth_scale_ratio*diff_depth)

    x_scale_ratio = 0.45
    min_x = kpt_3d[:, :, 0].min()
    max_x = kpt_3d[:, :, 0].max()
    mean_x = (min_x + max_x) / 2
    diff_x = max_x - min_x
    ax.set_xlim(mean_x - x_scale_ratio*diff_x, mean_x + x_scale_ratio*diff_x)

    if all_angle:
        for i in tqdm(range(0, 360)):
            ax.view_init(elev=7, azim=i)
            plt.savefig(f"{output_dir}/{model_type}_axim_{i}.jpg")

    # azims = [(0, 180), (0, 90), (-90, 0), (-45, 90)]
    # for i in range(len(azims)):
    #     ax.view_init(elev=azims[i][0], azim=azims[i][1])
    #     plt.savefig(f"elev_{i}.jpg")

    # import sys
    # sys.exit(0)

    ax.view_init(elev=7, azim=120)
    fig.canvas.draw()
    img_3dpos = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img_3dpos = img_3dpos.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    # img is rgb, convert to opencv's default bgr
    img_3dpos = cv2.cvtColor(img_3dpos,cv2.COLOR_RGB2BGR)

    return img_3dpos
