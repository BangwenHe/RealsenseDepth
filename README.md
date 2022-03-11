# Realsense Depth

使用Intel Realsense D435i传感器进行深度的捕获。同时，为了纠正深度摄像头和RGB摄像头之间存在的物理距离，
我们使用了透视变换来将更大的RGB图像缩放到深度图上。最终的结果如下：

![](images/Snipaste_2022-03-11_20-22-21.png)