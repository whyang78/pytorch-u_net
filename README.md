# pytorch-u_net
对原始网络进行了改进，使用padding进行数据增强，使得编码输入的图片大小与编码输出的图片大小一致，使得更加方便。   
同时上采样的时候采用了两种方法upsample和反卷积。    
使用的2015_BOE_Chiu数据集.   
问题：batchsize image_size 受GPU内存限制故设置较小，可以适当增大。   

参考网站：https://www.cnblogs.com/kk17/p/9883565.html （unet论文讲解）
         https://blog.csdn.net/maliang_1993/article/details/82084983 （网络结构理解）
