# tfjsInNode
using @tfjs-node 
2019.11.21 mtcnnload.js 新增node-image库进行图像分辨率转换处理。导入mobilenetv1分类模型。
2019.11.22 delectFace.js 新增图像金字塔处理图片。尝试导入Pnet模型
2019.11.25 delectFace.js toolMatrix.js 新增detect_face_12net、NMS函数，完成了python的矩阵操作在js上的实现。
2019.11.26 delectFace.js toolMatrix.js 完善了detect_face_12net，尝试使用tfjs api简化图片处理过程。成功将keras框架的网络导入到项目中。
2019.11.27 toolMatrix.js 完善了NMS函数，新增了rect2square函数。自定义数据测试函数的正确性
2019.11.28  mtcnnload.js delectFace.js toolMatrix.js 导入了Rnet和Onet，完善了代码，已经实现基本的多人脸检测，对于某些图片仍然存在丢失部分人脸的现象。
2019.11.29 inceptionResNetV2.js 尝试构建inception_ResNet_V2网络。
2019.12.2 inceptionResNetV2.js mtcnnload.js 构建了可训练inception_ResNet_V2，初步导入facenet预训练模型，该模型可以将人脸图片转化成128维的特征向量。其中lambda层还为完全导入，所以目前还不能预测
2019.12.3 mtcnnload.js 为了实现模型的lambda自定义keras层，在js中注册了自定义层，但由于json模型文件中的层函数被转码了，所以在python上进行测试。初步求出两张人脸之间的欧氏距离，由于模型不完整所以结果有些错误。
2019.12.4 facenet.js 完成了对输入数据的标准化操作，实现函数prewhiten
2019.12.5 facenet.js 导入了新inception_ResNet_V1模型，对输入数据添加人人脸检测，处理集中人脸区域，但结果还是没达到预期。
2019.12.6 facenet.js 发现了导入模型的错误，成功导入了模型。并将结果l2正则化使在js预测的结果与python上一致。看了一些分类器算法，接下来要实现将结果分类
2019.12.9 randomForest.js 利用ml-random-forest包实现随机森林来将人脸特征向量与对应的人进行分类，在原有基础上再构建出128个分类器，利用128个分类器再对结果进行投票，可以再一定程度上解决人脸图像较少的问题。