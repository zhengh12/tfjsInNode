const kmeans = require('ml-kmeans');
const facenet = require("./facenet")
const fs = require("fs");
const tf = require("@tensorflow/tfjs-node");
const RandomForest = require('ml-random-forest') 
const detectFace = require("./detectFace")
const { agnes } = require('ml-hclust');
const clustering = require('density-clustering');
const snapList = require('snap-to-grid-clustering').snapList
const {Matrix, EigenvalueDecomposition, covariance} = require('ml-matrix');

//主文件夹下包含多个子文件夹，每个子文件夹作为一个人脸分类，一个子文件夹包含至少一张人脸图片
async function loadFiles(dataPath, FacenetModel, Pnet, Rnet, Onet){
    let dirArr=[]
    let dir = fs.readdirSync(dataPath)
    dir.map(item=>{
        let stat = fs.lstatSync(dataPath + item)
        if (stat.isDirectory() === true) { 
            let subDirArr = []
            let subDir = fs.readdirSync(dataPath + item + '/')
            subDir.map(val=>{
                let stat = fs.lstatSync(dataPath + item + '/' + val)
                if(stat.isDirectory() === false){
                    subDirArr.push(dataPath + item + '/' + val)
                }
            })
            dirArr.push(subDirArr)
        }
    })
    let vectors = []
    for(let i=0; i<dirArr.length; i++){
        let subvector = []
        for(val of dirArr[i]){
            console.log("loading file: "+val)
            let vector = await facenet.faceVector(FacenetModel, val, Pnet, Rnet, Onet)
            vector = vector.arraySync()[0]
            // vector.push(i)
            // vector = vector.map((val,index)=>{
            //     return vector.slice(0,index).concat(vector.slice(index+1,vector.length)).concat([index.toString()])
            // })
            subvector.push(vector)
        }
        vectors.push(subvector)
    }
    console.log("loading file over")
    return vectors
}

async function createTree(){
    const trainDatapath = './public/images/RandomForestTrainData1/'
    const modelPath = "./public/model/Facenet1/model.json"
    const pModelPath = './public/model/Pnet/model.json'
    const rModelPath = './public/model/Rnet/model.json'
    const oModelPath = './public/model/Onet/model.json'
    const FacenetModel = await facenet.loadFacenetModel(modelPath)
    const mtcnnModel = await detectFace.loadModel(pModelPath, rModelPath, oModelPath)

    let vectors = await loadFiles(trainDatapath, FacenetModel, mtcnnModel[0], mtcnnModel[1], mtcnnModel[2])

    //求得输入数据的协方差矩阵，再求出协方差矩阵的特征值。
    //这样就可以求得各个特征分量在总体矩阵中的有关性占比。
    let vectorsAll = []
    for(subVectors of vectors){
        for(val of subVectors){
            vectorsAll.push(val)
        }
    } 
    let tensor = tf.tensor(vectorsAll)
    tensor = tf.sub(tensor, tf.mean(tensor,0)).arraySync()
    let vectorMatrix = new Matrix(tensor)
    let covMatrix = covariance(vectorMatrix) //协方差矩阵
    //console.log(covMatrix)
    let EigenvalueMatrix = new EigenvalueDecomposition(covMatrix); //特征类
    let real = EigenvalueMatrix.realEigenvalues //特征数组

    //console.log(real)
    // console.log(dirArr)
    // let vectors = []
    // dirArr.map((val,indexs)=>{
    //      val.map(val=>{
    //         let vector = facenet.faceVector(modelPath,val)
    //         vector = vector.arraySync()[0]
    //         vector = vector.map((val,index)=>{
    //             return vector.slice(0,index).concat(vector.slice(index+1,vector.length)).push(indexs)
    //         })
    //         vectors.push(vector)
    //     })
    // })
    //128维向量特征加权卷积后的维度为 128-convolutionSize+1
    //再对每一类取类间平均值
    //for(let j=2; j<=100; j=j+1){
        let convolutionSize = 64
        let convectors = []
        let convectorsAll = []
        for(subVectors of vectors){
            let avgVector = tf.scalar(0)
            for(val of subVectors){
                let convector = []
                for(let i=0; i<val.length; i++){
                    let EigenvalueSum = real.slice(i, i+convolutionSize).reduce(function (a, b) { return a + b;})
                    let res = i+convolutionSize<=val.length ? val.slice(i, i+convolutionSize).reduce(function (a, b, index) { return a + b * real[i+index] }, 0)/EigenvalueSum : null
                    if(res !== null){
                        convector.push(res)
                    }    
                }
                
                // let out = tf.tensor(val)
                // let out1 = tf.tensor([...convector,...val.slice(val.length-convolutionSize,val.length)])
                // let dist = tf.sqrt(tf.sum(tf.squaredDifference(out,out1)))
                // dist.print()
                avgVector = tf.add(avgVector, convector)
                convectorsAll.push(convector)
            }
            avgVector = tf.div(avgVector,tf.scalar(subVectors.length))
            convectors.push(avgVector.arraySync())
        }

        //层次聚类
        // const tree = agnes(convectorsAll, {
        //     method: 'ward',
        //     isDistanceMatrix: 'false'
        // });
        // let str = JSON.stringify(tree,"","\t")
        // // fs.writeFileSync('./public/randomForestJson/data.json',str)
        // console.log(str)

        //optics密度聚类
        // let optics = new clustering.OPTICS();
        // // parameters: 2 - neighborhood radius, 2 - number of points in neighborhood to form a cluster
        // let clusters = optics.run(convectorsAll, 0.5, 10);
        // let plot = optics.getReachabilityPlot();
        // console.log("j: ",j,clusters);

        //grid网络聚类
        // console.log(snapList(convectorsAll, 1))

        // //手动选择距离最远的三个点作为三分类的起始聚类中心
        // let center = []
        // let max = 0
        // for(let m=0; m<convectors.length; m++){
        //     for(let n=0; n<convectors.length; n++){
        //         for(let l=0; l<convectors.length; l++){
        //             let dis = tf.sqrt(tf.sum(tf.squaredDifference(tf.tensor(convectors[m]),tf.tensor(convectors[n])))).arraySync()
        //             + tf.sqrt(tf.sum(tf.squaredDifference(tf.tensor(convectors[m]),tf.tensor(convectors[l])))).arraySync()
        //             + tf.sqrt(tf.sum(tf.squaredDifference(tf.tensor(convectors[n]),tf.tensor(convectors[l])))).arraySync()
        //             center = dis>max ? [convectors[m],convectors[n],convectors[l]] : center
        //             max = dis>max ? dis : max
        //         }
        //     }
        // }

        //手动选择距离最远的两个点作为二分类的起始聚类中心
        let center = []
        let max = 0
        for(let m=0; m<convectors.length; m++){
            for(let n=0; n<convectors.length; n++){
                let dis = tf.sqrt(tf.sum(tf.squaredDifference(tf.tensor(convectors[m]),tf.tensor(convectors[n])))).arraySync()
                center = dis>max ? [convectors[m],convectors[n]] : center
                max = dis>max ? dis : max
            }
        }
        //测试平均卷积之后向量类内间距和类间间距的变化
        //结论：当卷积核大小不断加大的时候，类内距离与类间间距不断减小，类内距离的均值与类间距离的均值的比值在趋势上不断加大
        //所以在高卷积的前提下用聚类反而能更容易将类内和类间的向量分辨开
        // let sameSum = tf.scalar(0)
        // let differentSum = tf.scalar(0)
        // let firstClassSum = 3
        // for(let k=0; k<convectors.length; k++){
        //     if(k<firstClassSum){
        //         sameSum = tf.add(tf.sqrt(tf.sum(tf.squaredDifference(tf.tensor(convectors[0]),tf.tensor(convectors[k])))),sameSum)
        //     }else{
        //         differentSum = tf.add(tf.sqrt(tf.sum(tf.squaredDifference(tf.tensor(convectors[0]),tf.tensor(convectors[k])))),differentSum)
        //     }
        // }
        // sameSum = tf.div(sameSum,tf.scalar(firstClassSum-1))
        // differentSum = tf.div(differentSum,tf.scalar(convectors.length-firstClassSum))
        // console.log("value: ")
        // sameSum.print()
        // differentSum.print()
        // console.log("Convolution kernel: ",j," scale: ")
        // tf.div(differentSum,sameSum).print()
        //k-means聚类
        let ans = kmeans(convectors, 2, { initialization: center, maxIterations:1000})
        console.log("time:",convolutionSize,ans)
    //}//for xunhuan

    //验证分类结果
    let predictions = []
    for(let i=0; i<vectors.length; i++){
        for(let j=0; j<vectors[i].length; j++){
            predictions.push(ans.clusters[i])
        }
    }
    const options = {
        seed: 3,
        maxFeatures: 0.8,
        replacement: true,
        nEstimators: 100,
        useSampleBagging: true
    };
    let classifer = new RandomForest.RandomForestClassifier(options);
    // console.log(predictions)
    classifer.train(convectorsAll,predictions)
    // predict classifer 验证随机森林分类器
    const predictDatapath = './public/images/RandomForestPredictData/'
    let predictVectors = await loadFiles(predictDatapath, FacenetModel, mtcnnModel[0], mtcnnModel[1], mtcnnModel[2])
    // let vector = await facenet.faceVector(FacenetModel,"./public/images/TaylorSwift0004.jpg")
    // let vector1 = await facenet.faceVector(FacenetModel,"./public/images/TaylorSwift00.jpg")
    // let vector2 = await facenet.faceVector(FacenetModel,"./public/images/Aaron_Peirsol.jpg")
    // predictVectors = [vector.arraySync()[0],vector1.arraySync()[0],vector2.arraySync()[0]]
    predictConVectors = []
    for(subVectors of predictVectors){
        for(val of subVectors){
            let convector = []
            for(let i=0; i<val.length; i++){
                let EigenvalueSum = real.slice(i, i+convolutionSize).reduce(function (a, b) { return a + b;})
                let res = i+convolutionSize<=val.length ? val.slice(i, i+convolutionSize).reduce(function (a, b, index) { return a + b * real[i+index] }, 0)/EigenvalueSum : null
                if(res !== null){
                    convector.push(res)
                }    
            }
            predictConVectors.push(convector)
        }
    }
    console.log(predictConVectors.length,predictConVectors[0].length)
    let result = classifer.predict(predictConVectors);
    console.log(result)
    // }
}

createTree()