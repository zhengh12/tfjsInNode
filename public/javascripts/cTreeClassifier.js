const kmeans = require('ml-kmeans');
const facenet = require("./facenet")
const fs = require("fs");
const tf = require("@tensorflow/tfjs-node");
const RandomForest = require('ml-random-forest') 

async function createTree(){
    let trainDatapath = './public/images/RandomForestTrainData2/'
    let dirArr=[]
    let dir = fs.readdirSync(trainDatapath)
    dir.map(item=>{
        let stat = fs.lstatSync(trainDatapath + item)
        if (stat.isDirectory() === true) { 
            let subDirArr = []
            let subDir = fs.readdirSync(trainDatapath + item + '/')
            subDir.map(val=>{
                let stat = fs.lstatSync(trainDatapath + item + '/' + val)
                if(stat.isDirectory() === false){
                    subDirArr.push(trainDatapath + item + '/' + val)
                }
            })
            dirArr.push(subDirArr)
        }
    })
    
    // console.log(dirArr)
    const modelPath = "./public/model/Facenet1/model.json"
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
    let vectors = []
    for(let i=0; i<dirArr.length; i++){
        let subvector = []
        for(val of dirArr[i]){
            console.log("loading file: "+val)
            let vector = await facenet.faceVector(modelPath,val)
            vector = vector.arraySync()[0]
            vector.push(i)
            // vector = vector.map((val,index)=>{
            //     return vector.slice(0,index).concat(vector.slice(index+1,vector.length)).concat([index.toString()])
            // })
            subvector.push(vector)
        }
        vectors.push(subvector)
    }
    console.log("loading file over")
    //128维向量平均卷积后的维度为 128-convolutionSize+1
    //再对每一类取类间平均值
    //for(let j=2; j<=100; j=j+1){
        let convolutionSize = 40
        let convectors = []
        let convectorsAll = []
        for(subVectors of vectors){
            let avgVector = tf.scalar(0)
            for(val of subVectors){
                let convector = []
                for(let i=0; i<val.length-1; i++){
                    let res = i+convolutionSize<=val.length-1 ? val.slice(i, i+convolutionSize).reduce(function (a, b) { return a + b;})/convolutionSize : null
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
        // console.log("long:",convectors.length,convectors)
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
        // let firstClassSum = 4
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
        nEstimators: 25
    };
    let classifer = new RandomForest.RandomForestClassifier(options);
    console.log(predictions)
    classifer.train(convectorsAll,predictions)
    // classifer.train(convectors,ans.clusters)
    let vector = await facenet.faceVector(modelPath,"./public/images/TaylorSwift0004.jpg")
    let vector1 = await facenet.faceVector(modelPath,"./public/images/TaylorSwift00.jpg")
    let vector2 = await facenet.faceVector(modelPath,"./public/images/Aaron_Peirsol.jpg")
    predictVectors = [vector.arraySync()[0],vector1.arraySync()[0],vector2.arraySync()[0]]
    predictConVectors = []
    for(val of predictVectors){
        let convector = []
        for(let i=0; i<val.length; i++){
            let res = i+convolutionSize<=val.length ? val.slice(i, i+convolutionSize).reduce(function (a, b) { return a + b;})/convolutionSize : null
            if(res !== null){
                convector.push(res)
            }    
        }
        predictConVectors.push(convector)
    }
    console.log(predictConVectors.length,predictConVectors[0].length)
    let result = classifer.predict(predictConVectors);
    console.log(result)
}

createTree()