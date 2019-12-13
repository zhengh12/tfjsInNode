const kmeans = require('ml-kmeans');
const facenet = require("./facenet")
const fs = require("fs");
const tf = require("@tensorflow/tfjs-node");
const RandomForest = require('ml-random-forest') 

async function createTree(){
    let trainDatapath = './public/images/RandomForestTrainData1/'
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
        for(val of dirArr[i]){
            console.log("loading file: "+val)
            let vector = await facenet.faceVector(modelPath,val)
            vector = vector.arraySync()[0]
            vector.push(i)
            // vector = vector.map((val,index)=>{
            //     return vector.slice(0,index).concat(vector.slice(index+1,vector.length)).concat([index.toString()])
            // })
            vectors.push(vector)
        }
    }
    console.log("loading file over")
    //128维向量平均卷积后的维度为 128-convolutionSize+1
    // for(let j=1; j<=20; j=j+1){
        let convolutionSize = 2
        let convectors = []
        for(val of vectors){
            let convector = []
            for(let i=0; i<val.length-1; i++){
                let res = i+convolutionSize<=val.length-1 ? val.slice(i, i+convolutionSize).reduce(function (a, b) { return a + b;})/convolutionSize: null
                if(res !== null){
                    convector.push(res)
                }    
            }
            // let out = tf.tensor(val)
            // let out1 = tf.tensor([...convector,...val.slice(val.length-convolutionSize,val.length)])
            // let dist = tf.sqrt(tf.sum(tf.squaredDifference(out,out1)))
            // dist.print()
            convectors.push(convector)
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
        // let ans = kmeans(convectors, 2, { initialization: 'kmeans++' })
        // console.log("time:",convolutionSize,ans)
    // }
    const options = {
        seed: 3,
        maxFeatures: 0.8,
        replacement: true,
        nEstimators: 25
    };
    let classifer = new RandomForest.RandomForestClassifier(options);
    let predictions = [ 1,
        1,
        1,
        1,
        0,
        0,
        0,
        0,
        0,
        0,
        1,
        1,
        1,
        1,
        1,
        1,
        0,
        0,
        0,
        0,
        0,
        0,
        1,
        1,
        1,
        0,
        0,
        0,
        1,
        1,
        1,
        0,
        0,
        0,
        0,
        0,
        0,
        1,
        1,
        1 ]
    classifer.train(convectors,predictions)
    let vector = await facenet.faceVector(modelPath,"./public/images/TaylorSwift00.jpg")
    let vector1 = await facenet.faceVector(modelPath,"./public/images/TaylorSwift01.jpeg")
    let result = classifer.predict([vector.arraySync()[0],vector1.arraySync()[0]]);
    console.log(result)
}

createTree()