const kmeans = require('ml-kmeans');
const facenet = require("./facenet")
const fs = require("fs");
const tf = require("@tensorflow/tfjs-node");

async function createTree(){
    let trainDatapath = './public/images/RandomForestTrainData/'
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
    
    console.log(dirArr)
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
    //128维向量卷积后的维度为 128-convolutionSize+1
    for(let j=2; j<=40; j=j+2){
        let convolutionSize = 64
        let convectors = []
        for(val of vectors){
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
            convectors.push(convector)
        }
        let ans = kmeans(convectors, 2, { initialization: 'kmeans++' })
        console.log("time:",convolutionSize,ans)
    }
}

createTree()