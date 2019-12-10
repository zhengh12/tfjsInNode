var fs = require('fs');
const tf = require("@tensorflow/tfjs-node");
// let cls_prob = [[0,1],[2,3],[4,5],[6,7],[8,9],[10,11],[12,13],[14,15],[16,17],[18,19]]
// //let cls_prob1 = [[0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0]]
// //let cls_prob1 = new Array(cls_prob[0].length).fill(new Array(cls_prob.length).fill(99));
// let cls_prob1 = Array(cls_prob[0].length).fill(null).map((val,index1) => {
//     return Array(cls_prob.length).fill(null).map((val,index2)=>{
//         return cls_prob[index2][index1]
//     })
// })

// console.log(cls_prob1)

// let cls_probs = [[[1,2],[3,4],[5,6]],[[7,8],[9,10],[11,12]],[[13,14],[15,16],[17,18]]]

// let cls_probs1 = Array(cls_probs[0][0].length).fill(null).map((val,index0) => {
//     return Array(cls_probs[0].length).fill(null).map((val,index1)=>{
//         return Array(cls_probs.length).fill(null).map((val,index2)=>{
//             return cls_probs[index2][index1][index0]
//         })
//     })
// })

// cls_prob.map((val,index1)=>{
//     val.map((val,index2)=>{
//         console.log(val,index1,index2)
//         console.log(cls_prob1[index2][index1])
//         cls_prob1[index2][index1] = val
//         console.log(cls_prob1)
//     })
// })

//console.log([...''.padEnd(100)].map((v,i)=>i))

// const input1 = tf.input({shape: [2, 2]});
// const input2 = tf.input({shape: [2, 2]});
// const input3 = tf.input({shape: [2, 2]});
// const multiplyLayer = tf.layers.multiply();
// const product = multiplyLayer.apply([input1, new tf.SymbolicTensor({})]);
// console.log(product.shape);

let components = []
const files = fs.readdirSync('./public/images/')
files.forEach(function (item, index) {
    let stat = fs.lstatSync("./public/images/" + item)
    if (stat.isDirectory() === true) { 
      components.push(item)
    }
})
let res = components.map(item=>{
    let file = fs.readdirSync('./public/images/'+item+'/')
    return file.map(val=>{
        let stat = fs.lstatSync('./public/images/'+item+'/'+val)
        if(stat.isDirectory() === false){
            return val
        }
    })
})

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

console.log(components);
console.log(dirArr);

// console.log(tf.mean(tf.tensor([[[1,2,3],[1,2,3]],[[1,2,3],[1,2,3]]]), [0,1,2]).arraySync())
// console.log(tf.add(tf.tensor([1,2,3]),tf.mul(tf.tensor([1,2,3]),tf.scalar(0.1))).print())
// console.log(tf.sqrt(tf.sum(tf.squaredDifference(tf.tensor([1,1,1]),tf.tensor([2,2,2])))).arraySync())
// tf.sqrt(tf.sum(tf.squaredDifference(tf.tensor([[1,1,1]]),tf.tensor([[2,2,2]])))).print()