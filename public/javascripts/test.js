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

console.log(tf.mean(tf.tensor([[[1,2,3],[1,2,3]],[[1,2,3],[1,2,3]]]), [0,1,2]).arraySync())