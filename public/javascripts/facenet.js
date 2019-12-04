const tf = require("@tensorflow/tfjs-node");
const fs = require("fs");
const images = require("images");
const detectFace = require("./detectFace")

async function main(){
    class Lambda extends tf.layers.Layer {
        //static className = 'Lambda';
        
        constructor() {
            super({});
        }
        
        // In this case, the output is a scalar.
        computeOutputShape(input) { return input[0]; }
    
        // call() is where we do the computation.
        //tf.layers.add().apply([input[0],input[1]])
        call(input,kwargs) { return input[0]}
        
        // Every layer needs a unique name.
        getClassName() { return 'Lambda'; }
           
    }
    Lambda.className = 'Lambda'
    
    // const input1 = tf.input({shape: [2, 2]});
    // const input2 = tf.input({shape: [2, 2]});
    // console.log(input1);
    // const sum = new Lambda().apply([input1, input2]);
    // console.log(JSON.stringify(sum.shape));
    // const a = tf.layers.add({batchInputShape:input1.shape}).apply([input1,sum])
    // console.log(a);
    tf.serialization.registerClass(Lambda);
    const model = await tf.loadLayersModel('file://./public/model/Facenet/model.json');
    //model.summary()
    let img = fs.readFileSync("./public/images/BillGates/BillGates0.png")
    let img1 = fs.readFileSync("./public/images/Tyler2.jpeg")
    let imgTensor = tf.node.decodeImage(img)
    let imgTensor1 = tf.node.decodeImage(img1)
    //imgTensor = tf.image.resizeNearestNeighbor(imgTensor,[160,160])
    let threshold = [0.6,0.6,0.7]
    let rectangles = await detectFace(imgTensor,threshold)
    console.log('rectangles:',rectangles[0])
    let rectangles1 = await detectFace(imgTensor1,threshold)
    console.log('rectangles:',rectangles1[0])

    imgTensor = imgTensor.arraySync()
    //console.log("imgTensor:",imgTensor)
    let imgTensorFace = []
    imgTensor.map((val,index)=>{
        if(index>rectangles[0][0]&&index<=rectangles[0][2]){
            imgTensorFace.push(val.slice(rectangles[0][1],rectangles[0][3]))
        }
    })

    imgTensor1 = imgTensor1.arraySync()
    let imgTensor1Face = []
    imgTensor1.map((val,index)=>{
        if(index>rectangles1[0][0]&&index<=rectangles1[0][2]){
            imgTensor1Face.push(val.slice(rectangles1[0][1],rectangles1[0][3]))
        }
    })
    prewhiten(tf.tensor([[[1,2,3],[1,2,3]],[[1,2,3],[1,2,3]]]))
    imgTensorFace = tf.image.resizeNearestNeighbor(tf.tensor(imgTensorFace,[rectangles[0][2]-rectangles[0][0],rectangles[0][3]-rectangles[0][1],3]),[160,160])
    imgTensorFace = prewhiten(imgTensorFace)
    let input = imgTensorFace.reshape([1,160,160,3])
    imgTensor1Face = tf.image.resizeNearestNeighbor(tf.tensor(imgTensor1Face,[rectangles1[0][2]-rectangles1[0][0],rectangles1[0][3]-rectangles1[0][1],3]),[160,160])
    imgTensor1Face = prewhiten(imgTensor1Face)
    let input1 = imgTensor1Face.reshape([1,160,160,3])
    let out = model.predict(input)
    let out1 = model.predict(input1)
    console.log("out:",out.arraySync())
    console.log("out1:",out1.arraySync())
    console.log(tf.sum(tf.squaredDifferenceStrict(out,out1)).arraySync())
    let dist = tf.sqrt(tf.sum(tf.squaredDifferenceStrict(out,out1)))
    console.log(dist.arraySync())
}

function prewhiten(x) {
    let axis = [0, 1, 2]
    let size = x.size
    let mean = tf.mean(x, axis)
    // let std = tf.square(tf.sub(x,mean))
    mean = mean.arraySync()
    x = x.arraySync()
    console.log('size',size)
    //console.log(x,mean,size)
    let sum = 0
    for(let i=0; i<x.length; i++){
        for(let j=0; j<x[0].length; j++){
            for(let k=0; k<x[0][0].length; k++){
                sum = sum + Math.pow(x[i][j][k] - mean,2)
            }
        }
    }
    let std = Math.sqrt(sum/size)
    console.log(std)
    //let std_adj = np.maximum(std, 1.0/np.sqrt(size))
    //let y = (x - mean) / std_adj
    let y = x.map(val=>{
        return val.map(val=>{
            return val.map(val=>{
                return (val-mean)/std
            })

        })
    })
    console.log("iamy:",y)
    y = tf.tensor(y)
    return y
}
main()