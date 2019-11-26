
const tf = require("@tensorflow/tfjs-node");
const fs = require("fs");
const gm = require("gm")
const images = require("images");
const detectFace = require("./detectFace")

function readImgSync(path) {
    fs.readFileSync(path,'Uint8Array')
}

async function loadLayersModel(){
    // images("./public/images/goldfinch, Carduelis carduelis/hjhj.jpg")
    // .resize(224,224)
    // .save("./public/images/goldfinch, Carduelis carduelis/hjhj.jpeg")
    //let img = readImg("C:/Users/1/Desktop/tensorflowjs/tfjsNode/public/images/TaylorSwift/TaylorSwift1.png")
    //let img2 = fs.readFileSync("./1.png")
    let img = fs.readFileSync("C:/Users/1/Desktop/tensorflowjs/tfjsNode/tfjsInNode/public/images/Tyler.jpeg")
    //let img = fs.readFileSync("C:/Users/1/Desktop/tensorflowjs/tfjsNode/tfjsInNode/public/images/TaylorSwift/TaylorSwift1.png")
    //tfjs/mobilenet_v1_0.25_224/imagenet_class_names.json
    //let img = new Image("C:/Users/1/Desktop/tensorflowjs/tfjsNode/public/images/TaylorSwift/TaylorSwift1.png")
    //let u8 = new Uint8Array(img)
    //console.log(imgg)
    let imgTensors = tf.node.decodeImage(img)
    //let imgTensors = imgTensor.reshape([1,224,224,3])
    let imgarr = await imgTensors.array()

    let threshold = [0.6,0.6,0.7]
    await detectFace(imgarr,threshold)

    const model = await tf.loadLayersModel('file://C:/Users/1/Desktop/tensorflowjs/tfjsNode/tfjsInNode/public/model/Pnet/model.json');
    // model.summary()
    // let arr = await model.predict(imgTensors).array()
    // let max = 0
    // let maxflag = 0
    // for(let i=0; i<arr[0].length; i++){
    //     if (max < arr[0][i]){
    //         max = arr[0][i]
    //         maxflag = i
    //     }
    // }
    // console.log(maxflag)
    //model.loadWeights('file://C:/Users/1/Desktop/tensorflowjs/tfjsNode/tfjsInNode/public/model/ssd_mobilenetv1_model-weights_manifest.json')
    return model
}

let model1 = loadLayersModel()

