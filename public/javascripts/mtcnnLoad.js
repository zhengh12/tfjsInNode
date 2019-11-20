
const tf = require("@tensorflow/tfjs-node");
const fs = require("fs");

function readImgSync(path) {
    fs.readFileSync(path,'Uint8Array')
}

async function loadLayersModel(){
    //let img = readImg("C:/Users/1/Desktop/tensorflowjs/tfjsNode/public/images/TaylorSwift/TaylorSwift1.png")
    let img = fs.readFileSync("C:/Users/1/Desktop/tensorflowjs/tfjsNode/public/images/TaylorSwift/TaylorSwift1.png")
    //tfjs/mobilenet_v1_0.25_224/imagenet_class_names.json
    //let img = new Image("C:/Users/1/Desktop/tensorflowjs/tfjsNode/public/images/TaylorSwift/TaylorSwift1.png")
    //let u8 = new Uint8Array(img)
    let imgTensor = tf.node.decodeImage(img)
    console.log(imgTensor)
    const model = await tf.loadLayersModel('file://C:/Users/1/Desktop/tensorflowjs/tfjsNode/public/model/minist.json');
    //const model = tf.loadLayersModel('indexeddb://my-model-1');
    //const model = tf.loadGraphModel(fs);
    model.summary()
    console.log("hihi2")
    return model
}

let model1 = loadLayersModel()

