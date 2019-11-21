
const tf = require("@tensorflow/tfjs-node");
const fs = require("fs");
const gm = require("gm")
const images = require("images");

function readImgSync(path) {
    fs.readFileSync(path,'Uint8Array')
}

async function loadLayersModel(){
    // images("./public/images/goldfinch, Carduelis carduelis/123.jpg")
    // .resize(224,224)
    // .save("./public/images/goldfinch, Carduelis carduelis/123.png")
    //.resize(224,224).save("./1231.jpg")
    //let img = readImg("C:/Users/1/Desktop/tensorflowjs/tfjsNode/public/images/TaylorSwift/TaylorSwift1.png")
    //let img2 = fs.readFileSync("./1.png")
    let img = fs.readFileSync("C:/Users/1/Desktop/tensorflowjs/tfjsNode/tfjsInNode/public/images/goldfinch, Carduelis carduelis/123.png")
    //let img = fs.readFileSync("C:/Users/1/Desktop/tensorflowjs/tfjsNode/tfjsInNode/public/images/TaylorSwift/TaylorSwift1.png")
    //tfjs/mobilenet_v1_0.25_224/imagenet_class_names.json
    //let img = new Image("C:/Users/1/Desktop/tensorflowjs/tfjsNode/public/images/TaylorSwift/TaylorSwift1.png")
    //let u8 = new Uint8Array(img)
    //console.log(imgg)
    let imgTensor = tf.node.decodeImage(img)
    imgTensor.reshape([1,224,224,4])
    console.log(imgTensor)
    const model = await tf.loadLayersModel('file://C:/Users/1/Desktop/tensorflowjs/tfjsNode/tfjsInNode/public/model/minist-2019_11_21 下午2_21_14.json');
    //const model = tf.loadLayersModel('indexeddb://my-model-1');
    //const model = tf.loadGraphModel(fs);
    model.summary()
    model.predict(imgTensor)
    //model.loadWeights('file://C:/Users/1/Desktop/tensorflowjs/tfjsNode/tfjsInNode/public/model/ssd_mobilenetv1_model-weights_manifest.json')
    return model
}

let model1 = loadLayersModel()

