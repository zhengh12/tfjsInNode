
const tf = require("@tensorflow/tfjs-node");
const fs = require("fs");
const images = require("images");
const detectFace = require("./detectFace")
const inceptionResNetV2 = require("./inceptionResNetV2")

function readImgSync(path) {
    fs.readFileSync(path,'Uint8Array')
}

async function loadLayersModel(){
    // images("./public/images/goldfinch, Carduelis carduelis/hjhj.jpg")
    // .resize(224,224)
    // .save("./public/images/goldfinch, Carduelis carduelis/hjhj.jpeg")
    //let img = readImg("C:/Users/1/Desktop/tensorflowjs/tfjsNode/public/images/TaylorSwift/TaylorSwift1.png")
    //let img2 = fs.readFileSync("./1.png")
    let img = fs.readFileSync("./public/images/Tyler.jpeg")
    //let img = fs.readFileSync("C:/Users/1/Desktop/tensorflowjs/tfjsNode/tfjsInNode/public/images/TaylorSwift/TaylorSwift1.png")
    //tfjs/mobilenet_v1_0.25_224/imagenet_class_names.json
    //let img = new Image("C:/Users/1/Desktop/tensorflowjs/tfjsNode/public/images/TaylorSwift/TaylorSwift1.png")
    //let u8 = new Uint8Array(img)
    //console.log(imgg)
    let imgTensors = tf.node.decodeImage(img)
    //let imgTensors = imgTensor.reshape([1,224,224,3])
    let imgarr = await imgTensors.array()

    let threshold = [0.6,0.6,0.7]

    // let rectangles = await detectFace(imgarr,threshold)
    // console.log(rectangles)
    // let image = images("./public/images/Tyler.jpeg")
    // rectangles.map(val=>{
    //     let x1 = val[0]
    //     let y1 = val[1]
    //     let x2 = val[2]
    //     let y2 = val[3]
    //     let imgborder = 5
    //     console.log(x1,y1,x2,y2)
    //     image.draw(images(x2-x1, imgborder).fill(127, 255, 170, 0.7),x1,y1)
    //     .draw(images(imgborder, y2-y1).fill(127, 255, 170, 0.7),x1,y1)
    //     .draw(images(x2-x1, imgborder).fill(127, 255, 170, 0.7),x1,y2-imgborder)
    //     .draw(images(imgborder, y2-y1).fill(127, 255, 170, 0.7),x2-imgborder,y1)
    // })
    // image.save("./public/Tyler.jpeg")

    // const model = await tf.loadLayersModel('file://./public/model/model.json');

    // let model = inceptionResNetV2.create_inception_resnet_v2()
    // model.summary()
    class Lambda extends tf.layers.Layer {
        //static className = 'Lambda';
        
        constructor() {
            super({});
        }
        
        // In this case, the output is a scalar.
        computeOutputShape(input) { return input[0]; }

        // call() is where we do the computation.
        call(input,kwargs) { return input[1];}
        
        // Every layer needs a unique name.
        getClassName() { return 'Lambda'; }
           
    }
    Lambda.className = 'Lambda'
    let models = tf.sequential()

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
    img = fs.readFileSync("C:/Users/1/Desktop/tensorflowjs/tfjsNode/tfjsInNode/public/images/TaylorSwift1.png")
    img1 = fs.readFileSync("C:/Users/1/Desktop/tensorflowjs/tfjsNode/tfjsInNode/public/images/TaylorSwift2.png")
    let imgTensor = tf.node.decodeImage(img)
    let imgTensor1 = tf.node.decodeImage(img1)
    //imgTensor = tf.image.resizeNearestNeighbor(imgTensor,[160,160])
    imgTensor = imgTensor.arraySync()
    imgTensor1 = imgTensor1.arraySync()
    // imgTensor = imgTensor.map(val=>{
    //     return val.map(val=>{
    //         let array = val.map(val=>{
    //             val = val - 127.5
    //             val = val / 127.5
    //             return val
    //         })
    //         array.splice(2,1,...array.splice(0, 1 , array[2]));
    //         return array
    //     })
    // })
    // imgTensor1 = imgTensor1.map(val=>{
    //     return val.map(val=>{
    //         let array = val.map(val=>{
    //             val = val - 127.5
    //             val = val / 127.5
    //             return val
    //         })
    //         array.splice(2,1,...array.splice(0, 1 , array[2]));
    //         return array
    //     })
    // })
    // console.log(imgTensor1)
    let input = tf.tensor(imgTensor,[160,160,3]).reshape([1,160,160,3])
    let input1 = tf.tensor(imgTensor1,[160,160,3]).reshape([1,160,160,3])
    let out = model.predict(input)
    let out1 = model.predict(input1)
    console.log(out1)
    let dist = tf.sqrt(tf.sum(tf.square(out,out1)))
    console.log(dist.arraySync())
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

