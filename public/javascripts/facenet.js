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
        computeOutputShape(input) {return input; }
    
        // call() is where we do the computation.
        //tf.layers.add().apply([input[0],input[1]])
        call(input,kwargs) {
            console.log('input',input)
            // console.log('input[1]',input[1])
            // let out = tf.mul(input[1],tf.scalar(0.06))
            // let out1 = tf.mul(input[0],tf.scalar(0.97))
            return [tf.mul(input[0],tf.scalar(1))]
        } 
        
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
    const model = await tf.loadLayersModel('file://./public/model/Facenet1/model.json');
    model.summary()
    let img = fs.readFileSync("./public/images/LarryPage/Larry_Page_0000.jpg")
    let img1 = fs.readFileSync("./public/images/LarryPage/Larry_Page_0002.jpg")
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
    // imgTensor1Face = imgTensor1Face.map(val=>{
    //     return val.map(val=>{
    //         let array = val.map(val=>{
    //             return val
    //         })
    //         array.splice(2,1,...array.splice(0, 1 , array[2]));
    //         return array
    //     })
    // })

    prewhiten(tf.tensor([[[1,2,3],[1,2,3]],[[1,2,3],[1,2,3]]]))
    imgTensorFace = tf.image.resizeNearestNeighbor(tf.tensor(imgTensorFace,[rectangles[0][2]-rectangles[0][0],rectangles[0][3]-rectangles[0][1],3]),[160,160])
    imgTensorFace = prewhiten(imgTensorFace)
    let input = imgTensorFace.reshape([1,160,160,3])
    imgTensor1Face = tf.image.resizeNearestNeighbor(tf.tensor(imgTensor1Face,[rectangles1[0][2]-rectangles1[0][0],rectangles1[0][3]-rectangles1[0][1],3]),[160,160])
    imgTensor1Face = prewhiten(imgTensor1Face)
    let input1 = imgTensor1Face.reshape([1,160,160,3])
    let out = model.predict(input)
    let out1 = model.predict(input1)
    let momo = tf.zeros([1,160,160,3])
    console.log("yaya",momo.shape)
    let out2 = model.predict(tf.zeros([1,160,160,3]))
    // console.log("out:",out.arraySync())
    // console.log("out1:",out1.arraySync())
    let out3 = tf.tensor([-0.16026781 ,-0.43058458, -0.08495475, -0.45368737,  0.37071365,  0.11544002
        ,-0.03815316 ,-0.19679411 ,-0.03364312  ,0.19351539,  0.2300798  ,-0.2994608
        , 0.14851429 ,-0.310086   ,-0.09590907  ,0.44437397,  0.32496366 ,-0.3346117
        , 0.26422447 ,-0.53023154 , 0.1836549   ,0.15576914, -0.16831647 ,-0.01893916
        , 0.23467176 ,-0.09293256 , 0.49017134  ,0.44747016, -0.12255521 ,-0.02273713
        ,-0.23011264 , 0.39446694 , 0.01850758  ,0.04234677  ,0.5098182  ,-0.44884047
        , 0.17330451 , 0.11631436 , 0.48783356  ,0.10806299  ,0.23611988 ,-0.00526098
        ,-0.0973383  ,-0.17481872 ,-0.43994665 ,-0.53889674  ,0.02620845 ,-0.20854124
        , 0.5323282  , 0.26953638 ,-0.49680495  ,0.3034832   ,0.21464384 ,-0.428988
        , 0.11885318 , 0.12331277 , 0.52949244  ,0.05367832 ,-0.3322937  ,-0.90414447
        ,-0.44209328 ,-0.35821617 ,-0.05992308  ,0.7086487  ,-0.32271174 , 0.48468256
        , 0.30310646 , 0.06076867 , 0.28565273 ,-0.46813235 ,-0.08829965 ,-0.42347565
        , 0.19438231 ,-0.42049563 ,-0.04389277 ,-0.09989235 , 0.02166604 , 0.336157
        ,-0.22772345 ,-0.294679   ,-0.5950171  ,-0.18462925 , 0.17137341 , 0.26968244
        , 0.4168208  , 0.33437648 , 0.4301593  ,-0.19249184 ,-0.62138295 , 0.86591107
        , 0.40483132 , 0.0497229  ,-0.02391302 , 0.03882176 , 0.78918195 , 0.20468558
        ,-0.03643258 ,-0.26830706 ,-0.75587875 ,-0.10048974 ,-0.22140677 , 0.5766636
        , 0.18364036 ,-0.3377278  , 0.00320715 , 0.14133668 ,-0.16474539 , 0.06373426
        ,-0.34667358 ,-0.41627762 , 0.22354744 , 0.198507   , 0.30029547 , 0.434904
        , 0.02097284 , 0.11842813 , 0.16883281 , 0.34432313 , 0.19626537 ,-0.05554222
        ,-0.16246805 , 0.07040231 , 0.30010045 , 0.04093454 , 0.2814713  ,-0.5083593
        , 0.18601538 , 0.18163924])
    let out3 = tf.tensor([-0.0990261734, -0.384425759, -0.147961706, -0.447421163,
            0.386348188, -0.00956525374,  0.0753497183, -0.260145128,
           -0.0118370894,  0.171039969,  0.243021518, -0.271708280,
            0.141100898, -0.303799450, -0.0388800651,  0.395115018,
            0.372875512, -3.79378140e-01  2.70814985e-01 -5.03493130e-01
            9.84710008e-02  7.79178217e-02 -2.34342992e-01 -7.56116882e-02
            1.64244339e-01 -1.02743737e-01  5.11597872e-01  4.34037417e-01
           -1.62378564e-01 -3.17336321e-02 -2.27899671e-01  3.53947818e-01
            5.56578115e-02 -4.60415296e-02  4.70759928e-01 -3.80175263e-01
            1.69982612e-01  5.50013632e-02  4.77226317e-01  1.10341378e-01
            2.45424256e-01  3.00409682e-02 -1.38272166e-01 -1.12411454e-01
           -3.93686712e-01 -5.93100309e-01 -7.13718235e-02 -1.18872136e-01
            4.99553293e-01  2.70739734e-01 -4.64011729e-01  2.72511750e-01
            6.95701689e-02 -4.05129611e-01  6.08364344e-02  1.65223628e-01
            6.17549896e-01  2.01887101e-01 -2.85607725e-01 -9.62759435e-01
           -4.82098907e-01 -3.55632246e-01 -2.28872187e-02  7.52430439e-01
           -2.73528099e-01  5.24031699e-01  2.82180935e-01 -1.37776220e-02
            2.88436979e-01 -4.68228787e-01 -1.35357141e-01 -4.56420481e-01
            2.65181988e-01 -4.09849644e-01 -1.35110125e-01 -1.42948523e-01
            2.73902565e-02  2.33400896e-01 -2.63375223e-01 -3.61697972e-01
           -5.92225254e-01 -9.62535813e-02  1.86938360e-01  2.02825263e-01
            3.52257669e-01  1.84106529e-01  3.95842850e-01 -1.59143940e-01
           -5.83384395e-01  9.10694540e-01  3.41903657e-01  3.18180025e-02
            1.94635242e-04  3.35271657e-02  6.06194079e-01  1.39420047e-01
           -1.74950752e-02 -3.31170559e-01 -7.96942890e-01 -2.80604139e-02
           -2.88735747e-01  6.00302219e-01  2.22013414e-01 -3.24921250e-01
            7.47855082e-02  1.34987131e-01 -5.45142069e-02  8.80107582e-02
           -2.18519181e-01 -3.79522026e-01  2.20661119e-01  2.17258021e-01
            3.37399393e-01  5.04880548e-01  4.95145172e-02  1.20242178e-01
            2.54263908e-01  3.44235003e-01  2.82776058e-01 -5.04020080e-02
           -1.63545832e-01  1.61966667e-01  3.13006699e-01  8.18568654e-03
            2.84631103e-01 -4.88040656e-01  1.87270194e-01  7.46491104e-02])
    // let dist = tf.div(tf.sqrt(tf.sum(tf.squaredDifference(out1,out))),tf.scalar(3))
    let dist = tf.sqrt(tf.sum(tf.squaredDifference(out,out1)))
    tf.squaredDifference(out2,out3).print()
    dist.print()
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
    y = tf.tensor(y)
    return y
}
main()