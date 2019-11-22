const tf = require("@tensorflow/tfjs-node");
const calculateScales = require("./toolMatrix")
const images = require("images");
const fs = require("fs");

async function detectFace(imgarr, threshold){
    let caffe_img = imgarr.map(val=>{
        return val.map(val=>{
            return val.map(val=>{
                val = val - 127.5
                val = val / 127.5
                return val
            })
        })
    })
    const origin_h = caffe_img.length
    const origin_w = caffe_img[0].length
    const ch = caffe_img[0][0].length
    let scales = calculateScales(imgarr)//获得
    console.log(origin_h,origin_w)
    //console.log(scales)
    //console.log(caffe_img[100][100])
    let out = []
    // del scales[:4]

    // 将图片转化为图形金字塔输入PNet中
    scales.map(async scale=>{
        let hs = Math.floor(origin_h * scale)
        let ws = Math.floor(origin_w * scale)
        images("./public/images/Tyler.jpeg")
        .resize(ws,hs).save("./public/Tyler.jpeg")
        let img = fs.readFileSync("C:/Users/1/Desktop/tensorflowjs/tfjsNode/tfjsInNode/public/Tyler.jpeg")
        let imgTensors = tf.node.decodeImage(img)
        let scale_img = await imgTensors.array()
        let scale_imgs = scale_img.map(val=>{
            return val.map(val=>{
                return val.map(val=>{
                    val = val - 127.5
                    val = val / 127.5
                    return val
                })
            })
        })
        console.log(scale_imgs.length,scale_imgs[0].length)
        console.log(scale_imgs[10][10])
        let input = tf.tensor(scale_imgs,[hs,ws,3]).reshape([1,hs,ws,3])
        //let ouput = Pnet.predict(input)
        //out.push(ouput)
    })

    // let arrs = [[[0,1],[2,3],[4,5],[6,7],[8,9],[10,11],[12,13],[14,15],[16,17],[18,19]],[[0,1],[2,3],[4,5],[6,7],[8,9],[10,11],[12,13],[14,15],[16,17],[18,19]]]
    // arrs = arrs.map(val=>{
    //     return val.map(val=>{
    //         return val[1]
    //     })
    // })
    // console.log(arrs)

    let rectangles = []
    out.map(val=>{
        let cls_prob = val[0][0].map(val=>{
            return val.map(val=>{
                return val[1]
            })
        })
        let roi = val[1][0]
        let out_w = cls_prob.length
        let out_h = cls_prob[0].length
    })
}

module.exports=detectFace