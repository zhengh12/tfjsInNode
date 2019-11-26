const tf = require("@tensorflow/tfjs-node");
const toolMatrix = require("./toolMatrix")
const images = require("images");
const fs = require("fs");

async function detectFace(imgarr, threshold){
    //导入Pnet预训练模型
    const Pnet = await tf.loadLayersModel('file://C:/Users/1/Desktop/tensorflowjs/tfjsNode/tfjsInNode/public/model/Pnet/model.json');
    
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
    let scales = toolMatrix.calculateScales(imgarr)//获得
    // console.log(origin_h,origin_w)
    //console.log(scales)
    //console.log(caffe_img[100][100])
    let out = []
    // del scales[:4]

    // 将图片转化为图形金字塔输入PNet中
    scales.map(scale=>{
        let hs = Math.floor(origin_h * scale)
        let ws = Math.floor(origin_w * scale)

        // let imgbf = fs.readFileSync("C:/Users/1/Desktop/tensorflowjs/tfjsNode/tfjsInNode/public/images/Tyler.jpeg")
        // let imgTensor = tf.node.decodeImage(imgbf)
        // let imgTensorss = imgTensor.reshape([1,origin_h,origin_w,3])
        // let ouputs = Pnet.predict(imgTensorss)
        // let outputs = await ouputs.map( val=>{
        //     return val.arraySync()
        // })
        // console.log(outputs[0][0][0][0])

        // images("./public/images/Tyler.jpeg")
        // .resize((ws,hs)).save("./public/Tyler.jpeg")
        let img = fs.readFileSync("C:/Users/1/Desktop/tensorflowjs/tfjsNode/tfjsInNode/public/images/Tyler.jpeg")
        let imgTensor = tf.node.decodeImage(img)
        let imgTensors = tf.image.resizeBilinear(imgTensor,[hs,ws])
        let scale_img = imgTensors.arraySync()
        let scale_imgs = scale_img.map(val=>{
            return val.map(val=>{
                let array = val.map(val=>{
                    val = val - 127.5
                    val = val / 127.5
                    return val
                })
                array.splice(2,1,...array.splice(0, 1 , array[2]));
                return array
            })
        })
        //console.log(scale_imgs.length,scale_imgs[0].length)
        // console.log(scale.toString()+":",scale_imgs[10][10])
        let input = tf.tensor(scale_imgs,[hs,ws,3]).reshape([1,hs,ws,3])

        let ouput = Pnet.predict(input)
        let output = ouput.map( val=>{
            return val.arraySync()
        })
        //console.log(scale.toString()+":",output[0][0][0][0])
        out.push(output)
        // console.log(out)
    })

    //console.log(out)

    // let arrs = [[[0,1],[2,3],[4,5],[6,7],[8,9],[10,11],[12,13],[14,15],[16,17],[18,19]],[[0,1],[2,3],[4,5],[6,7],[8,9],[10,11],[12,13],[14,15],[16,17],[18,19]]]
    // arrs = arrs.map(val=>{
    //     return val.map(val=>{
    //         return val[1]
    //     })
    // })
    // console.log(arrs)

    let rectangles = []
    out.map((val,index)=>{
        let cls_prob = val[0][0].map(val=>{
            return val.map(val=>{
                return val[1]
            })
        })
        let roi = val[1][0]
        let out_h = cls_prob.length
        let out_w = cls_prob[0].length
        let out_side = Math.max(out_h, out_w)
        //cls_prob = np.swapaxes(cls_prob, 0, 1)
        //roi = np.swapaxes(roi, 0, 2)
        cls_prob = Array(cls_prob[0].length).fill(null).map((val,index1) => {
            return Array(cls_prob.length).fill(null).map((val,index2)=>{
                return cls_prob[index2][index1]
            })
        })//数组按0,1轴转置
        roi = Array(roi[0][0].length).fill(null).map((val,index0) => {
            return Array(roi[0].length).fill(null).map((val,index1)=>{
                return Array(roi.length).fill(null).map((val,index2)=>{
                    return roi[index2][index1][index0]
                })
            })
        })//数组按0,2轴转置
        //console.log(cls_prob.length,cls_prob[0].length)
        //console.log(roi.length,roi[0].length,roi[0][0].length)
        let rectangle = toolMatrix.detect_face_12net(cls_prob, roi, out_side, 1 / scales[index], origin_w, origin_h, threshold[0])
        // rectangle.map(val=>{
        //     rectangles.push(val)
        // })    
    })
    // rectangles = tools.NMS(rectangles, 0.7, 'iou')
}

module.exports=detectFace