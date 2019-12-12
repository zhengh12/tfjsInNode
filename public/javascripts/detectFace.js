const tf = require("@tensorflow/tfjs-node");
const toolMatrix = require("./toolMatrix")
const images = require("images");
const fs = require("fs");

async function detectFace(imgarrs, threshold){
    //导入Pnet预训练模型
    const Pnet = await tf.loadLayersModel('file://./public/model/Pnet/model.json');
    const Rnet = await tf.loadLayersModel('file://./public/model/Rnet/model.json');
    const Onet = await tf.loadLayersModel('file://./public/model/Onet/model.json');
    let imgarr = imgarrs.arraySync()
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
        let imgTensors = tf.image.resizeNearestNeighbor(imgarrs,[hs,ws])
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
        rectangle.map(val=>{
            rectangles.push(val)
        })    
    })
    rectangles = toolMatrix.NMS(rectangles, 0.7, 'iou')
    //console.log('rectangles12',rectangles,rectangles.length)

    if (rectangles.length === 0){
        return rectangles
    }

    let crop_number = 0
    out = []
    let predict_24_batch = []
    //rectangles=[[153.0, 445.0, 170.0, 462.0, 0.6271253228187561],[189.0, 513.0, 206.0, 529.0, 0.6269974112510681]]
    rectangles.map(val=>{
        //console.log(caffe_img)
        let crop_img = caffe_img.map(vals=>{
            // console.log("val:",vals)
            // console.log("vals:",vals.slice(val[0],val[2]+1))
            return vals.slice(val[0],val[2])
        }).slice(val[1],val[3])
        // console.log(crop_img)
        // console.log(crop_img.length,crop_img[0].length)
        let input = tf.tensor(crop_img,[crop_img.length,crop_img[0].length,3])
        // console.log(input)
        let scale_img = tf.image.resizeNearestNeighbor(input,[24,24])
        //console.log("scale_img",scale_img)
        scale_img = scale_img.arraySync()
        //console.log("scale_imgs",scale_img[0][0])
        predict_24_batch.push(scale_img)
    })
    predict_24_batch = tf.tensor(predict_24_batch)
    out = Rnet.predict(predict_24_batch)
    // console.log(out,out.length)
    let cls_prob = out[0].arraySync()  
    let roi_prob = out[1].arraySync() 
    rectangles = toolMatrix.filter_face_24net(cls_prob, roi_prob, rectangles, origin_w, origin_h, threshold[1])

    //console.log('rectangles24',rectangles,rectangles.length)

    if (rectangles.length === 0){
        return rectangles
    }

    let predict_batch = []
    rectangles.map(val=>{
        //console.log(caffe_img)
        let crop_img = caffe_img.map(vals=>{
            // console.log("val:",vals)
            // console.log("vals:",vals.slice(val[0],val[2]+1))
            return vals.slice(val[0],val[2])
        }).slice(val[1],val[3])
        // console.log(crop_img)
        // console.log(crop_img.length,crop_img[0].length)
        let input = tf.tensor(crop_img,[crop_img.length,crop_img[0].length,3])
        // console.log(input)
        let scale_img = tf.image.resizeBilinear(input,[48,48])
        //console.log("scale_img",scale_img)
        scale_img = scale_img.arraySync()
        //console.log("scale_imgs",scale_img[0][0])
        predict_batch.push(scale_img)
    })
    predict_batch = tf.tensor(predict_batch)
    let output = Onet.predict(predict_batch)
    // console.log(output,output.length)
    cls_prob = output[0].arraySync()
    // console.log(cls_prob)
    roi_prob = output[1].arraySync()
    let pts_prob = output[2].arraySync()
    rectangles = toolMatrix.filter_face_48net(cls_prob, roi_prob, pts_prob, rectangles, origin_w, origin_h, threshold[2])
    return rectangles
}

module.exports=detectFace