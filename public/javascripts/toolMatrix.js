
function calculateScales(img){
    let caffe_img = img 
    let pr_scale = 1.0
    let h = caffe_img.length
    let w = caffe_img[0].length
    let ch = caffe_img[0][0].length
    if (Math.min(w,h)>500){
        pr_scale = 500.0/Math.min(h,w)
        w = w*pr_scale
        h = h*pr_scale
    }
    else if (Math.max(w,h)<500){
        pr_scale = 500.0/Math.max(h,w)
        w = w*pr_scale
        h = h*pr_scale
    }
    //multi-scale
    let scales = []
    let factor = 0.709
    let factor_count = 0
    let minl = Math.min(h,w)
    while (minl >= 12){
        scales.push(pr_scale*Math.pow(factor, factor_count))
        minl *= factor
        factor_count += 1
    }
    return scales
}

function detect_face_12net(cls_prob,roi,out_side,scale,width,height,threshold){
    let in_side = 2*out_side+11
    let stride = 0
    if(out_side != 1){
        stride = (in_side-12)/(out_side-1)
    }
    //(x,y) = np.where(cls_prob>=threshold)
    //////////////////////////////
    let x = []
    let y = []
    cls_prob.map((val,index0)=>{
        return val.map((val,index1)=>{
            if(val>=threshold){
                x.push(index0)
                y.push(index1)
            }
        }) 
    })

    console.log(x.length,y.length)
    let boundingbox = [x,y]
    boundingbox = Array(boundingbox[0].length).fill(null).map((val,index1) => {
        return Array(boundingbox.length).fill(null).map((val,index2)=>{
            return boundingbox[index2][index1]
        })
    })
    //console.log(boundingbox,boundingbox.length)
    // boundingbox = np.array([x,y]).T
    // bb1 = np.fix((stride * (boundingbox) + 0 ) * scale)
    let bb1 = boundingbox.map(val=>{
        return val.map(val=>{
            return Math.floor((val * stride + 0) * scale/1.0)
        })
    })
    //console.log(bb1,bb1.length)
    // bb2 = np.fix((stride * (boundingbox) + 11) * scale)
    let bb2 = boundingbox.map(val=>{
        return val.map(val=>{
            return Math.floor((val * stride + 11) * scale/1.0)
        })
    })
    //console.log(bb2,bb2.length)
    // boundingbox = np.concatenate((bb1,bb2),axis = 1)
    let arr1 = []
    bb1.map((val,index0)=>{
        let arr2 = []
        val.map((val,index1)=>{
            arr2.splice(index1,0,val) //保证是以x1,y1,x2,y2顺序
            arr2.push(bb2[index0][index1])
        })
        arr1.push(arr2)
    })
    boundingbox = arr1
    //console.log(boundingbox,boundingbox.length)
    // dx1 = roi[0][x,y]
    let dx1 = x.map((val,index)=>{
        return roi[0][val][y[index]]
    })
    //console.log("cls_prob",cls_prob)
    //console.log(x,y)
    //console.log(dx1,dx1.length)
    // dx2 = roi[1][x,y]
    let dx2 = x.map((val,index)=>{
        return roi[1][val][y[index]]
    })
    // dx3 = roi[2][x,y]
    let dx3 = x.map((val,index)=>{
        return roi[2][val][y[index]]
    })
    // dx4 = roi[3][x,y]
    let dx4 = x.map((val,index)=>{
        return roi[3][val][y[index]]
    })
    // score = np.array([cls_prob[x,y]]).T
    let score = x.map((val,index)=>{
        return [cls_prob[val][y[index]]]
    })
    //console.log(score)
    // offset = np.array([dx1,dx2,dx3,dx4]).T
    // console.log("dx1:",dx1)
    // console.log("dx2:",dx2)
    // console.log("dx3:",dx3)
    // console.log("dx4:",dx4)
    let offset = dx1.map((val,index)=>{
        return [val,dx2[index],dx3[index],dx4[index]]
    })
    //console.log(offset)
    // boundingbox = boundingbox + offset*12.0*scale
    boundingbox = boundingbox.map((val,index0)=>{
        return val.map((val,index1)=>{
            return val + offset[index0][index1]*12.0*scale
        })
    })
    //console.log("boundingbox",boundingbox)
    // rectangles = np.concatenate((boundingbox,score),axis=1)
    let rectangles = boundingbox.map((val,index)=>{
        return val.concat(score[index])
    })
    //console.log("rectangles:",rectangles)
    rectangles = rect2square(rectangles)
    // let rectangless = rect2square([[1,4,2,6,0.3],[1,4,2,7,0.2],[1,5,2,6,0.1]]) //测试数据
    console.log("rectangles:",rectangles)
    let pick = []
    // for i in range(len(rectangles)){
    //     x1 = int(max(0     ,rectangles[i][0]))
    //     y1 = int(max(0     ,rectangles[i][1]))
    //     x2 = int(min(width ,rectangles[i][2]))
    //     y2 = int(min(height,rectangles[i][3]))
    //     sc = rectangles[i][4]
    //     if x2>x1 and y2>y1:
    //         pick.append([x1,y1,x2,y2,sc])
    // }
    rectangles.map(val=>{
        let x1 = Math.floor(Math.max(0,val[0]))
        let y1 = Math.floor(Math.max(0,val[1]))
        let x2 = Math.floor(Math.min(width,val[2]))
        let y2 = Math.floor(Math.min(height,val[3]))
        let sc = val[4]
        if(x2>x1 && y2>y1){
            pick.push([x1,y1,x2,y2,sc])
        }
    })
    console.log(pick)
    return NMS(pick,0.3,'iou')
}

function rect2square(rectangles){
    //w = rectangles[:,2] - rectangles[:,0]
    let w = rectangles.map(val=>{
        return val[2]-val[0]
    })
    //h = rectangles[:,3] - rectangles[:,1]
    let h = rectangles.map(val=>{
        return val[3]-val[1]
    })
    //l = np.maximum(w,h).T
    let l = w.map((val,index)=>{
        return val>h[index] ? val : h[index] //两个之中取最大值
    })
    console.log("l:",l)
    //rectangles[:,0] = rectangles[:,0] + w*0.5 - l*0.5
    // rectangles[:,1] = rectangles[:,1] + h*0.5 - l*0.5 
    // rectangles[:,2:4] = rectangles[:,0:2] + np.repeat([l], 2, axis = 0).T 
    rectangles = rectangles.map((val,index)=>{
        let res = []
        res.push(val[0] + w[index]*0.5 - l[index]*0.5)
        res.push(val[1] + h[index]*0.5 - l[index]*0.5)
        res.push(val[0] + w[index]*0.5 - l[index]*0.5 + l[index])
        res.push(val[1] + h[index]*0.5 - l[index]*0.5 + l[index])
        res.push(val[4])
        return res
    })
    return rectangles
}

function NMS(rectangles,threshold,type){
    if(rectangles.length==0){
        return rectangles
    }
    let boxes = rectangles
    //x1 = boxes[:,0]
    let x1 = boxes.map(val=>{return val[0]})
    let y1 = boxes.map(val=>{return val[1]})
    let x2 = boxes.map(val=>{return val[2]})
    let y2 = boxes.map(val=>{return val[3]})
    let s  = boxes.map(val=>{return val[4]})
    //area = np.multiply(x2-x1+1, y2-y1+1)
    let area = x1.map((val,index)=>{
        return (x2[index]-val+1) * (y2[index]-y1[index]+1)
    })
    //console.log(area)
    // I = np.array(s.argsort())
    let I1 = s
    I1.sort(function(a,b){
        return a - b;
    })
    s  = boxes.map(val=>{return val[4]})
    let I = []
    I1.map(val=>{
        s.map((vals,index)=>{
            if(val === vals){
                I.push(index)
            } 
        })
    })
    console.log(I)
    let pick = []
    // while len(I)>0:
    //     xx1 = np.maximum(x1[I[-1]], x1[I[0:-1]]) //I[-1] have hightest prob score, I[0:-1]->others
    //     yy1 = np.maximum(y1[I[-1]], y1[I[0:-1]])
    //     xx2 = np.minimum(x2[I[-1]], x2[I[0:-1]])
    //     yy2 = np.minimum(y2[I[-1]], y2[I[0:-1]])
    //     w = np.maximum(0.0, xx2 - xx1 + 1)
    //     h = np.maximum(0.0, yy2 - yy1 + 1)
    //     inter = w * h
    //     if type == 'iom':
    //         o = inter / np.minimum(area[I[-1]], area[I[0:-1]])
    //     else:
    //         o = inter / (area[I[-1]] + area[I[0:-1]] - inter)
    //     pick.append(I[-1])
    //     I = I[np.where(o<=threshold)[0]]
    while(I.length>0){
        let II = I[I.length-1]
        I.splice(I.length-1,1)
        // let x11 = x1[I[I.length-1]]
        // let y11 = y1[I[I.length-1]]
        // let x22 = x2[I[I.length-1]]
        // let y22 = y2[I[I.length-1]]
        // x1.splice(I[I.length-1],1)
        // y1.splice(I[I.length-1],1)
        // x2.splice(I[I.length-1],1)
        // y2.splice(I[I.length-1],1)
        let xx1 = I.map(val=>{
            return Math.max(x1[val],x1[II])
        })
        let yy1 = I.map(val=>{
            return Math.max(y1[val],y1[II])
        })
        let xx2 = I.map(val=>{
            return Math.min(x2[val],x2[II])
        })
        let yy2 = I.map(val=>{
            return Math.min(y2[val],y2[II])
        })
        // console.log(xx1)
        // console.log(yy1)
        // console.log(xx2)
        // console.log(yy2)
        //Math.max(0.0, xx2 - xx1 + 1)
        let w = xx1.map((val,index)=>{
            return Math.max(0.0, xx2[index] - val + 1)
        })
        // let h = Math.max(0.0, yy2 - yy1 + 1)
        let h = yy1.map((val,index)=>{
            return Math.max(0.0, yy2[index] - val + 1)
        })
        //let inter = w * h
        let inter = w.map((val,index)=>{
            return val * h[index]
        })
        // console.log("inter",inter)
        // console.log("area:",area)
        //let o = type === 'iom' ?  [1,1] : inter / (area[I[-1]] + area[I[0:-1]] - inter)
        
        let o = I.map((val,index)=>{
            return inter[index] / (area[II] + area[val] - inter[index])
        })
        console.log(o)
        pick.push(II)
        console.log("pick",pick)
        //I = I[np.where(o<=threshold)[0]]
        let III=[]
        o.map((val,index)=>{
            if(val<=threshold){
                III.push(index)
            }
        })
        I.push(II)
        //console.log(I)
        I = III.map(val=>{
            return I[val]
        })
        console.log(I)
    }
    // result_rectangle = boxes[pick].tolist()
    let result_rectangle = pick.map(val=>{
        return boxes[val]
    })
    console.log("result_rectangle:",result_rectangle)
    return result_rectangle
}

exports.calculateScales = calculateScales
exports.detect_face_12net = detect_face_12net
exports.NMS = NMS