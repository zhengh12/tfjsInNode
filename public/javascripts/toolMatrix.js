
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
    cls_prob.map((val,index0)=>{
        return val.map((val,index1)=>{
            if(val>=threshold){
                return [index0,index1]
            }
        }) 
    })
    boundingbox = np.array([x,y]).T
    bb1 = np.fix((stride * (boundingbox) + 0 ) * scale)
    bb2 = np.fix((stride * (boundingbox) + 11) * scale)
    boundingbox = np.concatenate((bb1,bb2),axis = 1)
    dx1 = roi[0][x,y]
    dx2 = roi[1][x,y]
    dx3 = roi[2][x,y]
    dx4 = roi[3][x,y]
    score = np.array([cls_prob[x,y]]).T
    offset = np.array([dx1,dx2,dx3,dx4]).T
    boundingbox = boundingbox + offset*12.0*scale
    rectangles = np.concatenate((boundingbox,score),axis=1)
    rectangles = rect2square(rectangles)
    pick = []
    for i in range(len(rectangles)):
        x1 = int(max(0     ,rectangles[i][0]))
        y1 = int(max(0     ,rectangles[i][1]))
        x2 = int(min(width ,rectangles[i][2]))
        y2 = int(min(height,rectangles[i][3]))
        sc = rectangles[i][4]
        if x2>x1 and y2>y1:
            pick.append([x1,y1,x2,y2,sc])
    return NMS(pick,0.3,'iou')
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
    area = np.multiply(x2-x1+1, y2-y1+1)
    I = np.array(s.argsort())
    pick = []
    while len(I)>0:
        xx1 = np.maximum(x1[I[-1]], x1[I[0:-1]]) //I[-1] have hightest prob score, I[0:-1]->others
        yy1 = np.maximum(y1[I[-1]], y1[I[0:-1]])
        xx2 = np.minimum(x2[I[-1]], x2[I[0:-1]])
        yy2 = np.minimum(y2[I[-1]], y2[I[0:-1]])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        if type == 'iom':
            o = inter / np.minimum(area[I[-1]], area[I[0:-1]])
        else:
            o = inter / (area[I[-1]] + area[I[0:-1]] - inter)
        pick.append(I[-1])
        I = I[np.where(o<=threshold)[0]]
    result_rectangle = boxes[pick].tolist()
    return result_rectangle
}

exports.calculateScales = calculateScales
exports.detect_face_12net = detect_face_12net
