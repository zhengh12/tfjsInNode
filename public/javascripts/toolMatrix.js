
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

module.exports=calculateScales