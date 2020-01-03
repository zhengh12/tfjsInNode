const tf = require("@tensorflow/tfjs-node");
const inceptionResNetV2 = require("./inceptionResNetV2")
const fs = require("fs");
const readLine = require("readline");
const lfw_folder = 'E:/tensorflow/facenet-master/data/lfw'
const identity_annot_filename = 'E:/tensorflow/facenet-master/data/identity_CelebA.txt'

function readFileToArr(fReadName, cb) {

    var arr = [];
    var readObj = readLine.createInterface({
        input: fs.createReadStream(fReadName)
    });

    readObj.on('line', function (line) {
        arr.push(line);
    });
    readObj.on('close', function () {
        console.log('readLine close....');
        cb(arr);
    });
}

async function get_data_stats(){
    let lines = []
    await new Promise(resolve => {
        readFileToArr(identity_annot_filename, function (arr) {
            resolve('resolved');
            lines = arr;
        })
    })
    console.log(lines)
    ids = []
    images = []
    image2id = []
    id2images = {}
    
    for (line of lines){
        if(line.length > 0){
            let tokens = line.split(' ')
            let image_name = tokens[0]
            if (image_name != '202599.jpg'){
                let id = tokens[1]
                ids.push(id)
                images.push(image_name)
                image2id[image_name] = id
            }
                
        }
    }

}

function get_random_triplets(){

}

function makeIterator(usage) {
    const numElements = 10;
    let index = 0;
    const batch_size = 128
    let datas = usage === 'train' ? get_random_triplets() : get_lfw_validation()

    const iterator = {
        next: () => {
            let result;
            if (index < numElements) {
                result = {value: {xs: tf.tensor(index), ys: tf.tensor(index)}, done: false};
                index++;
                return result;
            }
            return {value: {xs: tf.tensor(index), ys: tf.tensor(index)}, done: true};
        }
    } 
    return iterator;
}

const ds = tf.data.generator(makeIterator);
get_data_stats()
// let model = inceptionResNetV2.create_inception_resnet_v2()
// model.compile(
//     {
//         loss: 'meanSquaredError',
//         optimizer: 'sgd',
//         metrics: ['MAE']
//     }
// )
// model.fitDataset(ds,{batchesPerEpoch:1, epochs:10, verbose:1, validationData:ds, validationBatches:1, callbacks:tf.node.tensorBoard('./public/logs')})

exports.ds = ds
// ds.forEachAsync(e => console.log(e[0].print()));