let tf = require('@tensorflow/tfjs-node')
let { Dataset, openDataset } = require('./../src/dataset')
let fs = require('fs')


function Float32Concat(first, second) {
    var firstLength = first.length,
        result = new Float32Array(firstLength + second.length);

    result.set(first);
    result.set(second, firstLength);

    return result;
}



async function test() {

    let dataset = new Dataset({ shape: [6 + 1000000], name: 'test' });


    for (let i = 0; i !== 1000; i++) {
        let addData = new Float32Array(1000000);
        for (let j = 0; j !== 10000; j++) {
            addData[j] = Math.random();
        }
        let data = Float32Concat(new Float32Array([i, 0, 1, 2, 3, i]), addData)
        dataset.push(data)
        await new Promise(res => setTimeout(res, 1))
    }

    let i = 0;
    await dataset.forEachAsync((tensor) => {
        let j = i++;
        let data = tensor.dataSync();
        if ((data[0] | 0) != (j | 0) || (data[1] | 0) != 0 || (data[2] | 0) != 1 || (data[3] | 0) != 2 || (data[4] | 0) != 3 || (data[5] | 0) != (j | 0)) {
            throw new Error('' + j + ' not equal' + data.slice(0, 6))

        }
    })

    await dataset.writeHeaderFile();
    console.log(process.memoryUsage());
    if (process.memoryUsage().rss > 300 * 1000 * 1000) {
        throw new Error(`data not cached (${process.memoryUsage().rss / 1000 / 1000}>${300})`)

    }
    let files = fs.readdirSync('./')

    if (!files.find(f => f === 'test.bin') || !files.find(f => f === 'test.ndlt')) {
        throw new Error('test file not found')
    }
    if (!(fs.statSync('./test.bin').size > 0)) {
        throw new Error('test.bin file small')
    }
    if (!(fs.statSync('./test.ndlt').size > 0)) {
        throw new Error('test.ndlt file small')
    }

    if (dataset.skip(100).list.length !== (dataset.list.length - 100)) {
        throw new Error('Skip not work')
    }
    if (dataset.take(100).list.length !== 100) {
        throw new Error('take not work')
    }
    if (dataset.repeat(2).list.length !== (dataset.list.length * 2)) {
        throw new Error('repeat not work')
    }

    console.log('success')
    dataset.destroy();

    dataset = await openDataset('test')

    i = 0;
    await dataset.forEachAsync((tensor) => {
        let j = i++;
        let data = tensor.dataSync();
        if ((data[0] | 0) != (j | 0) || (data[1] | 0) != 0 || (data[2] | 0) != 1 || (data[3] | 0) != 2 || (data[4] | 0) != 3 || (data[5] | 0) != (j | 0)) {
            throw new Error('' + j + ' not equal' + data.slice(0, 6))

        }
    })
    dataset.destroy(true);

    var normalDataset;
    try {
        normalDataset = new Dataset({ name: 'normal', shape: [768, 1024, 3] });
        await normalDataset.push(new Uint8Array(10))
        throw new Error('not')
    } catch (err) {
        if (err.message === 'not') {
            throw new Error('Not acceptable lenght')
        }
    }

    normalDataset.push(Buffer.alloc(768 * 1024 * 3));
    normalDataset.push(Buffer.alloc(768 * 1024 * 3));
    let loss = (result, output, mask, hit) => {
        let start = result.shape.map(v => 0)
        start[start.length - 3] = start[start.length - 2] = 2
        let end = result.shape.map(v => v)
        end[end.length - 2] = end[end.length - 2] - 4
        end[end.length - 3] = end[end.length - 3] - 4

        return tf.pow(result.sub(output), 2)
            .slice(start, end)
            .mean();
    }
    let optimizer = tf.train.adadelta(0.03);


    const input = tf.input({ shape: [768, 1024, 3] });
    const drop = tf.layers.dropout({ rate: 0.05 }).apply(input);
    const conv = tf.layers.conv2d({ filters: 3, kernelSize: [1, 1], activation: 'elu', padding: 'same' }).apply(drop);
    var model = tf.model({ inputs: input, outputs: conv })

    model.compile({ optimizer, loss });


    normalDataset.destroy(true)
}
test().catch(err => {
    console.error(err);
    process.exit(1)
});
