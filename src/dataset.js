
let fsPromises = require('fs').promises
let fs = require('fs')
let tfData = require('@tensorflow/tfjs-data');
let tfCore = require('@tensorflow/tfjs-core');
let lz4 = require('lz4');
let path = require('path')
const {
    Worker, isMainThread, parentPort, workerData
} = require('worker_threads');

let uniqueId = 0;

const FLOAT32TYPE = 2;


let uniqueCommandId = 0;
class Dataset extends tfData.Dataset {
    constructor(params = {}) {
        super();
        let workerData = params;
        workerData.name = params.name || (uniqueId++);
        workerData.dataFile = (this.name) + '.bin'
        this.shape = workerData.shape = params.shape;

        workerData.compressLevel = typeof params.compressLevel === 'number' ? params.compressLevel : 1;

        this.worker = new Worker(path.resolve(__dirname,'dataset_worker.js'), {
            workerData: workerData
        });
        this.worker.dataI = 0;
        this.list = [];
    }
    push(objs) {
        this.list.push(this.worker.dataI++);
        let clone = objs.slice(0);
        this.worker.postMessage({ op: 'push', data: clone }, [clone.buffer])
    }
    send(objs) {
        this.list.push(this.worker.dataI++);
        this.worker.postMessage({ op: 'push', data: objs }, [objs.buffer])
    }
    async get(i) {

        let index = this.list[i];
        let commandId = uniqueCommandId++;
        this.worker.postMessage({ index, id: commandId, op: 'get' })
        let result = await new Promise((res, rej) => {
            let onMessage = (message) => {
                if (message.error) {
                    rej(message.error)
                }
                if (message.id === commandId) {
                    this.worker.off('message', onMessage)
                    res(message.data)
                }
            }
            this.worker.on('message', onMessage)

        })
        return result;
    }
    async forEachAsync(func) {
        await this.fileOpen;

        for (let i = 0; i !== this.list.length; i++) {

            let out = await this.get(this.list[i]);

            tfCore.tidy(() => {
                func(tfCore.tensor(out));
            })
        }

    }
    async writeHeaderFile() {
        let commandId = uniqueCommandId++;
        this.worker.postMessage({ id: commandId, op: 'writeHeaderFile' })
        let result = await new Promise((res, rej) => {
            let onMessage = (message) => {
                if (message.error) {
                    rej(message.error)
                }
                if (message.id === commandId) {
                    this.worker.off('message', onMessage)
                    res(message.data)
                }
            }
            this.worker.on('message', onMessage)

        })
        return result;
    }
    destroy() {
        this.worker.terminate();
    }
}
class Zip {
    constructor(datasets) {

        this.datasets = datasets;
        this.firstDataset = Object.values(datasets)[0]
        this.keys = Object.keys(datasets);
        this.tensorBuffers = {};

        this.prefetches = {};
        this.keys.forEach(key => {
            this.prefetches[key] = [];
            this.tensorBuffers[key] = tfCore.buffer([1].concat(this.datasets[key].shape));
        })
    }
    batch() { throw new Error('not implemented') }
    concatenate() { throw new Error('not implemented') }
    filter() { throw new Error('not implemented') }
    forEachAsync() { throw new Error('not implemented') }
    map() { throw new Error('not implemented') }
    mapAsync() { throw new Error('not implemented') }
    prefetch() { throw new Error('not implemented') }
    repeat() { throw new Error('not implemented') }
    shuffle() { throw new Error('not implemented') }
    skip() { throw new Error('not implemented') }
    take() { throw new Error('not implemented') }
    toArray() { throw new Error('not implemented') }
    async forEachAsync(func) {
        let list = this.firstDataset.list;
        for (let i = 0; i !== list.length; i++) {
            let out = {}

            for (let k = 0; k !== this.keys.length; k++) {
                let key = this.keys[k];
                out[key] = await this.datasets[key].get(this.datasets[key].list[i]);
            }

            tfCore.tidy(() => {
                for (let k = 0; k !== this.keys.length; k++) {
                    let key = this.keys[k];
                    out[key] = tfCore.tensor(out[key], [1, out[key].length]);
                }
                func(out);
            })
        }
    }
    async iterator() {
        let i = 0;
        let count = this.firstDataset.list.length;

        return {
            next: async () => {
                if (i % 256 === 0) {
                    console.log('i', i);
                }
                if (count === i) {
                    return { value: null, done: true };
                }
                i++;
                let out = {}

                let getingDatas = this.keys.map(async key => {

                    //check prefetch data
                    let prefetchLine = this.prefetches[key];
                    let prefetch = prefetchLine.find(prefetch => prefetch.i === this.datasets[key].list[i])

                    if (prefetch) {
                        if (prefetch.data) {
                            out[key] = prefetch.data;
                        } else {
                            out[key] = await prefetch.promise;
                        }
                    } else {
                        out[key] = await this.datasets[key].get(this.datasets[key].list[i]);
                    }
                })
                await Promise.all(getingDatas);

                const prefetchSize = 9;
                //start prefetch for all datasets
                for (let k = 0; k !== this.keys.length; k++) {
                    let key = this.keys[k];

                    let ids = this.datasets[key].list.slice(i + 1, i + 1 + prefetchSize);

                    this.prefetches[key] = this.prefetches[key].filter(prefetch => {
                        let prefetchExist = ids.findIndex(id => id === prefetch.i);
                        if (prefetchExist > -1) {
                            ids[prefetchExist] = -1;
                        }
                        return prefetchExist > -1;
                    })
                    ids = ids.filter(id => id !== -1)

                    for (let ni = 0; ni !== ids.length; ni++) {
                        let i = ids[ni];
                        let prefetch = {}
                        let promise = this.datasets[key].get(i)
                            .then(data => {
                                return prefetch.data = data;
                            })
                        prefetch.promise = promise;
                        prefetch.i = i;

                        this.prefetches[key].push(prefetch)
                    }

                }

                for (let k = 0; k !== this.keys.length; k++) {
                    let key = this.keys[k];
                    this.tensorBuffers[key].values = out[key]
                    out[key] = this.tensorBuffers[key].toTensor();

                }



                return { value: out, done: false };
            }
        }
    }
}
function zip(objects) {

    return new Zip(objects);
}


module.exports.zip = zip;
module.exports.Dataset = Dataset;
