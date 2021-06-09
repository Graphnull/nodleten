"use strict";
var __createBinding = (this && this.__createBinding) || (Object.create ? (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    Object.defineProperty(o, k2, { enumerable: true, get: function() { return m[k]; } });
}) : (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    o[k2] = m[k];
}));
var __setModuleDefault = (this && this.__setModuleDefault) || (Object.create ? (function(o, v) {
    Object.defineProperty(o, "default", { enumerable: true, value: v });
}) : function(o, v) {
    o["default"] = v;
});
var __importStar = (this && this.__importStar) || function (mod) {
    if (mod && mod.__esModule) return mod;
    var result = {};
    if (mod != null) for (var k in mod) if (k !== "default" && Object.prototype.hasOwnProperty.call(mod, k)) __createBinding(result, mod, k);
    __setModuleDefault(result, mod);
    return result;
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.zip = exports.Dataset = void 0;
const tfCore = __importStar(require("@tensorflow/tfjs-core"));
const path = __importStar(require("path"));
const worker_threads_1 = require("worker_threads");
let uniqueId = 0;
const FLOAT32TYPE = 2;
let uniqueCommandId = 0;
/**
 * Create a Dataset with disk cache
 *
 * Example:
 * const rgbDataset = new Dataset({shape:[2, 2, 3]})
 *
 * let image = new Float32Array([0,1,2,3,4,5,1,2,3,4,5,6])
 * rgbDataset.push(image)
 *
 * @param WorkerData
 */
class Dataset {
    constructor(params) {
        let workerData = params || {};
        workerData.name = params.name || String(uniqueId++);
        this.name = workerData.name;
        this.shape = workerData.shape = params.shape;
        workerData.compressLevel = typeof params.compressLevel === 'number' ? params.compressLevel : 1;
        this.worker = new worker_threads_1.Worker(path.resolve(__dirname, 'dataset_worker.js'), {
            workerData: workerData
        });
        this.worker.dataI = 0;
        this.list = [];
    }
    push(objs) {
        this.list.push(this.worker.dataI++);
        let clone = objs.slice(0);
        this.worker.postMessage({ op: 'push', data: clone }, [clone.buffer]);
    }
    send(objs) {
        this.list.push(this.worker.dataI++);
        this.worker.postMessage({ op: 'push', data: objs }, [objs.buffer]);
    }
    async get(i) {
        let index = this.list[i];
        let commandId = uniqueCommandId++;
        this.worker.postMessage({ index, id: commandId, op: 'get' });
        return await new Promise((res, rej) => {
            let onMessage = (message) => {
                if (message.error) {
                    rej(message.error);
                }
                if (message.id === commandId) {
                    this.worker.off('message', onMessage);
                    res(message.data);
                }
            };
            this.worker.on('message', onMessage);
        });
    }
    async forEachAsync(func) {
        for (let i = 0; i !== this.list.length; i++) {
            let out = await this.get(this.list[i]);
            tfCore.tidy(() => {
                func(tfCore.tensor(out));
            });
        }
    }
    async writeHeaderFile() {
        let commandId = uniqueCommandId++;
        this.worker.postMessage({ id: commandId, op: 'writeHeaderFile' });
        let result = await new Promise((res, rej) => {
            let onMessage = (message) => {
                if (message.error) {
                    rej(message.error);
                }
                if (message.id === commandId) {
                    this.worker.off('message', onMessage);
                    res(message.data);
                }
            };
            this.worker.on('message', onMessage);
        });
        return result;
    }
    destroy() {
        this.worker.terminate();
    }
}
exports.Dataset = Dataset;
class Zip {
    constructor(datasets) {
        this.datasets = datasets;
        this.firstDataset = Object.values(datasets)[0];
        if (!this.firstDataset) {
            throw new Error('dataset not found');
        }
        this.keys = Object.keys(datasets);
        this.tensorBuffers = {};
        this.prefetches = {};
        this.keys.forEach(key => {
            this.prefetches[key] = [];
            let dataset = this.datasets[key];
            this.tensorBuffers[key] = tfCore.buffer([1].concat(dataset.shape));
        });
    }
    batch() { throw new Error('not implemented'); }
    concatenate() { throw new Error('not implemented'); }
    filter() { throw new Error('not implemented'); }
    map() { throw new Error('not implemented'); }
    mapAsync() { throw new Error('not implemented'); }
    prefetch() { throw new Error('not implemented'); }
    repeat() { throw new Error('not implemented'); }
    shuffle() { throw new Error('not implemented'); }
    skip() { throw new Error('not implemented'); }
    take() { throw new Error('not implemented'); }
    toArray() { throw new Error('not implemented'); }
    async forEachAsync(func) {
        let list = this.firstDataset.list;
        for (let i = 0; i !== list.length; i++) {
            let typedArrays = {};
            for (let k = 0; k !== this.keys.length; k++) {
                let key = this.keys[k];
                typedArrays[key] = await this.datasets[key].get(this.datasets[key].list[i]);
            }
            let out = {};
            tfCore.tidy(() => {
                for (let k = 0; k !== this.keys.length; k++) {
                    let key = this.keys[k];
                    out[key] = tfCore.tensor(typedArrays[key], [1, typedArrays[key].length]);
                }
                func(out);
            });
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
                let results = {};
                let getingDatas = this.keys.map(async (key) => {
                    //check prefetch data
                    let prefetchLine = this.prefetches[key];
                    let prefetch = prefetchLine.find(prefetch => prefetch.i === this.datasets[key].list[i]);
                    if (prefetch) {
                        if (prefetch.data) {
                            results[key] = prefetch.data;
                        }
                        else {
                            results[key] = await prefetch.promise;
                        }
                    }
                    else {
                        results[key] = await this.datasets[key].get(this.datasets[key].list[i]);
                    }
                });
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
                    });
                    ids = ids.filter(id => id !== -1);
                    for (let ni = 0; ni !== ids.length; ni++) {
                        let i = ids[ni];
                        let prefetch = {
                            i,
                            promise: this.datasets[key].get(i)
                                .then(data => {
                                return prefetch.data = data;
                            })
                        };
                        this.prefetches[key].push(prefetch);
                    }
                }
                let out = {};
                for (let k = 0; k !== this.keys.length; k++) {
                    let key = this.keys[k];
                    this.tensorBuffers[key].values = results[key];
                    out[key] = this.tensorBuffers[key].toTensor();
                }
                return { value: out, done: false };
            }
        };
    }
}
/**
 * Create a Dataset by zipping dict
 *
 * Example:
 * const xDataset = new Dataset({shape:[100, 100]})
 * const yDataset = new Dataset({shape:[1]})
 *
 * const xyDataset = zip({xs: xDataset, ys: yDataset})
 *
 * model.fitDataset(xyDataset, {});
 *
 * @param {[key]:Dataset}
 * @returns Zip
 */
function zip(objects) {
    return new Zip(objects);
}
exports.zip = zip;
