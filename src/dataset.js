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
exports.zip = exports.openDataset = exports.Dataset = exports.parseHeaderBuffer = exports.DataTypeMap = exports.TFTypes = exports.acceptableTypedArrays = exports.ReTYPEIDS = exports.TYPEIDS = void 0;
const tfCore = __importStar(require("@tensorflow/tfjs-core"));
const path = __importStar(require("path"));
const fs = __importStar(require("fs"));
const lz4 = __importStar(require("lz4"));
const worker_threads_1 = require("worker_threads");
let uniqueId = 0;
exports.TYPEIDS = {
    'Float32Array': 1,
    'Int32Array': 2,
    'Uint8Array': 3,
    'Uint32Array': 4,
    'Buffer': 5,
    'Int8Array': 6,
    'Uint16Array': 7,
    'Int16Array': 8,
};
exports.ReTYPEIDS = [null, Float32Array, Int32Array, Uint8Array, Uint32Array, Buffer, Int8Array, Uint16Array, Int16Array];
exports.acceptableTypedArrays = Object.keys(exports.TYPEIDS);
exports.TFTypes = {
    float32: 1,
    int32: 2,
    bool: 3,
    complex64: 4,
    string: 5
};
exports.DataTypeMap = {
    float32: Float32Array,
    int32: Int32Array,
    bool: Uint8Array,
    complex64: Float32Array,
    string: String
};
let parseHeaderBuffer = (name, header) => {
    let magic = Buffer.from('ndlt');
    if (Buffer.compare(header.slice(0, magic.length), magic) !== 0) {
        throw new Error(`${name + '.ndlt'} is not header file`);
    }
    // 8 dim shape
    let shape = Array.from(new Uint32Array(header.buffer, header.byteOffset + magic.length, 8));
    shape = shape.filter(v => v);
    let compressLevel = header[4 + 8 * Uint32Array.BYTES_PER_ELEMENT];
    let count = Number((new BigUint64Array(header.buffer, header.byteOffset + 4 + 8 * Uint32Array.BYTES_PER_ELEMENT + 1 + 1 + 2, 1))[0]);
    let params = {
        name,
        shape,
        compressLevel,
        count,
        needCreate: false
    };
    let list = [];
    let headLength = 4 + 8 * Uint32Array.BYTES_PER_ELEMENT + 1 + 1 + 2 + 8;
    if (header.length !== headLength) {
        let listArray = new BigUint64Array(count * 7);
        let decodedLength = lz4.decodeBlock(header.subarray(headLength), Buffer.from(listArray.buffer, listArray.byteOffset, listArray.byteLength));
        if (decodedLength !== count * 7 * 8) {
            throw new Error('It is not possible to decompress the data');
        }
        for (let i = 0; i !== count; i++) {
            list.push(listArray.subarray(i * 7, i * 7 + 7));
        }
    }
    return { params, list };
};
exports.parseHeaderBuffer = parseHeaderBuffer;
let uniqueCommandId = 0;
/**
 * Create a Dataset with disk cache
 *
 * Example:
 * ```
 * const rgbDataset = new Dataset({shape:[2, 2, 3]})
 *
 * let image = new Float32Array([0,1,2,3,4,5,1,2,3,4,5,6])
 * rgbDataset.push(image)
 * ```
 * @param WorkerData
 */
class Dataset {
    constructor(params) {
        this.inputSize = 1;
        let workerData = params || {};
        workerData.name = params.name || String(uniqueId++);
        this.name = workerData.name;
        this.dataFile = (this.name) + '.bin';
        params.needCreate = this.needCreate = typeof params.needCreate === 'boolean' ? params.needCreate : true;
        let shape = params.shape;
        if (!Array.isArray(shape)) {
            throw new Error(JSON.stringify(shape) + ' shape is not number array');
        }
        shape.forEach(dim => {
            if (typeof dim !== 'number') {
                throw new Error(JSON.stringify(shape) + ' shape is not number array');
            }
            this.inputSize *= dim;
        });
        this.shape = workerData.shape = shape;
        workerData.compressLevel = typeof params.compressLevel === 'number' ? params.compressLevel : 1;
        this.worker = new worker_threads_1.Worker(path.resolve(__dirname, 'dataset_worker.js'), {
            workerData: workerData
        });
        this.worker.dataI = 0;
        this.list = [];
        if (params.count) {
            for (let i = 0; i !== params.count; i++) {
                this.list.push(this.worker.dataI++);
            }
        }
        this.initializing = new Promise((res, rej) => {
            let onMessage = (message) => {
                if (message.error) {
                    rej(message.error);
                }
                if (message.id === -1) {
                    this.worker.off('message', onMessage);
                    res();
                }
            };
            this.worker.on('message', onMessage);
        });
    }
    async _sendCommand(op, data, transferableObjects) {
        let commandId = uniqueCommandId++;
        this.worker.postMessage({ op, id: commandId, data }, transferableObjects);
        return new Promise((res, rej) => {
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
    push(objs) {
        if (exports.acceptableTypedArrays.indexOf(objs.constructor.name) < 0) {
            throw new Error(`Input object ${objs.constructor.name} type is not accept`);
        }
        if (objs.length !== this.inputSize) {
            throw new Error(`Input size have ${objs.length}. expected ${this.inputSize}`);
        }
        this.list.push(this.worker.dataI++);
        let clone = objs.slice(0);
        return this._sendCommand('push', clone, [clone.buffer]);
    }
    send(objs) {
        if (exports.acceptableTypedArrays.indexOf(objs.constructor.name) < 0) {
            throw new Error(`Input object ${objs.constructor.name} type is not accept`);
        }
        if (objs.length !== this.inputSize) {
            throw new Error(`Input size have ${objs.length}. expected ${this.inputSize}`);
        }
        this.list.push(this.worker.dataI++);
        return this._sendCommand('push', objs, [objs.buffer]);
    }
    async get(i) {
        let index = this.list[i];
        return this._sendCommand('get', index);
    }
    async forEachAsync(func) {
        for (let i = 0; i !== this.list.length; i++) {
            let out = await this.get(this.list[i]);
            tfCore.tidy(() => {
                func(tfCore.tensor(out, [1].concat(this.shape)));
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
    destroy(deleteCacheFile) {
        this.worker.terminate();
        if (deleteCacheFile) {
            fs.unlinkSync(this.dataFile);
        }
    }
}
exports.Dataset = Dataset;
async function openDataset(name) {
    let f = await fs.promises.open(name + '.ndlt', 'r');
    //magic + shapeinfo+ compressioninfo + typeinfo + other+ count
    let len = 4 + 8 * Uint32Array.BYTES_PER_ELEMENT + 1 + 1 + 2 + 8;
    let header = Buffer.alloc(len);
    await f.read(header, 0, len, 0);
    await f.close();
    let { params } = exports.parseHeaderBuffer(name, header);
    let dataset = new Dataset(params);
    await dataset.initializing;
    return dataset;
}
exports.openDataset = openDataset;
class Zip {
    constructor(datasets) {
        this.datasets = datasets;
        this.firstDataset = Object.values(datasets)[0];
        if (!this.firstDataset) {
            throw new Error('dataset not found');
        }
        this.keys = Object.keys(datasets);
        this.prefetches = {};
        this.keys.forEach(key => {
            this.prefetches[key] = [];
            let dataset = this.datasets[key];
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
                    out[key] = tfCore.tensor(typedArrays[key], [1].concat(this.datasets[key].shape));
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
                if (count === i) {
                    return { value: null, done: true };
                }
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
                    out[key] = tfCore.tensor(results[key], [1].concat(this.datasets[key].shape)); // this.tensorBuffers[key].toTensor();
                }
                i++;
                return { value: out, done: false };
            }
        };
    }
}
/**
 * Create a Dataset by zipping dict
 *
 * Example:
 * ```
 * const xDataset = new Dataset({shape:[100, 100]})
 * const yDataset = new Dataset({shape:[1]})
 *
 * const xyDataset = zip({xs: xDataset, ys: yDataset})
 *
 * model.fitDataset(xyDataset, {});
 * ```
 * @param {[key]:Dataset}
 * @returns Zip
 */
function zip(objects) {
    return new Zip(objects);
}
exports.zip = zip;
