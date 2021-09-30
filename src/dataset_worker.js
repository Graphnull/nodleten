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
const fs = __importStar(require("fs"));
const tfCore = __importStar(require("@tensorflow/tfjs-core"));
const lz4 = __importStar(require("lz4"));
const worker_threads_1 = require("worker_threads");
const dataset_1 = require("./dataset");
class Dataset {
    constructor(params) {
        this.inputSize = 1;
        this.name = params.name;
        this.dataFile = (this.name) + '.bin';
        this.needCreate = params.needCreate;
        this.compressLevel = typeof params.compressLevel === 'number' ? params.compressLevel : 1;
        this.f; // file with dataset
        this._p = 0; //file position
        let shape = params.shape;
        if (!Array.isArray(shape)) {
            throw new Error('Shape field not array');
        }
        shape.forEach(dim => {
            if (typeof dim !== 'number') {
                throw new Error(JSON.stringify(shape) + ' shape is not number array');
            }
            this.inputSize *= dim;
        });
        this.shape = shape;
        this.buf = Buffer.alloc(this.inputSize * 4);
        this.compressBuf = Buffer.alloc(lz4.encodeBound(this.inputSize * 4 * 1.5));
        this.decompressBuf = Buffer.alloc(this.compressBuf.length);
        //position + lenght+ flags+shape
        this.dataInfoProto = new BigUint64Array(1 + 1 + 1 + 4);
        let shapebuf = new Uint32Array(8);
        shapebuf.set(shape, 0);
        Buffer.from(this.dataInfoProto.buffer, this.dataInfoProto.byteOffset, this.dataInfoProto.byteLength).set(Buffer.from(shapebuf.buffer, shapebuf.byteOffset, shapebuf.byteLength), 3 * 8);
        this.list = [];
        this.cache = {};
        this.writeQueue = [];
        this.writing = false;
        this.initialized = this.init(worker_threads_1.parentPort);
    }
    async init(parentPort) {
        this.f = await fs.promises.open((this.name) + '.bin', 'a+');
        if (this.needCreate) {
            await this.f.truncate(0);
            parentPort.postMessage({ id: -1, data: 0 });
        }
        else {
            let header = await fs.promises.readFile((this.name) + '.ndlt');
            let parsed;
            try {
                parsed = dataset_1.parseHeaderBuffer(this.name, header);
            }
            catch (err) {
                parentPort.postMessage({ id: -1, error: err });
                return;
            }
            if (parsed.params.compressLevel !== this.compressLevel) {
                throw new Error('compressLevel not equal');
            }
            this.list = parsed.list;
            parentPort.postMessage({ id: -1, data: parsed.list.length });
        }
    }
    async writeHeaderFile() {
        let headerFile = (this.name) + '.ndlt';
        let f = fs.createWriteStream(headerFile);
        let magicSymbol = Buffer.from('ndlt');
        let shapeSerialized = new Uint32Array(8);
        shapeSerialized.set(this.shape, 0);
        let shapeBuf = Buffer.from(shapeSerialized.buffer, shapeSerialized.byteOffset, shapeSerialized.byteLength);
        let count = new BigUint64Array([BigInt(this.list.length)]);
        let header = Buffer.concat([magicSymbol, shapeBuf, Buffer.from([this.compressLevel, 0, 0, 0]), Buffer.from(count.buffer, count.byteOffset, count.byteLength)]);
        f.write(header);
        let out = [];
        for (let i = 0; i !== this.list.length; i++) {
            let dataInfo = this.list[i];
            out.push(Buffer.from(dataInfo.buffer, dataInfo.byteOffset, dataInfo.byteLength));
        }
        let input = Buffer.concat(out);
        let output = Buffer.alloc(lz4.encodeBound(input.length));
        let length = lz4.encodeBlock(input, output);
        f.write(output.subarray(0, length));
        f.end();
        await new Promise((res) => {
            f.on('finish', res);
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
    async push(data) {
        if (!data) {
            throw new Error('Data not found');
        }
        if (dataset_1.acceptableTypedArrays.indexOf(data.constructor.name) < 0) {
            throw new Error(`Input object expected ${data.constructor.name} type not accept`);
        }
        if (data.length !== this.inputSize) {
            throw new Error(`Input size have ${data.length}. expected ${this.inputSize}`);
        }
        let length = lz4.encodeBlock(Buffer.from(data.buffer, data.byteOffset, data.byteLength), this.compressBuf);
        if (length > this.compressBuf.length) {
            throw new Error(`compressBuf.length (${this.compressBuf.length})<compressedSize (${length})`);
        }
        let writeData = Buffer.allocUnsafe(length);
        this.compressBuf.copy(writeData, 0, 0, length);
        writeData.copy(this.compressBuf, 0, 0, length);
        let p = this._p;
        this._p += length;
        let dataInfo = this.dataInfoProto.slice(0);
        dataInfo[0] = BigInt(p);
        dataInfo[1] = BigInt(length);
        dataInfo[2] = BigInt(dataset_1.TYPEIDS[data.constructor.name]);
        this.list.push(dataInfo);
        this.cache[p] = { writeData };
        this.writeQueue.push({
            data: writeData,
            write: async () => {
                try {
                    this.writing = true;
                    await this.f.write(writeData, 0, length, p);
                    delete this.cache[p];
                    this.writing = false;
                }
                catch (err) {
                    console.log('unhandled error');
                    process.exit(1);
                }
            }
        });
        await this.initialized;
        //TODO batch write
        //if (true) {
        if (!this.writing) {
            while (this.writeQueue.length) {
                await this.writeQueue[0].write();
                this.writeQueue.shift();
            }
        }
        // } else {
        //     let writeQueueLength = 0;
        //     this.writeQueue.forEach((data) => {
        //         writeQueueLength += data.length;
        //     })
        //     console.log('writeQueueLength: ', writeQueueLength);
        //     if ((writeQueueLength + writeData.length) > (1024 * 1024 * 4)) {
        //         await this.initialized;
        //         let writing = this.f.write(Buffer.concat(this.writeQueue.concat([writeData])), 0, writeQueueLength + writeData.length, p)//TODO, //TODO);
        //         await writing;
        //         Object.keys(this.cache).forEach((key) => {
        //             delete this.cache[key];
        //         })
        //         this.cache = {};
        //         this.writeQueue = [];
        //     } else {
        //         this.writeQueue.push(writeData)
        //     }
        // }
    }
    async sync() {
        await this.initialized;
        let promises = Object.keys(this.cache).map(_p => {
            let p = Number(_p);
            let data = this.cache[_p].writeData;
            return this.f.write(data, 0, data.length, p);
        });
        await Promise.all(promises);
        this.cache = {};
    }
    async get(dataInfo) {
        let p = Number(dataInfo[0]);
        let len = Number(dataInfo[1]);
        let out = this.buf;
        let uncompressedSize = 0;
        if (this.cache[p]) {
            let writeData = this.cache[p].writeData;
            uncompressedSize = lz4.decodeBlock(writeData, out);
        }
        else {
            let temp = Buffer.allocUnsafe(len);
            await this.f.read(temp, 0, len, p);
            uncompressedSize = lz4.decodeBlock(temp, out);
        }
        let mainType = dataset_1.ReTYPEIDS[Number(dataInfo[2])];
        if (!mainType) {
            throw new Error('Unknown type of data');
        }
        if (uncompressedSize !== this.inputSize * mainType.BYTES_PER_ELEMENT) {
            throw new Error(`UncompressedSize (${uncompressedSize}) !== out.length (${this.inputSize * mainType.BYTES_PER_ELEMENT})`);
        }
        if (Number(dataInfo[2]) === 1) { //is Float32Array
            return new mainType(this.buf.buffer, this.buf.byteOffset, uncompressedSize / mainType.BYTES_PER_ELEMENT);
        }
        else {
            let innerType = new mainType(this.buf.buffer, this.buf.byteOffset, uncompressedSize / mainType.BYTES_PER_ELEMENT);
            return new Int32Array(innerType);
        }
    }
    async forEachAsync(func) {
        await this.initialized;
        for (let i = 0; i !== this.list.length; i++) {
            let out = await this.get(this.list[i]);
            tfCore.tidy(() => {
                func(tfCore.tensor(out));
            });
        }
    }
    async iterator() {
    }
}
if (!worker_threads_1.isMainThread) {
    let dataset = new Dataset(worker_threads_1.workerData);
    if (!worker_threads_1.parentPort) {
        throw new Error('parentPort not found');
    }
    //dataset.initialized = dataset.init(parentPort as any)
    worker_threads_1.parentPort.on('message', async ({ id, op, data }) => {
        if (!worker_threads_1.parentPort) {
            throw new Error('parentPort not found');
        }
        try {
            switch (op) {
                case ('get'): {
                    let dataInfo = dataset.list[data];
                    if (!dataInfo) {
                        throw new Error(`data in ${data} index not found`);
                    }
                    let result = await dataset.get(dataInfo);
                    worker_threads_1.parentPort.postMessage({ id, data: result });
                    break;
                }
                case ('push'): { // push
                    dataset.push(data);
                    worker_threads_1.parentPort.postMessage({ id });
                    break;
                }
                case ('writeHeaderFile'): {
                    await dataset.sync();
                    await dataset.writeHeaderFile();
                    worker_threads_1.parentPort.postMessage({ id });
                    break;
                }
                default: {
                    throw new Error('unknown command: ' + op);
                }
            }
        }
        catch (error) {
            worker_threads_1.parentPort.postMessage({ id, error });
        }
    });
}
