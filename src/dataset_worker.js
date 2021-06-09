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
const fs_1 = require("fs");
const fs = __importStar(require("fs"));
const tfCore = __importStar(require("@tensorflow/tfjs-core"));
const lz4 = __importStar(require("lz4"));
const worker_threads_1 = require("worker_threads");
let uniqueId = 0;
const FLOAT32TYPE = 2;
class Dataset {
    constructor(params = {}) {
        this.name = params.name || (uniqueId++);
        this.dataFile = (this.name) + '.bin';
        this.shape = params.shape;
        this.compressLevel = typeof params.compressLevel === 'number' ? params.compressLevel : 1;
        this.f; // file with dataset
        this._p = 0; //file position
        if (!Array.isArray(this.shape)) {
            throw new Error('Shape field not array');
        }
        let len = 0;
        let shape = this.shape;
        if (!Array.isArray(shape)) {
            throw new Error(JSON.stringify(shape) + ' shape is not number array');
        }
        shape.forEach(dim => {
            if (typeof dim !== 'number') {
                throw new Error(JSON.stringify(shape) + ' shape is not number array');
            }
            len += dim | 0;
        });
        this.buf = new Float32Array(len);
        this.compressBuf = Buffer.alloc(lz4.encodeBound(len * 4 * 1.5)); //Buffer.alloc(lz4.encodeBound(len*4));
        this.decompressBuf = Buffer.alloc(this.compressBuf.length);
        this.fileOpen = fs_1.promises.open(this.dataFile, 'a+').then((f) => {
            this.f = f;
            //clear file
            return this.f.truncate(0);
        });
        this.list = [];
        this.cache = {};
        this.writeQueue = [];
        this.writing = false;
    }
    async writeHeaderFile() {
        let headerFile = (this.name) + '.ndlt';
        let f = fs.createWriteStream(headerFile);
        let magicSymbol = Buffer.from('ndlt');
        let shapeSerialized = Buffer.from(JSON.stringify(this.shape));
        let header = Buffer.concat([magicSymbol, Buffer.from(new Uint32Array([shapeSerialized.length])), shapeSerialized, Buffer.from([this.compressLevel, 0, 0, 0, 0])]);
        f.write(header);
        for (let i = 0; i !== this.list.length; i++) {
            let dataInfo = this.list[i];
            f.write(Buffer.from(dataInfo.buffer, dataInfo.byteOffset, dataInfo.byteLength));
        }
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
    async push(objs) {
        if (!objs) {
            throw new Error('Data not found');
        }
        let data = objs;
        let length = lz4.encodeBlock(Buffer.from(data.buffer, data.byteOffset, data.byteLength), this.compressBuf);
        if (length > this.compressBuf.length) {
            throw new Error(`compressBuf.length (${this.compressBuf.length})<compressedSize (${length})`);
        }
        let writeData = Buffer.allocUnsafe(length);
        this.compressBuf.copy(writeData, 0, 0, length);
        writeData.copy(this.compressBuf, 0, 0, length);
        let p = this._p;
        this._p += length;
        let dataInfo = new BigUint64Array([BigInt(p), BigInt(length), BigInt(FLOAT32TYPE), 0n]);
        this.list.push(dataInfo);
        this.cache[p] = { writeData };
        this.writeQueue.push({
            data: writeData,
            write: async () => {
                this.writing = true;
                await this.f.write(writeData, 0, length, p);
                delete this.cache[p];
                this.writing = false;
            }
        });
        await this.fileOpen;
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
        //         await this.fileOpen;
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
        await this.fileOpen;
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
        let out = Buffer.from(this.buf.buffer, this.buf.byteOffset, this.buf.byteLength);
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
        if (uncompressedSize !== out.length) {
            console.log(p, len, out.length, this.cache[p]);
            throw new Error(`uncompressedSize (${uncompressedSize}) !== this.buf.byteLength (${this.buf.byteLength})`);
        }
        return this.buf;
    }
    async forEachAsync(func) {
        await this.fileOpen;
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
let dataset = new Dataset(worker_threads_1.workerData);
if (!worker_threads_1.parentPort) {
    throw new Error('parentPort not found');
}
worker_threads_1.parentPort.on('message', async (message) => {
    if (!worker_threads_1.parentPort) {
        throw new Error('parentPort not found');
    }
    try {
        switch (message.op) {
            case ('get'): {
                let dataInfo = dataset.list[message.index];
                if (!dataInfo) {
                    console.log('ddddd', dataset.list.length);
                    throw new Error(`data in ${message.index} index not found`);
                }
                let data = await dataset.get(dataInfo);
                worker_threads_1.parentPort.postMessage({ id: message.id, data });
                break;
            }
            case ('push'): { // push
                dataset.push(message.data);
                worker_threads_1.parentPort.postMessage({ id: message.id });
                break;
            }
            case ('writeHeaderFile'): {
                await dataset.sync();
                await dataset.writeHeaderFile();
                worker_threads_1.parentPort.postMessage({ id: message.id });
                break;
            }
            default: {
                throw new Error('unknown command: ' + message.op);
            }
        }
    }
    catch (error) {
        worker_threads_1.parentPort.postMessage({ id: message.id, error });
    }
});
