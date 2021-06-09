import {promises as fsPromises } from 'fs'
import * as fs from 'fs'
import * as tfCore from '@tensorflow/tfjs-core'
import * as lz4 from 'lz4'
import {parentPort, workerData} from 'worker_threads'
import { TypedArray } from '@tensorflow/tfjs-core'
import { WorkerData, Dtype } from './dataset'

let uniqueId = 0;


const TYPEIDS = {
    'Float32Array':2,
    'Uint8Array':8,
    'Buffer':9,
    'Int8Array':10,
    'Uint16Array':11,
    'Int16Array':12,
    'Uint32Array':13,
    'Int32Array':14,
}
interface WriteOperation {
    write: Function
    data: Buffer
}

interface Message {
    id: Number
    op:string
    data?: any
}

class Dataset {
    name:string
    dataFile:string
    shape:number[]
    compressLevel:number
    f?: fs.promises.FileHandle
    _p: number
    fileOpen: Promise<void | fs.promises.FileHandle>
    list: BigUint64Array[]
    cache:any
    writeQueue: WriteOperation[]
    writing:boolean
    buf:Buffer
    compressBuf:Buffer
    decompressBuf:Buffer
    inputSize = 1;
    type:Dtype
    constructor(params: WorkerData = {}) {
        this.name = params.name || String(uniqueId++);
        this.dataFile = (this.name) + '.bin'
        this.type = params.type||'Float32Array';

        let mainType = global[this.type];
        this.compressLevel = typeof params.compressLevel === 'number' ? params.compressLevel : 1;

        this.f;// file with dataset

        this._p = 0;//file position

        let shape = params.shape;
        if (!Array.isArray(shape)) {
            throw new Error('Shape field not array');
        }

        shape.forEach(dim => {
            if (typeof dim !== 'number') {
                throw new Error(JSON.stringify(shape) + ' shape is not number array');
            }
            this.inputSize*=dim;
        })
        
        this.shape = shape;

        this.buf = Buffer.alloc(this.inputSize*mainType.BYTES_PER_ELEMENT);
        
        this.compressBuf = Buffer.alloc(lz4.encodeBound(this.inputSize * mainType.BYTES_PER_ELEMENT * 1.5));
        this.decompressBuf = Buffer.alloc(this.compressBuf.length);

        this.fileOpen = fsPromises.open(this.dataFile, 'a+').then((f) => {
            this.f = f;
            //clear file
            return this.f.truncate(0)
        });

        this.list = [];

        this.cache = {};

        this.writeQueue = [];
        this.writing = false;
    }
    async writeHeaderFile() {

        let headerFile = (this.name) + '.ndlt'
        let f = fs.createWriteStream(headerFile);

        let magicSymbol = Buffer.from('ndlt')
        let shapeSerialized = Buffer.from(JSON.stringify(this.shape));
        let header = Buffer.concat([magicSymbol, Buffer.from(new Uint32Array([shapeSerialized.length])), shapeSerialized, Buffer.from([this.compressLevel, 0, 0, 0, 0])]);
        f.write(header);

        for (let i = 0; i !== this.list.length; i++) {
            let dataInfo = this.list[i]
            f.write(Buffer.from(dataInfo.buffer, dataInfo.byteOffset, dataInfo.byteLength))
        }
        f.end();
        await new Promise((res) => {
            f.on('finish', res)
        })

    }

    batch() { throw new Error('not implemented') }
    concatenate() { throw new Error('not implemented') }
    filter() { throw new Error('not implemented') }
    map() { throw new Error('not implemented') }
    mapAsync() { throw new Error('not implemented') }
    prefetch() { throw new Error('not implemented') }
    repeat() { throw new Error('not implemented') }
    shuffle() { throw new Error('not implemented') }
    skip() { throw new Error('not implemented') }
    take() { throw new Error('not implemented') }
    toArray() { throw new Error('not implemented') }
    async push(objs: TypedArray) {
        if (!objs) {
            throw new Error('Data not found')
        }
        if(objs.length!==this.inputSize){
            throw new Error(`Input size have ${objs.length}. expected ${this.inputSize}`)
        }
        let data = objs;

        let length = lz4.encodeBlock(Buffer.from(data.buffer, data.byteOffset, data.byteLength), this.compressBuf)

        if (length > this.compressBuf.length) {
            throw new Error(`compressBuf.length (${this.compressBuf.length})<compressedSize (${length})`)
        }

        let writeData = Buffer.allocUnsafe(length)
        this.compressBuf.copy(writeData, 0, 0, length);
        writeData.copy(this.compressBuf, 0,0,length)

        let p = this._p;
        this._p += length;

        let dataInfo = new BigUint64Array([BigInt(p), BigInt(length), BigInt(TYPEIDS[this.type]), 0n]);

        this.list.push(dataInfo);



        this.cache[p] = { writeData }
        this.writeQueue.push({
            data: writeData, write: async () => {
                try{
                this.writing = true;
                await (this.f as fsPromises.FileHandle).write(writeData, 0, length, p);
                delete this.cache[p];
                this.writing = false;
                }catch(err){
                    console.log('unhandled error');
                    process.exit(1);
                }

            }
        })


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
            let data = this.cache[_p].writeData
            return (this.f as fsPromises.FileHandle).write(data, 0, data.length, p);
        })
        await Promise.all(promises)
        this.cache = {};
    }
    async get(dataInfo: BigUint64Array) {

        let p = Number(dataInfo[0]);
        let len = Number(dataInfo[1]);
        let out = this.buf;
        let uncompressedSize = 0;
        if (this.cache[p]) {
            let writeData = this.cache[p].writeData;
            uncompressedSize = lz4.decodeBlock(writeData, out)
        } else {
            let temp = Buffer.allocUnsafe(len);
            await (this.f as fsPromises.FileHandle).read(temp, 0, len, p);
            uncompressedSize = lz4.decodeBlock(temp, out)
        }

        if (uncompressedSize !== out.length) {
            console.log(p, len, out.length, this.cache[p]);
            throw new Error(`uncompressedSize (${uncompressedSize}) !== this.buf.byteLength (${this.buf.byteLength})`)
        }

        let mainType = global[this.type];
        return new mainType(this.buf.buffer, this.buf.byteOffset, this.buf.byteLength/mainType.BYTES_PER_ELEMENT )
    }
    async forEachAsync(func: Function) {
        await this.fileOpen;

        for (let i = 0; i !== this.list.length; i++) {

            let out = await this.get(this.list[i]);

            tfCore.tidy(() => {
                func(tfCore.tensor(out));
            })
        }


    }
    async iterator() {

    }
}

let dataset = new Dataset(workerData);

if(!parentPort){
    throw new Error('parentPort not found')
}
parentPort.on('message', async ({id, op, data}: Message) => {
    if(!parentPort){
        throw new Error('parentPort not found')
    }
    try {
        switch (op) {
            case ('get'): {
                let dataInfo = dataset.list[data];
                if (!dataInfo) {
                    console.log('ddddd', dataset.list.length);
                    throw new Error(`data in ${data} index not found`)
                }

                let result = await dataset.get(dataInfo)

                parentPort.postMessage({ id, data: result });

                break;
            }
            case ('push'): {// push
                dataset.push(data);
                parentPort.postMessage({ id });
                break;
            }
            case ('writeHeaderFile'): {
                await dataset.sync();
                await dataset.writeHeaderFile();
                parentPort.postMessage({ id });
                break;
            }
            default: {
                throw new Error('unknown command: ' + op);
            }
        }
    } catch (error) {
        parentPort.postMessage({ id, error });
    }
});

