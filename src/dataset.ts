import * as tfCore from '@tensorflow/tfjs-core'
import * as path from 'path'
import * as fs from 'fs'
import * as lz4 from 'lz4'
import {
    Worker, isMainThread, parentPort, workerData
} from 'worker_threads'
import { TypedArray } from '@tensorflow/tfjs-core';
import cloneDeep from 'lodash.clonedeep';

let uniqueId = 0;
export const TYPEIDS = {
    'Float32Array': 1,
    'Int32Array': 2,
    'Uint8Array': 3,
    'Uint32Array': 4,
    'Buffer': 5,
    'Int8Array': 6,
    'Uint16Array': 7,
    'Int16Array': 8,
}

export const ReTYPEIDS = [null, Float32Array, Int32Array, Uint8Array, Uint32Array, Buffer, Int8Array, Uint16Array, Int16Array];

export let acceptableTypedArrays = Object.keys(TYPEIDS);
export const TFTypes = {
    float32: 1,
    int32: 2,
    bool: 3,
    complex64: 4,
    string: 5
}
export const DataTypeMap = {
    float32: Float32Array,
    int32: Int32Array,
    bool: Uint8Array,
    complex64: Float32Array,
    string: String
}

export type Dtype = 'Float32Array' | 'Uint8Array' | 'Buffer' | 'Int8Array' | 'Uint16Array' | 'Int16Array' | 'Uint32Array' | 'Int32Array'
export interface WorkerData {
    name: string,
    shape?: number[],
    compressLevel?: number,
    count?: number,
    needCreate: boolean
}

interface Prefetch {
    i: number
    promise: Promise<TypedArray>
    data?: TypedArray
}
interface Prefetches {
    [thingName: string]: Prefetch[]
}

interface ZipResult {
    [thingName: string]: tfCore.Tensor
}

interface Datasets {
    [thingName: string]: Dataset
}
type DatasetWorker = Worker & { dataI: number }

interface ZipTypedArrays {
    [thingName: string]: TypedArray
}

export let parseHeaderBuffer = (name: string, header: Buffer) => {

    let magic = Buffer.from('ndlt')
    if (Buffer.compare(header.slice(0, magic.length), magic) !== 0) {
        throw new Error(`${name + '.ndlt'} is not header file`)
    }
    // 8 dim shape
    let shape = Array.from(new Uint32Array(header.buffer, header.byteOffset + magic.length, 8))
    shape = shape.filter(v => v)

    let compressLevel = header[4 + 8 * Uint32Array.BYTES_PER_ELEMENT]

    let count = Number((new BigUint64Array(header.buffer, header.byteOffset + 4 + 8 * Uint32Array.BYTES_PER_ELEMENT + 1 + 1 + 2, 1))[0])

    let params: WorkerData = {
        name,
        shape,
        compressLevel,
        count,
        needCreate: false
    }
    let list: BigUint64Array[] = [];
    let headLength = 4 + 8 * Uint32Array.BYTES_PER_ELEMENT + 1 + 1 + 2 + 8;
    if (header.length !== headLength) {
        let listArray = new BigUint64Array(count * 7)
        let decodedLength = lz4.decodeBlock(header.subarray(headLength), Buffer.from(listArray.buffer, listArray.byteOffset, listArray.byteLength))
        if (decodedLength !== count * 7 * 8) {
            throw new Error('It is not possible to decompress the data')
        }
        for (let i = 0; i !== count; i++) {
            list.push(listArray.subarray(i * 7, i * 7 + 7))
        }
    }
    return { params, list };
}
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
export class Dataset {
    private name: string
    dataFile: string
    shape: number[]
    private worker: DatasetWorker
    list: number[]
    inputSize = 1;
    initializing: Promise<void>
    needCreate: boolean;
    constructor(params: WorkerData) {
        let workerData: WorkerData = params || {};
        workerData.name = params.name || String(uniqueId++);
        this.name = workerData.name;
        this.dataFile = (this.name) + '.bin'
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
        })

        this.shape = workerData.shape = shape;


        workerData.compressLevel = typeof params.compressLevel === 'number' ? params.compressLevel : 1;

        this.worker = new Worker(path.resolve(__dirname, 'dataset_worker.js'), {
            workerData: workerData
        }) as DatasetWorker;
        this.worker.dataI = 0;
        this.list = [];
        if (params.count) {
            for (let i = 0; i !== params.count; i++) {
                this.list.push(this.worker.dataI++);
            }
        }
        this.initializing = new Promise((res, rej) => {
            let onMessage = (message: any) => {
                if (message.error) {
                    rej(message.error);
                }
                if (message.id === -1) {
                    this.worker.off('message', onMessage);
                    res();
                }
            };
            this.worker.on('message', onMessage);
        })
        this.generator = this.generator.bind(this)
    }
    private _clone() {
        return Object.assign(Object.create(Object.getPrototypeOf(this)), this)
    }
    private async _sendCommand(op: string, data?: any, transferableObjects?: any[]): Promise<any> {
        let commandId = uniqueCommandId++;
        this.worker.postMessage({ op, id: commandId, data }, transferableObjects);
        return new Promise((res, rej) => {
            let onMessage = (message: any) => {
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
    push(objs: TypedArray) {

        if (acceptableTypedArrays.indexOf(objs.constructor.name) < 0) {
            throw new Error(`Input object ${objs.constructor.name} type is not accept`)
        }
        if (objs.length !== this.inputSize) {
            throw new Error(`Input size have ${objs.length}. expected ${this.inputSize}`)
        }
        this.list.push(this.worker.dataI++);
        let clone = objs.slice(0);
        return this._sendCommand('push', clone, [clone.buffer]);
    }

    repeat(count: number) {
        let newDataset = this._clone() as Dataset
        let res: number[] = [];
        for (let i = 0; i < count; i++) {
            res = res.concat(newDataset.list)
        }
        newDataset.list = res;

        return newDataset;
    }
    skip(count: number) {
        let newDataset = cloneDeep(this) as Dataset

        newDataset.list = newDataset.list.slice(count)

        return newDataset;
    }
    take(count: number) {
        let newDataset = cloneDeep(this) as Dataset
        newDataset.list = newDataset.list.slice(0, count);
        return newDataset;
    }
    async *generator() {
        for (let i = 0; i !== this.list.length; i++) {
            let out = await this.get(this.list[i]);
            yield tfCore.tensor(out, [1].concat(this.shape));

        }
    }

    send(objs: TypedArray) {
        if (acceptableTypedArrays.indexOf(objs.constructor.name) < 0) {
            throw new Error(`Input object ${objs.constructor.name} type is not accept`)
        }
        if (objs.length !== this.inputSize) {
            throw new Error(`Input size have ${objs.length}. expected ${this.inputSize}`)
        }
        this.list.push(this.worker.dataI++);

        return this._sendCommand('push', objs, [objs.buffer]);
    }
    async get(i: number): Promise<TypedArray> {

        let index = this.list[i];

        return this._sendCommand('get', index);
    }
    async forEachAsync(func: Function) {

        for (let i = 0; i !== this.list.length; i++) {

            let out = await this.get(this.list[i]);
            tfCore.tidy(() => {
                func(tfCore.tensor(out, [1].concat(this.shape)));
            })
        }

    }
    async writeHeaderFile() {
        let commandId = uniqueCommandId++;
        this.worker.postMessage({ id: commandId, op: 'writeHeaderFile' })
        let result = await new Promise((res, rej) => {
            let onMessage = (message: any) => {
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
    destroy(deleteCacheFile: boolean) {
        this.worker.terminate();
        if (deleteCacheFile) {
            fs.unlinkSync(this.dataFile);
        }
    }
}
export async function openDataset(name: string): Promise<Dataset> {

    let f = await fs.promises.open(name + '.ndlt', 'r');

    //magic + shapeinfo+ compressioninfo + typeinfo + other+ count
    let len = 4 + 8 * Uint32Array.BYTES_PER_ELEMENT + 1 + 1 + 2 + 8
    let header = Buffer.alloc(len)
    await f.read(header, 0, len, 0);
    await f.close();

    let { params } = parseHeaderBuffer(name, header);

    let dataset = new Dataset(params)
    await dataset.initializing;
    return dataset;
}

