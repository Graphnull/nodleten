import * as tfData from '@tensorflow/tfjs-data'
import * as tfCore from '@tensorflow/tfjs-core'
import * as path from 'path'
import {
    Worker, isMainThread, parentPort, workerData
} from 'worker_threads'
import { TypedArray } from '@tensorflow/tfjs-core';

let uniqueId = 0;

const FLOAT32TYPE = 2;

interface WorkerData {
    name:string,
    dataFile:string,
    shape:string,
    compressLevel:number,
}

interface Prefetch {
    i:number
    promise:Promise<TypedArray>
    data?:TypedArray
}
interface Prefetches {
    [thingName: string]: Prefetch[]
  }

interface ZipResult {
    [thingName: string]: tfCore.Tensor
  }

interface TensorBuffers {
    [thingName: string]: tfCore.TensorBuffer<any, any>
  }

  interface Datasets {
    [thingName: string]: Dataset
  }
type DatasetWorker = Worker & {dataI: number}

interface ZipTypedArrays {
    [thingName: string]: TypedArray
}

let uniqueCommandId = 0;
export class Dataset {
    name:string
    shape:number[]
    worker: DatasetWorker
    list: number[]
    constructor(params:any = {}) {
        let workerData: WorkerData = params;
        workerData.name = params.name || (uniqueId++);
        this.name = workerData.name;
        this.shape = workerData.shape = params.shape;

        workerData.compressLevel = typeof params.compressLevel === 'number' ? params.compressLevel : 1;

        this.worker = new Worker(path.resolve(__dirname,'dataset_worker.js'), {
            workerData: workerData
        }) as DatasetWorker;
        this.worker.dataI = 0;
        this.list = [];
    }
    push(objs: TypedArray) {
        this.list.push(this.worker.dataI++);
        let clone = objs.slice(0);
        this.worker.postMessage({ op: 'push', data: clone }, [clone.buffer])
    }
    send(objs: TypedArray) {
        this.list.push(this.worker.dataI++);
        this.worker.postMessage({ op: 'push', data: objs }, [objs.buffer])
    }
    async get(i: number): Promise<TypedArray> {

        let index = this.list[i];
        let commandId = uniqueCommandId++;
        this.worker.postMessage({ index, id: commandId, op: 'get' })
        return await new Promise((res, rej) => {
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
    }
    async forEachAsync(func: any) {
        //TODO

        for (let i = 0; i !== this.list.length; i++) {

            let out = await this.get(this.list[i]);

            throw new Error('not implemented')
            // tfCore.tidy(() => {
            //     func(tfCore.tensor(out));
            // })
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
    destroy() {
        this.worker.terminate();
    }
}


class Zip {
    datasets:Datasets
    firstDataset: Dataset
    keys: string[]
    tensorBuffers: TensorBuffers
    prefetches:Prefetches
    constructor(datasets: Datasets) {

        this.datasets = datasets;
        this.firstDataset = Object.values(datasets)[0];
        if(!this.firstDataset){
            throw new Error('dataset not found')
        }
        this.keys = Object.keys(datasets);
        this.tensorBuffers = {};
        this.prefetches = {};

        this.keys.forEach(key => {
            this.prefetches[key] = [];
            let dataset = this.datasets[key]
            this.tensorBuffers[key] = tfCore.buffer([1].concat( dataset.shape ));
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
    async forEachAsync(func: Function) {
        let list = this.firstDataset.list;
        for (let i = 0; i !== list.length; i++) {

            let typedArrays: ZipTypedArrays = {};
            for (let k = 0; k !== this.keys.length; k++) {
                let key = this.keys[k];
                typedArrays[key] = await this.datasets[key].get(this.datasets[key].list[i]);
            }
            let out:ZipResult = {}

            tfCore.tidy(() => {
                for (let k = 0; k !== this.keys.length; k++) {
                    let key = this.keys[k];
                    out[key] = tfCore.tensor(typedArrays[key], [1, typedArrays[key].length]);
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
                let results:ZipTypedArrays = {}
                let getingDatas = this.keys.map(async key => {

                    //check prefetch data
                    let prefetchLine = this.prefetches[key];
                    let prefetch = prefetchLine.find(prefetch => prefetch.i === this.datasets[key].list[i])

                    if (prefetch) {
                        if (prefetch.data) {
                            results[key] = prefetch.data;
                        } else {
                            results[key] = await prefetch.promise;
                        }
                    } else {
                        results[key] = await this.datasets[key].get(this.datasets[key].list[i]);
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
                        let prefetch:Prefetch = {
                            i,
                            promise:this.datasets[key].get(i)
                            .then(data => {
                                return prefetch.data = data;
                            })
                        }
                        this.prefetches[key].push(prefetch)
                    }

                }

                let out:ZipResult = {}
                for (let k = 0; k !== this.keys.length; k++) {
                    let key = this.keys[k];
                    this.tensorBuffers[key].values = results[key]
                    out[key] = this.tensorBuffers[key].toTensor();

                }



                return { value: out, done: false };
            }
        }
    }
}
export function zip(objects: any) {

    return new Zip(objects);
}

