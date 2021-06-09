
let fsPromises = require('fs').promises
let fs = require('fs')
let tfData = require('@tensorflow/tfjs-data');
let tfCore = require('@tensorflow/tfjs-core');
let lz4 = require('lz4');
const {
    Worker, isMainThread, parentPort, workerData
} = require('worker_threads');

let uniqueId = 0;

const FLOAT32TYPE = 2;

if(isMainThread){
 
let uniqueCommandId =0;
class Dataset extends tfData.Dataset {
    constructor(params={}){
        super();
        let workerData = params;
        workerData.name = params.name||(uniqueId++);
        workerData.dataFile = (this.name)+'.bin'
        this.shape = workerData.shape = params.shape;
        
        workerData.compressLevel= typeof params.compressLevel ==='number'?params.compressLevel:1;

        this.worker = new Worker(__filename, {
            workerData: workerData
          });
        this.worker.dataI =0;
        this.list=[];
    }
    push(objs){
        this.list.push(this.worker.dataI++);
        let clone = objs.slice(0);
        this.worker.postMessage({op:'push',data:clone},[clone.buffer])
    }
    send(objs){
        this.list.push(this.worker.dataI++);
        this.worker.postMessage({op:'push',data:objs},[objs.buffer])
    }
    async get(i){

        let index = this.list[i];
        let commandId = uniqueCommandId++;
        this.worker.postMessage({index,id:commandId,op:'get'})
        let result = await new Promise((res, rej)=>{
            let onMessage = (message)=>{
                if(message.error){
                    rej(message.error)
                }
                if(message.id === commandId){
                    this.worker.off('message', onMessage)
                    res(message.data)
                }
            }
            this.worker.on('message', onMessage)

        })
        return result;
    }
    async forEachAsync(func){
        await this.fileOpen;

        for(let i=0;i!== this.list.length;i++){

            let out = await this.get(this.list[i]);
            
            tfCore.tidy(()=>{
                func(tfCore.tensor(out));
            })
        }
        
    }
    async writeHeaderFile(){
        let commandId = uniqueCommandId++;
        this.worker.postMessage({id:commandId,op:'writeHeaderFile'})
        let result = await new Promise((res, rej)=>{
            let onMessage = (message)=>{
                if(message.error){
                    rej(message.error)
                }
                if(message.id === commandId){
                    this.worker.off('message', onMessage)
                    res(message.data)
                }
            }
            this.worker.on('message', onMessage)

        })
        return result;
    }
    destroy(){
        this.worker.terminate();
    }
}
class Zip {
    constructor(datasets){
        
        this.datasets = datasets;
        this.firstDataset = Object.values(datasets)[0]
        this.keys = Object.keys(datasets);
        this.tensorBuffers = {};

        this.prefetches ={};
        this.keys.forEach(key=>{
            this.prefetches[key] = [];
            this.tensorBuffers[key] = tfCore.buffer([1].concat(this.datasets[key].shape));
        })
    }
    batch (){throw new Error('not implemented')}
    concatenate (){throw new Error('not implemented')}
    filter (){throw new Error('not implemented')}
    forEachAsync (){throw new Error('not implemented')}
    map (){throw new Error('not implemented')}
    mapAsync (){throw new Error('not implemented')}
    prefetch (){throw new Error('not implemented')}
    repeat (){throw new Error('not implemented')}
    shuffle (){throw new Error('not implemented')}
    skip (){throw new Error('not implemented')}
    take (){throw new Error('not implemented')}
    toArray (){throw new Error('not implemented')}
    async forEachAsync(func){
        let list = this.firstDataset.list;
        for(let i=0;i!== list.length;i++){
            let out = {}

            for(let k=0;k!== this.keys.length;k++){
                let key = this.keys[k];
                out[key] = await this.datasets[key].get(this.datasets[key].list[i]);
            }
            
            tfCore.tidy(()=>{
                for(let k=0;k!== this.keys.length;k++){
                    let key = this.keys[k];
                    out[key] = tfCore.tensor(out[key],[1, out[key].length]);
                }
                func(out);
            })
        }
    }
    async iterator(){
        let i=0;
        let count = this.firstDataset.list.length;

        return {
            next: async ()=>{
                if(i%256===0){
                    console.log('i',i);
                }
                if(count===i){
                    return {value: null,done:true};
                }
                i++;
                let out = {}

                let getingDatas = this.keys.map(async key=>{

                    //check prefetch data
                    let prefetchLine = this.prefetches[key];
                    let prefetch = prefetchLine.find(prefetch=>prefetch.i === this.datasets[key].list[i])
                    
                    if(prefetch){
                        if(prefetch.data){
                            out[key] = prefetch.data;
                        }else{
                            out[key] = await prefetch.promise;
                        }
                    }else{
                            out[key] = await this.datasets[key].get(this.datasets[key].list[i]);
                    }
                })
                await Promise.all(getingDatas);

                const prefetchSize = 9;
                //start prefetch for all datasets
                for(let k=0;k!== this.keys.length;k++){
                    let key = this.keys[k];

                    let ids = this.datasets[key].list.slice(i+1,i+1+prefetchSize);
                    
                    this.prefetches[key] = this.prefetches[key].filter(prefetch=>{
                        let prefetchExist = ids.findIndex(id=>id===prefetch.i);
                        if(prefetchExist>-1){
                            ids[prefetchExist]=-1;
                        }
                        return prefetchExist>-1;
                    })
                    ids=ids.filter(id=>id!==-1)

                    for(let ni=0;ni!==ids.length;ni++){
                        let i = ids[ni];
                        let prefetch = {}
                        let promise = this.datasets[key].get(i)
                        .then(data=>{
                            return prefetch.data=data;
                        })
                        prefetch.promise = promise;
                        prefetch.i = i;

                        this.prefetches[key].push(prefetch)
                    }
                    
                }

                for(let k=0;k!== this.keys.length;k++){
                    let key = this.keys[k];
                    this.tensorBuffers[key].values = out[key]
                    out[key] = this.tensorBuffers[key].toTensor();
                    
                }
                
                

                return {value:out,done:false};
            }
        }
    }
}
function zip(objects){

    return new Zip(objects);
}


module.exports.zip = zip;
module.exports.Dataset = Dataset;
}else{
//const {performance} = require('perf_hooks');
//      const inspector = require('inspector');
    
//  const session = new inspector.Session();


//  setTimeout(()=>{
//     const fd = fs.openSync('profile.heapsnapshot', 'w');

//     session.connect();
    
//     session.on('HeapProfiler.addHeapSnapshotChunk', (m) => {
//       fs.writeSync(fd, m.params.chunk);
//     });
    
//     session.post('HeapProfiler.takeHeapSnapshot', null, (err, r) => {
//       console.log('HeapProfiler.takeHeapSnapshot done:', err, r);
//       session.disconnect();
//       fs.closeSync(fd);
//     });
//  },10000)


// rss: 2123,
// heapTotal: 9,
// heapUsed: 6,
// external: 1635,
// arrayBuffers: 1630


// rss: 101,
// heapTotal: 7,
// heapUsed: 5,
// external: 21,
// arrayBuffers: 20

// session.post('Profiler.enable', () => {
//     session.post('Profiler.start', () => {})
// })
class Dataset {
    constructor(params={}){
        this.name = params.name||(uniqueId++);
        this.dataFile = (this.name)+'.bin'
        this.shape = params.shape;
        
        this.compressLevel= typeof params.compressLevel ==='number'?params.compressLevel:1;

        this.f = null;// file with dataset

        this._p = 0;//file position
        this._computeStrict();

        this.fileOpen = fsPromises.open(this.dataFile, 'a+').then((f)=>{
            this.f = f;
            //clear file
            return this.f.truncate(0)
        });

        this.list = [];
        
        this.cache={};
       
        this.writeQueue=[];
        this.writing = false;
    }
    async writeHeaderFile (){

        let headerFile = (this.name)+'.ndlt'
        let f = fs.createWriteStream(headerFile);

        let magicSymbol = Buffer.from('ndlt')
        let shapeSerialized = Buffer.from(JSON.stringify(this.shape));
        let header = Buffer.concat([magicSymbol,Buffer.from(new Uint32Array([shapeSerialized.length])),shapeSerialized, Buffer.from([this.compressLevel,0,0,0,0])]);
        f.write(header);

        for(let i=0;i!==this.list.length;i++){
            let dataInfo = this.list[i]
            f.write(Buffer.from(dataInfo.buffer,dataInfo.byteOffset,dataInfo.byteLength))
        }
        f.end();
        await new Promise((res)=>{
            f.on('finish',res)
        })
        
    }
    _computeStrict(){
        if(!Array.isArray(this.shape)){
            throw new Error('Shape field not array');
        }

        let len = 0;

        let shape = this.shape
        if(!Array.isArray(shape)){
            throw new Error(JSON.stringify(shape)+' shape is not number array');
        }
        shape.forEach(dim=>{
            if(typeof dim !=='number'){
                throw new Error(JSON.stringify(shape)+' shape is not number array');
            }
            len+=dim|0
        })
        
        this.buf = new Float32Array(len);
        this.compressBuf = Buffer.alloc(lz4.encodeBound(len*4*1.5));//Buffer.alloc(lz4.encodeBound(len*4));
        this.decompressBuf = Buffer.alloc(this.compressBuf.length);
    }

    batch (){throw new Error('not implemented')}
    concatenate (){throw new Error('not implemented')}
    filter (){throw new Error('not implemented')}
    forEachAsync (){throw new Error('not implemented')}
    map (){throw new Error('not implemented')}
    mapAsync (){throw new Error('not implemented')}
    prefetch (){throw new Error('not implemented')}
    repeat (){throw new Error('not implemented')}
    shuffle (){throw new Error('not implemented')}
    skip (){throw new Error('not implemented')}
    take (){throw new Error('not implemented')}
    toArray (){throw new Error('not implemented')}
    async push(objs){
        if(!objs){
            throw new Error('Data not found')
        }

        let data = objs;

        let  length = lz4.encodeBlock(Buffer.from(data.buffer,data.byteOffset, data.byteLength), this.compressBuf)

        if(length>this.compressBuf.length){
            throw new Error(`compressBuf.length (${this.compressBuf.length})<compressedSize (${length})`)
        }

        let writeData = Uint8Array.prototype.slice.call(this.compressBuf, 0, length)

        let p = this._p;
        this._p+=length;

        let dataInfo = new BigUint64Array([BigInt(p), BigInt(length), BigInt(FLOAT32TYPE), 0n]);
        
        this.list.push(dataInfo);



        this.cache[p]={writeData}
        this.writeQueue.push({data:writeData, write:async()=>{
            this.writing = true;
            await this.f.write(writeData, 0, length, p);
            delete this.cache[p];
            this.writing = false;

        }})


        await this.fileOpen;
        //TODO batch write
        if(true){
            if(!this.writing){
                while(this.writeQueue.length){
                    await this.writeQueue[0].write();
                    this.writeQueue.shift();
                }

            }
        }else{
            let writeQueueLength = 0;
            
            this.writeQueue.forEach((data)=>{
                writeQueueLength+=data.length;
            })
            console.log('writeQueueLength: ', writeQueueLength);
            if((writeQueueLength+writeData.length)>(1024*1024*4)){
                await this.fileOpen;
                let writing = this.f.write(Buffer.concat(this.writeQueue.concat([writeData])), 0, writeQueueLength+writeData.length, p)//TODO, //TODO);
                await writing;
                Object.keys(this.cache).forEach((key)=>{
                    delete this.cache[key];
                })
                this.cache={};
                this.writeQueue=[];
            }else{
                this.writeQueue.push(writeData)
            }
        }
      


    }
    async sync(){

        await this.fileOpen;
        let promises = Object.keys(this.cache).map(_p=>{
            let p = Number(_p);
            let data = this.cache[_p].writeData
            return this.f.write(data, 0, data.length, p);
        })
        await Promise.all(promises)
        this.cache={};
    }
    async get(dataInfo){
        
        let p = Number(dataInfo[0]);
        let len = Number(dataInfo[1]);
        let out = Buffer.from(this.buf.buffer, this.buf.byteOffset, this.buf.byteLength);
        let uncompressedSize = 0;
        if(this.cache[p]){
            let writeData = this.cache[p].writeData;
            uncompressedSize = lz4.decodeBlock(writeData, out)
        }else{
            let temp = Buffer.allocUnsafe(len);
            await this.f.read(temp, 0, len, p);
            uncompressedSize = lz4.decodeBlock(temp, out)
        }

        if(uncompressedSize !== out.length){
            console.log(p, len, out.length, this.cache[p]);
            throw new Error(`uncompressedSize (${uncompressedSize}) !== this.buf.byteLength (${this.buf.byteLength})`)
        }

        return this.buf
    }
    async forEachAsync(func){
        await this.fileOpen;

        for(let i=0;i!== this.list.length;i++){

            let out = await this.get(this.list[i]);
            
            tfCore.tidy(()=>{
                func(tfCore.tensor(out));
            })
        }
        

    }
    async iterator(){

    }
    _writeFile(){
        this.f.write(header);
    }
}

let dataset = new Dataset(workerData);


parentPort.on('message', async (message) => {
    try{
        switch(message.op){
            case('get'):{
                let dataInfo = dataset.list[message.index];
                if(!dataInfo){
                    console.log('ddddd',dataset.list.length);
                    throw new Error(`data in ${message.index} index not found`)
                }
                
                let data = await dataset.get(dataInfo)

                parentPort.postMessage({id:message.id,data});
                
            break;
            }
            case('push'):{// push
                dataset.push(message.data);
                break;
            }
            case('writeHeaderFile'):{
                await dataset.sync();
                await dataset.writeHeaderFile();
                parentPort.postMessage({id:message.id});
                break;
            }
            default:{
                throw new Error('unknown command: '+message.op);
            }
        }
    }catch(error){
        parentPort.postMessage({id:message.id, error});
    }
  });
}

