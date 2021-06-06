
let fsPromises = require('fs/promises')

let uniqueId = 0;

class Dataset {
    constructor(params={}){
        this.name = params.name||(uniqueId++)+'.bin'
        this.struct = params.struct;
        
        this.compressed = true;
        if(!Array.isArray(this.struct)){
            throw new Error('Struct field not array');
        }

        let len = 0;

        this.struct.forEach(shape=>{
            if(!Array.isArray(shape)){
                throw new Error(JSON.stringify(shape)+' shape is not number array');
            }
            shape.forEach(dim=>{
                if(typeof dim !=='number'){
                    throw new Error(JSON.stringify(shape)+' shape is not number array');
                }
                len+=dim|0
            })
        })
        this.buf = new Float32Array(len);
        this.f = fsPromises.open(this.name);
        let header = Buffer.concat([Buffer.from('ndl'),Buffer.from(new Uint32Array([this.compressed?1:0,0]))])
        this.f.write(header);
        
       
    }

    push(objs){
        let offset = 0;
        Object.keys(objs).forEach(key=>{
            let data = objs[key];
            if(!Array.isArray(data)){
                throw new Error(JSON.stringify(data)+key+' data is not array');
            }

            data.forEach(arr=>{
                if(!(data instanceof Float32Array)){
                    throw new Error(JSON.stringify(arr)+key+' data is not Float32Array');
                }
            })
            this.buf.set(arr, offset);
            offset+=arr.length;
        })
        
    }
    _writeFile(){
        this.f.write(header);
    }
}