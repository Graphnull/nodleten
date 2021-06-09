declare module 'lz4' {


declare function encodeBound(len:number): number

function encodeBlock(inp:Buffer, out:Buffer):number
function decodeBlock(inp:Buffer, out:Buffer):number
}