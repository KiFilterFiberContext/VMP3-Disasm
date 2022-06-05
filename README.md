# VMProtect Disassembler
**WIP** Disassembler for x86 binaries virtualized by VMProtect 3

## How?
Unlike [NoVMP](https://github.com/can1357/NoVmp/), this project attempts to lift VMProtect's virtual machine handlers into IL purely through instruction emulation using [Triton](https://github.com/JonathanSalwan/Triton).  The procedure starts by emulating the VMINIT and recording the initial VM state, then proceeding to use this information to deobfuscate the rest of the VM handlers by tainting only relevant VM registers.  The simplified VM handlers are converted to VM instructions by pattern matching certain instruction sequences then emulated to extract runtime information that is logged by the disassembler.  

## TODO
- [x] Handle VMINIT and VMEXIT handlers
- [x] Implement initial VM handler deobfuscator
- [ ] Handle PE/ELF relocations and imports 
- [ ] Implement VM IL semantics to represent VM instruction
- [ ] Implement VM IL optimizer (stuff like constant propagation, etc.)
- [ ] Finish lifter for remaining VM handlers (VMPUSH, VMADD, VMDIV, etc.)
- [ ] Handle virtual conditionals (likely to use Triton DSE)
- [ ] Handle VM context swapping 

## References
- https://back.engineering/17/05/2021/
- https://whereisr0da.github.io/blog/posts/2021-02-16-vmp-3/
- most of https://forum.tuts4you.com

