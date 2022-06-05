#pragma once

#include <triton/x8664Cpu.hpp>
#include <triton/x86Cpu.hpp>
#include <triton/x86Specifications.hpp>

#include <triton/tritonToLLVM.hpp>
#include <triton/llvmToTriton.hpp>

namespace vmp::arch
{	
	// represents VM instruction type 
	enum handler_t
	{
		VM_INIT,
		VM_POPV,
		VM_PUSHC,
		VM_UNK
	};
	
	// IR representation for VM instruction 
	// we lift every VM handler block to this
	struct vm_ins_t
	{
		uint64_t rva;
		handler_t type;
		
		
		
		vm_ins_t( uint64_t handler_va, handler_t handler_type ) : rva( handler_va ), type( handler_type ) {};
	};
}


// VMProtect 3 is a stack-machine so all results and flags are written onto the stack and operations (i.e. add/mul) are performed on the virtual "registers" (relative to RSP)

/*
RVA [VIP+N => 0xABCD] : VPOPV V[stack_offset] ; V[stack_offset] => VALUE ; VSP => VALUE; VRK => VALUE


mov edi, dword ptr [vsp]
add vsp, 4
movzx eax, byte ptr [vip]
add vip, 1
... 
mov dword ptr [esp+eax], edi 

type: VM_POPV 
rva: 0xDEADBEEF

*/

/*
VPUSHC [constant]
*/