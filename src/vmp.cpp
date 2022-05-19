#include <vmp.h>
#include <algorithm>

using namespace triton;
using namespace triton::arch;
using namespace triton::arch::x86;

// add reg, 4 == lea reg, [reg+4]
// /devirt ../virt-32-basic.exe -entry 0x951e7

namespace vmp
{
	std::vector<x86_ins> deobf( triton::API* api, vm_context* vctx, uint64_t block_rva )
	{
		std::vector<x86_ins> ins_trace;
		std::map<triton::arch::register_e, bool>  traced;

		while ( true )
		{
			if ( !api->isConcreteMemoryValueDefined( block_rva ) )
				return {};
			
			std::vector opcodes = api->getConcreteMemoryAreaValue( block_rva, 16 );
			x86_ins ins = x86_ins( block_rva, ( triton::uint8* ) opcodes.data(), opcodes.size() );
			
			api->disassembly( ins );
			
			if ( std::find( BLACKLIST.begin(), BLACKLIST.end(), ins.getType() ) != BLACKLIST.end() )
			{				
				block_rva = ins.getNextAddress();
				continue;
			}
			
			if ( ( ins.getType() == ID_INS_JMP || ins.getType() == ID_INS_JA ) && 
			      ins.operands[0].getType() == OP_IMM )
			{
				block_rva = ins.operands[0].getImmediate().getValue();
				continue;
			}
			
			ins_trace.push_back( ins );	
			if ( ( ins.getType() == ID_INS_JMP && ins.operands[0].getType() == OP_REG ) || ins.getType() == ID_INS_RET )
				break;
						
			block_rva = ins.getNextAddress();
		}
		
		return ins_trace;
	}
	
	void init( triton::API* api, vm_context* vctx )
	{
		if ( vctx->vip )
			return;
				
		// check for push r64; call imm64; (VMENTER)
		std::vector ins = api->disassembly( vctx->eip, 2 );
		if ( ins[0].getType() == ID_INS_PUSH && ins[0].operands[0].getType() == OP_IMM && 
			  ins[1].getType() == ID_INS_CALL && ins[1].operands[0].getType() == OP_IMM )
		{
			// push vm handler info for identification
			vctx->handlers.push_back( { 
				.type = arch::VM_INIT,
				.rva = (uint64_t) ins[1].operands[0].getImmediate().getValue()
			} );
			
			// emulate VMINIT until end of VM handler
			while ( true )
			{
				if ( !api->isConcreteMemoryValueDefined( vctx->eip ) )
					break;
			
				std::vector opcodes = api->getConcreteMemoryAreaValue( vctx->eip, 16 );
				x86_ins ins = x86_ins( vctx->eip, ( triton::uint8* ) opcodes.data(), opcodes.size() );
				
				// processing instruction updates Triton symbolic engine
				api->processing( ins );
							
				// break when we reach end of control flow
				if ( ( ins.getType() == ID_INS_JMP && ins.operands[0].getType() == OP_REG ) || ins.getType() == ID_INS_RET )
					break;
				
				if ( ins.isMemoryRead() )
				{
					// mov r64, [rsp+0x28] -> loads encrypted vip
					if ( ins.operands[0].getType() == OP_REG && 
						  ins.operands[1].getMemory().getBaseRegister().getId() == ID_REG_X86_ESP &&
						  ins.operands[1].getMemory().getDisplacement().getValue() == 0x28 )
					{
						vctx->vip_reg = ins.operands[0].getRegister().getId();
					}
					
					// mov r64, [vip]; add vip, 4 -> loads encrypted jump RVA
					if ( vctx->vip_reg && ins.operands[1].getMemory().getBaseRegister().getId() == vctx->vip_reg )
						vctx->jmp_rva_reg = ins.operands[0].getRegister().getId();
				}
				
				if ( ins.getType() == ID_INS_MOV && 
					  ins.operands[0].getType() == OP_REG && 
					  ins.operands[1].getType() == OP_REG )
				{
					// mov r64, rsp -> loads virtual stack pointer
					if ( ins.operands[1].getRegister().getId() == ID_REG_X86_ESP )
						vctx->vsp_reg = ins.operands[0].getRegister().getId();
				}
				
				if ( ins.getType() == ID_INS_XOR &&
					  ins.operands[0].getType() == OP_REG &&
					  ins.operands[1].getType() == OP_REG )
				{
					// xor r64, r64 -> updates vm rolling key 
					if ( vctx->jmp_rva_reg && ins.operands[1].getRegister().getId() == vctx->jmp_rva_reg )
						vctx->vrk_reg = ins.operands[0].getRegister().getId();
				}
				
				// update EIP 
				vctx->eip = (uint64_t) api->getConcreteRegisterValue( api->registers.x86_eip );
			}
			
			// check if VM context is initialized
			if ( vctx->vip_reg && vctx->vsp_reg && vctx->vrk_reg )
				vctx->vip = (uint64_t) api->getConcreteRegisterValue( api->getRegister( vctx->vip_reg ) );
		}
	}
}