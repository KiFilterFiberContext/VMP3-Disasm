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
		std::map<triton::arch::register_e, bool>  traced; // used for tainting registers since symbolic engine only works when calling `processing`

		while ( true )
		{
			if ( !api->isConcreteMemoryValueDefined( block_rva ) )
				return {};
			
			std::vector opcodes = api->getConcreteMemoryAreaValue( block_rva, 16 );
			x86_ins ins = x86_ins( block_rva, ( triton::uint8* ) opcodes.data(), opcodes.size() );
			
			api->disassembly( ins );
						
			// skip VMP instruction mutation
			if ( std::find( BLACKLIST.begin(), BLACKLIST.end(), ins.getType() ) != BLACKLIST.end() )
				goto next;
			
			// break at end of VM handler
			if ( ( ins.getType() == ID_INS_JMP && ins.operands[0].getType() == OP_REG ) || ins.getType() == ID_INS_RET )
			{
				ins_trace.push_back( ins );	
				break;
			}
			
			// whitelist certain instructions
			if ( ins.getType() == ID_INS_PUSH || ins.getType() == ID_INS_PUSHFD || ins.getType() == ID_INS_POP )
			{
				ins_trace.push_back( ins );	
				goto next;
			}
						
			// follow direct jumps to continue control flow i.e. jmp imm32
			// do not record jumps in cleaned stream
			if ( ( ins.getType() == ID_INS_JMP || ins.getType() == ID_INS_JA ) && 
			      ins.operands[0].getType() == OP_IMM )
			{
				block_rva = ins.operands[0].getImmediate().getValue();
				continue;
			}
			
			// all memory accesses should be relevant as in VMP they are only used to access VIP/VSP
			// start tracing memory read into register i.e. mov r64, [r64]
			if ( ins.operands[1].getType() == OP_MEM && ins.operands[0].getType() == OP_REG )
			{ 
				traced[ ins.operands[0].getRegister().getParent() ] = true;
				
				ins_trace.push_back( ins );	
				goto next;
			}
			
			// stop tracing register if memory write from register i.e. mov [r64], r64
			if ( ins.operands[0].getType() == OP_MEM && ins.operands[1].getType() == OP_REG )
			{
				traced[ ins.operands[1].getRegister().getParent() ] = false;
				
				ins_trace.push_back( ins );	
				goto next;
			}
			
			// record all arithmetic operations involving VSP or VIP
			if ( ( ins.getType() == ID_INS_SUB || ins.getType() == ID_INS_ADD ) && 
				  ins.operands[0].getType() == OP_REG &&
				  ( ins.operands[0].getRegister().getId() == vctx->vip_reg || ins.operands[0].getRegister().getId() == vctx->vsp_reg ) )
			{
				ins_trace.push_back( ins );	
				goto next;
			}
			
			// record all instructions containing traced and virtual registers (only destination)
			if ( ins.operands[0].getType() == OP_REG && traced[ ins.operands[0].getRegister().getParent() ] )
			{				
				ins_trace.push_back( ins );	
				goto next;		
			}
			
			// record vm key rolling -> xor r64, r64
			if ( ins.getType() == ID_INS_XOR && ins.operands[0].getType() == OP_REG && ins.operands[1].getType() == OP_REG )
			{
				if ( ins.operands[0].getRegister().getParent() == vctx->vrk_reg && traced[ ins.operands[1].getRegister().getParent() ] )
				{
					ins_trace.push_back( ins );	
					goto next;
				}
			}
			
			// record decryption and updating of VM handler RVA
			if ( ins.getType() == ID_INS_ADD && ins.operands[0].getType() == OP_REG && ins.operands[1].getType() == OP_REG )
			{
				if ( ins.operands[0].getRegister().getId() == vctx->jmp_rva_reg )
				{
					ins_trace.push_back( ins );	
					goto next;	
				}
			}
									
		next:
			block_rva = ins.getNextAddress();
		}
		
		return ins_trace;
	}
	
	void process( triton::API* api, vm_context* vctx, bool verbose )
	{
		if ( !vctx->vip )
			return;
		
		// get cleaned VM handler
		std::vector is = vmp::deobf( api, vctx, vctx->eip );
		if ( !is.size() )
			return;
		
		if ( verbose )
		{
			for ( auto& ins : is )
				std::cout << ins.getDisassembly() << std::endl;
		}
		
		// for ( auto& handler : vctx->handlers )
		// {
		//      if ( handler.rva == vctx->eip )
		// }
	
		// what to do after getting VM handler type?
		// probably setup breakpoints at certain mem access instructions to extract relevant values for the specific handler
		// function table one for each handler that decides how to process that handler and extract information
		
		
		
		return;
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
				{
					vctx->eip = (uint64_t) api->getConcreteRegisterValue( api->registers.x86_eip );	// jump to next handler
					break;
				}
				
				if ( !vctx->vip_reg && ins.isMemoryRead() )
				{
					// mov r64, [rsp+0x28] -> loads encrypted vip
					if ( ins.operands[0].getType() == OP_REG && 
						  ins.operands[1].getMemory().getBaseRegister().getId() == ID_REG_X86_ESP &&
						  ins.operands[1].getMemory().getDisplacement().getValue() == 0x28 )
					{
						std::cout << "VIP : " << ins.getDisassembly() << std::endl;
						vctx->vip_reg = ins.operands[0].getRegister().getId();
					}
				}
				
				// lea r64, [] -> loads encrypted jump RVA
				if ( !vctx->jmp_rva_reg && ins.getType() == ID_INS_LEA && ins.operands[0].getType() == OP_REG && ins.operands[1].getType() == OP_MEM && ins.operands[0].getRegister().getId() != vctx->vip_reg )
				{
					std::cout << "RVA : " << ins.getDisassembly() << std::endl;
					vctx->jmp_rva_reg = ins.operands[0].getRegister().getId();
				}
					
				if ( ins.getType() == ID_INS_MOV && 
					  ins.operands[0].getType() == OP_REG && 
					  ins.operands[1].getType() == OP_REG )
				{
					// mov r64, rsp -> loads virtual stack pointer
					if ( !vctx->vsp_reg && ins.operands[1].getRegister().getId() == ID_REG_X86_ESP )
					{
						std::cout << "VSP : " << ins.getDisassembly() << std::endl;
						vctx->vsp_reg = ins.operands[0].getRegister().getId();
					}
					
					if ( !vctx->vrk_reg && vctx->vip_reg && ins.operands[1].getRegister().getId() == vctx->vip_reg )
					{
						std::cout << "VRK : " << ins.getDisassembly() << std::endl;
						vctx->vrk_reg = ins.operands[0].getRegister().getParent();
					}
				}
				
				// update EIP 
				vctx->eip = (uint64_t) api->getConcreteRegisterValue( api->registers.x86_eip );
			}
			
			// check if VM context is initialized
			if ( vctx->vip_reg && vctx->vsp_reg && vctx->vrk_reg && vctx->jmp_rva_reg )
				vctx->vip = (uint64_t) api->getConcreteRegisterValue( api->getRegister( vctx->vip_reg ) );
		}
	}
	
	
}