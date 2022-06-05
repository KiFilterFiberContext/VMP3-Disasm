#include <vmp.h>
#include <algorithm>
#include <ranges>

using namespace triton;
using namespace triton::arch;
using namespace triton::arch::x86;

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
			if ( ( ins.getType() == ID_INS_SUB || ins.getType() == ID_INS_ADD || ins.getType() == ID_INS_MOV ) && 
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
	
	arch::handler_t identify( const vm_context* vctx, const std::vector<x86_ins>& cleaned )
	{
		for ( auto& handler : vctx->handlers ) 
		{
			if ( handler.rva == vctx->eip )
				return handler.type;
		}
		
		std::vector<x86_ins> is;
		for ( auto ins : cleaned )
		{
			if ( ins.getType() == ID_INS_POP || ins.getType() == ID_INS_PUSHFD )
			{
				is.push_back( ins );
				continue;
			}
			
			if ( ins.operands.size() < 2 )
				continue;
			
			if ( ins.operands[1].getType() == OP_MEM || ins.operands[0].getType() == OP_MEM )
			{
				is.push_back( ins );
				continue;
			}
			
			if ( ins.operands[0].getType() == OP_REG && ins.operands[0].getRegister().getId() == vctx->vsp_reg )
			{
				is.push_back( ins );
				continue;
			}
		}
		
		// pattern match the VM handers
		
		// mov r64, [vsp + imm32]
		auto is_read_vsp = []( const vm_context* vctx, x86_ins ins, size_t disp = 0 ) -> bool {
			return ( ins.getType() == ID_INS_MOV || ins.getType() == ID_INS_MOVZX ) &&
			   ins.operands[1].getType() == OP_MEM &&
			   ins.operands[1].getMemory().getBaseRegister().getId() == vctx->vsp_reg &&
			   ins.operands[1].getMemory().getDisplacement().getValue() == disp;
		};
		
		// mov [vsp + imm32], r64
		auto is_write_vsp = []( const vm_context* vctx, x86_ins ins, size_t disp = 0 ) -> bool {
			return ( ins.getType() == ID_INS_MOV || ins.getType() == ID_INS_MOVZX ) &&
			   ins.operands[0].getType() == OP_MEM &&
			   ins.operands[0].getMemory().getBaseRegister().getId() == vctx->vsp_reg &&
			   ins.operands[0].getMemory().getDisplacement().getValue() == disp;
		};
		
		// sub/add vsp, imm32
		auto is_displace_vsp = []( const vm_context* vctx, x86_ins ins, size_t disp = 4 ) -> bool {
			return ( ins.getType() == ID_INS_ADD || ins.getType() == ID_INS_SUB ) &&
				ins.operands[0].getType() == OP_REG &&
				ins.operands[0].getRegister().getId() == vctx->vsp_reg &&
				ins.operands[1].getType() == OP_IMM &&
				ins.operands[1].getImmediate().getValue() == disp;
		};
		
		// mov/movzx r64, [rip]
		auto is_read_vip = []( const vm_context* vctx, x86_ins ins ) -> bool {
			return ( ins.getType() == ID_INS_MOV || ins.getType() == ID_INS_MOVZX ) &&
				ins.operands[0].getType() == OP_REG &&
				ins.operands[1].getType() == OP_MEM &&
				ins.operands[1].getMemory().getBaseRegister().getId() == vctx->vip_reg;
		};
		
		// mov dword ptr [esp + r64], r64
		auto is_write_vreg = []( const vm_context* vctx, x86_ins ins ) -> bool {
			return ( ins.getType() == ID_INS_MOV || ins.getType() == ID_INS_MOVZX ) &&
				ins.operands[0].getType() == OP_MEM &&
				ins.operands[1].getType() == OP_REG &&
				ins.operands[0].getMemory().getBaseRegister().getId() == ID_REG_X86_ESP &&
				ins.operands[0].getMemory().getIndexRegister().getId() != ID_REG_INVALID;
		};
		
		if ( is_read_vsp( vctx, is[0] ) && 
			is_displace_vsp( vctx, is[1] ) && 
			is_read_vip( vctx, is[2] ) && 
			is_write_vreg( vctx, is[3] ) )
		{
			return arch::VM_POPV;
		}
		
		return arch::VM_UNK;
	}
	
	int process( triton::API* api, vm_context* vctx, bool verbose )
	{
		if ( !vctx->vip )
			return 0;
		
		// get cleaned VM handler
		std::vector is = vmp::deobf( api, vctx, vctx->eip );
		
		// check if handler is valid
		if ( !is.size() )
			return 0;
	
		// identify VM handler type
		arch::handler_t handler_type = vmp::identify( vctx, is );
		if ( handler_type == arch::VM_UNK )
		{
			std::cout << "[!] Unable to classify VM handler at EIP: 0x" << std::hex << vctx->eip << std::endl;
			for ( auto& ins : is )
				std::cout << ins.getDisassembly() << std::endl;
			
			return 0;
		}
		
		// emulate handler with function depending on handler type
		
		arch::vm_ins_t handler( vctx->eip, handler_type );
		// push_back vm_ins_t struct inside of emulate function when we can extract relevant values for each VM instruction
		
		
		return 0; // temporary
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