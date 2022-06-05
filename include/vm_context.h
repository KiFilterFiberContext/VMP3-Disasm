#pragma once

#include <arch.h>

namespace vmp
{
	// represents internal vm state
	struct vm_context 
	{
		triton::arch::register_e vip_reg;
		triton::arch::register_e vsp_reg;
		triton::arch::register_e vrk_reg;
		
		triton::arch::register_e jmp_rva_reg;
		std::vector<vmp::arch::vm_ins_t> handlers;
		
		uint64_t vip;
		
		uint64_t eip;
	};
}