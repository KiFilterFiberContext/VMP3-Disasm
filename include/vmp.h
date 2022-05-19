#pragma once 

#include <triton/api.hpp>
#include <triton/architecture.hpp>
#include <triton/bitsVector.hpp>
#include <triton/config.hpp>
#include <triton/exceptions.hpp>
#include <triton/immediate.hpp>
#include <triton/instruction.hpp>
#include <triton/memoryAccess.hpp>
#include <triton/operandWrapper.hpp>
#include <triton/register.hpp>
#include <triton/x8664Cpu.hpp>
#include <triton/x86Cpu.hpp>
#include <triton/x86Specifications.hpp>

#include <triton/tritonToLLVM.hpp>
#include <triton/llvmToTriton.hpp>

#include <vm_context.h>

namespace vmp
{	
	using x86_ins = triton::arch::Instruction;
	using namespace triton::arch::x86;
	
	static constexpr std::array<triton::arch::x86::instruction_e, 8> BLACKLIST = {
		ID_INS_CLC,
		ID_INS_CMC,
		ID_INS_CMP,
		ID_INS_BTS,
		ID_INS_TEST,
		ID_INS_MOVSX,
		ID_INS_BTR,
		ID_INS_CWDE
	};

	// initialize VM by parsing VMINIT
	void init( triton::API* api, vm_context* vctx );
	
	// process VM handler and return VIP
	void process( triton::API* api, vm_context* vctx, bool clean );
	
	// deobfuscate VM basic block
	std::vector<x86_ins> deobf( triton::API* api, uint64_t block_rva );
}