#include <iostream>
#include <memory>

#include <sstream>
#include <ostream>
#include <exception>
#include <algorithm>
#include <array>

#include <vmp.h>

#include <LIEF/PE.hpp>
#include <LIEF/logging.hpp>

using namespace LIEF::PE;

// https://github.com/JonathanSalwan/VMProtect-devirtualization/blob/main/attack_vmp.py
// https://github.com/JonathanSalwan/Triton/blob/243026c9c1e07a5ca834c4aaf628d1079f6a85ea/src/examples/python/ctf-writeups/google2016-unbreakable/solve.py#L57
// https://triton.quarkslab.com/documentation/doxygen/classtriton_1_1API.html#a98f2f895bead72e105f281431b0d4845
// https://github.com/can1357/NoVmp/blob/486725650cdfc9f9352fbe474015f5a19dd71018/NoVmp/vmprotect/vtil_lifter.cpp

int main(int argc, char** argv)
{
	triton::API ctx( triton::arch::ARCH_X86 );
		
	ctx.setMode( triton::modes::ALIGNED_MEMORY, true );
	ctx.setMode( triton::modes::AST_OPTIMIZATIONS, true );
	ctx.setMode( triton::modes::CONSTANT_FOLDING, true );
	
	if ( argc < 2 )
	{
		std::cout << "USAGE: devirt \"<file>\" [-entry <RVA>]" << std::endl;
		return 1;
	}
		
	// map all binaries are PREFERRED image base 
	// TODO: parse relocs and imports or use PIN to dump memory at entrypoint
	//
	std::unique_ptr<const Binary> binary = Parser::parse( argv[1] );
	if ( !binary || binary->optional_header().magic() != PE_TYPE::PE32 )
	{
		std::cout << "Invalid file: " << argv[1] << std::endl;
		return 2;
	}		
	
	auto image_base = binary->optional_header().imagebase();
	for (const Section& section : binary->sections()) 
	{
		const auto content = section.content();			
		ctx.setConcreteMemoryAreaValue( section.virtual_address() + image_base, content.data(), content.size() );
	}
		
	auto entry_eip = binary->entrypoint();
	if ( argc == 4 && !strcmp( argv[2], "-entry" ) )
		entry_eip = image_base + std::stoul( argv[3], nullptr, 16 );
		
	vmp::vm_context vm_state { .eip = entry_eip };
	
	vmp::init( &ctx, &vm_state );
	if ( !vm_state.vip )
	{
		std::cout << "[!] Failed to parse VMENTER" << std::endl;
		return 1;
	}	
	
	while ( true )
	{
		// unroll vm handler CHECK
		// deobfuscate handler CHECK mostly
		// identify handler 
		// emulate
		
		vmp::process( &ctx, &vm_state, true );
		break;
	}

	return 0;
}
