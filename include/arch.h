#pragma once

namespace vmp
{	
	namespace arch
	{
		// represents VM instruction from stack-machine
		enum ins_t
		{
			VM_PUSH,
			VM_POP,
			VM_EXIT,
			VM_INIT,
			VM_SWAP,
			VM_ADD,
			UNKNOWN
		};
		
		struct ins 
		{
			ins_t type;
			uint64_t rva;
		};
	}
}