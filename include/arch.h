#pragma once

namespace vmp
{	
	namespace arch
	{
		// represents VM instruction type 
		enum ins_t
		{
			VM_INIT,
			VM_POPD,
			VM_UNK
		};
		
		struct ins 
		{
			ins_t type;
			uint64_t rva;
		};
	}
}