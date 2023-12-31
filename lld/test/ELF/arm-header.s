# REQUIRES: arm
# RUN: llvm-mc -filetype=obj -triple=armv7aeb-none-linux-gnueabi -mcpu=cortex-a8 %s -o %t.o
# RUN: ld.lld %t.o -o %t1
# RUN: llvm-readobj -h %t1 | FileCheck %s

# RUN: llvm-mc -filetype=obj -triple=armv7aeb-none-linux-gnueabi -mcpu=cortex-a8 %s -o %t.o
# RUN: ld.lld --be8 %t.o -o %t1
# RUN: llvm-readobj -h %t1 | FileCheck -check-prefix=CHECK-EB %s

# CHECK: Format: elf32-bigarm

# CHECK: ElfHeader {
# CHECK-NEXT:   Ident {
# CHECK-NEXT:     Magic: (7F 45 4C 46)
# CHECK-NEXT:     Class: 32-bit (0x1)
# CHECK-NEXT:     DataEncoding: BigEndian (0x2)
# CHECK-NEXT:     FileVersion: 1
# CHECK-NEXT:     OS/ABI: SystemV (0x0)
# CHECK-NEXT:     ABIVersion: 0
# CHECK-NEXT:     Unused: (00 00 00 00 00 00 00)
# CHECK-NEXT:   }

# CHECK:   Flags [ (0x5000200)
# CHECK-NEXT:     0x200
# CHECK-NEXT:     0x1000000
# CHECK-NEXT:     0x4000000
# CHECK-NEXT:   ]

# CHECK-NOT: 0x800000

# CHECK-EB: Format: elf32-bigarm

# CHECK-EB: ElfHeader {
# CHECK-EB-NEXT:   Ident {
# CHECK-EB-NEXT:     Magic: (7F 45 4C 46)
# CHECK-EB-NEXT:     Class: 32-bit (0x1)
# CHECK-EB-NEXT:     DataEncoding: BigEndian (0x2)
# CHECK-EB-NEXT:     FileVersion: 1
# CHECK-EB-NEXT:     OS/ABI: SystemV (0x0)
# CHECK-EB-NEXT:     ABIVersion: 0
# CHECK-EB-NEXT:     Unused: (00 00 00 00 00 00 00)
# CHECK-EB-NEXT:   }

# CHECK-EB:   Flags [ (0x5800200)
# CHECK-EB-NEXT:     0x200
# CHECK-EB-NEXT:     0x800000
# CHECK-EB-NEXT:     0x1000000
# CHECK-EB-NEXT:     0x4000000
# CHECK-EB-NEXT:   ]

