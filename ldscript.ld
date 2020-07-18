SECTIONS
{
  . = 0x10000;
  .text 0x10000 : { *(.text) }
  .rodata : { *(.rodata) }
}

