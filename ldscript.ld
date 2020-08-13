SECTIONS
{
  . = 0x10000;
  .text : { *(.text) }
  .rodata : { *(.rodata) }
  .bss : { *(.bss) }
}

