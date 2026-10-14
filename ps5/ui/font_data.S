// The baked fonts (make_font.py) and the header's logo (make_logo.py),
// linked into the title as they are.
    .section .rodata
    .balign 16
    .global ts5_font_data
    .global ts5_font_data_end
ts5_font_data:
    .incbin "font.bin"
ts5_font_data_end:
    .balign 16
    .global ts5_logo_data
    .global ts5_logo_data_end
ts5_logo_data:
    .incbin "logo.bin"
ts5_logo_data_end:
    .section .note.GNU-stack, "", %progbits
