// The G2P's lexicon (scripts/pack-g2p-lexicon.py), linked in as it is.
    .section .rodata
    .balign 16
    .global ts5_lexicon_data
    .global ts5_lexicon_data_end
ts5_lexicon_data:
    .incbin "lexicon.bin"
ts5_lexicon_data_end:
    .section .note.GNU-stack, "", %progbits
